/* cwnd_monitor -- sample the kernel's congestion state for the sockets talking to an
 * API host, every N milliseconds, as NDJSON on stdout.
 *
 * Why this exists: an LLM turn spends seconds idle waiting for the server to think.
 * With net.ipv4.tcp_slow_start_after_idle=1 -- the default -- the kernel resets the
 * congestion window back to the initial value once the idle gap exceeds one RTO, so
 * the *next* request re-enters slow start and pays extra round trips to re-open the
 * window it had already earned. That is invisible from inside the process: requests
 * has no idea what cwnd is, and a pcap shows the consequence (segments trickling out
 * in bursts of ten) but never the cause. cwnd is kernel state, and this reads it.
 *
 * Why netlink and not getsockopt: getsockopt(TCP_INFO) needs the file descriptor, and
 * the descriptor belongs to the Python process that opened it. sock_diag has no such
 * requirement -- it is what `ss -ti` uses, and `ss` reads congestion state for sockets
 * it does not own, unprivileged, every day. Same uid and same network namespace is all
 * the kernel asks. So the client stays in Python, unmodified, and this watches from
 * outside; nothing about the traffic being measured is perturbed by measuring it.
 *
 * Sampling, not tracing -- and the period has to be chosen against the path's RTT, not
 * against the idle gap. The gap is seconds long and easy to catch. The *reset* is
 * visible only until slow start doubles the window back, which takes a few RTTs. On a
 * 3 ms path -- api.openai.com terminates at a CDN edge one hop away, measured at
 * 3.3 ms -- the climb from 10 segments back to 65 fits inside 10 ms, so a 10 ms
 * sampler steps over the very event it exists to record. Hence 2 ms by default, which
 * only became affordable once a tick stopped being a full table walk: see `struct
 * tracked` below.
 *
 * What sampling still cannot do is catch anything shorter than one period. A tcp_probe
 * eBPF program could, and needs root and a BTF toolchain; this does not.
 *
 * Build:  cc -O2 -Wall -o cwnd_monitor cwnd_monitor.c
 * Usage:  cwnd_monitor --dst 1.2.3.4,5.6.7.8 --port 443 --interval-ms 10 \
 *                      --max-seconds 900 [--label openai:stateless]
 *
 * Stdout is one JSON object per line: a single {"type":"meta"} header, then a
 * {"type":"sample"} per (tick, socket), then a {"type":"end"} trailer. Line buffered,
 * so a reader gets each sample as it happens rather than at exit. Diagnostics go to
 * stderr and never interleave with the data.
 *
 * Exits on SIGTERM/SIGINT (the trailer is still written) or after --max-seconds, which
 * exists so an orphaned monitor cannot outlive the run that started it.
 */

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <linux/inet_diag.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sock_diag.h>
#include <linux/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAX_DSTS 64
#define RECV_BUF (256 * 1024)

/* TCP states we care about. LISTEN and TIME_WAIT are excluded: a listener has no
 * congestion window, and a socket in TIME_WAIT is already finished paying for one.
 * Everything else is on the path from handshake to close, and the reset we are hunting
 * happens in ESTABLISHED.
 *
 * The state numbers are spelled out rather than taken from a header: the kernel's
 * enum lives behind __KERNEL__ in <linux/tcp.h>, and glibc's <netinet/tcp.h> -- which
 * does define them -- collides with <linux/tcp.h> over struct tcp_info, which is the
 * one thing here that must come from the kernel headers. The values are ABI and have
 * not moved since 2.0. */
#define ST_ESTABLISHED  1
#define ST_SYN_SENT     2
#define ST_SYN_RECV     3
#define ST_FIN_WAIT1    4
#define ST_FIN_WAIT2    5
#define ST_CLOSE_WAIT   8
#define ST_LAST_ACK     9
#define ST_CLOSING     11

#define WANTED_STATES ( \
      (1u << ST_ESTABLISHED) | (1u << ST_SYN_SENT)  | (1u << ST_SYN_RECV)   \
    | (1u << ST_FIN_WAIT1)   | (1u << ST_FIN_WAIT2) | (1u << ST_CLOSE_WAIT) \
    | (1u << ST_LAST_ACK)    | (1u << ST_CLOSING))

static const char *STATE_NAMES[] = {
    "UNKNOWN", "ESTABLISHED", "SYN_SENT", "SYN_RECV", "FIN_WAIT1", "FIN_WAIT2",
    "TIME_WAIT", "CLOSE", "CLOSE_WAIT", "LAST_ACK", "LISTEN", "CLOSING",
};

/* tcp_ca_state, from include/net/tcp.h. Not "open" vs "not open": a window that
 * shrank because of loss (Recovery/Loss) and a window that shrank because the
 * connection went quiet (Open, after idle) are different findings, and only the
 * second one is what this tool was built to show. */
static const char *CA_NAMES[] = {"open", "disorder", "cwr", "recovery", "loss"};

struct dst {
    int family;                 /* AF_INET or AF_INET6 */
    unsigned char addr[16];     /* network order; 4 bytes used for v4 */
};

static struct dst g_dsts[MAX_DSTS];
static int g_ndst = 0;
static int g_port = 443;
static const char *g_label = "";

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) { (void)sig; g_stop = 1; }

static const char *state_name(unsigned st)
{
    return st < sizeof(STATE_NAMES) / sizeof(STATE_NAMES[0]) ? STATE_NAMES[st]
                                                             : "UNKNOWN";
}

static const char *ca_name(unsigned ca)
{
    return ca < sizeof(CA_NAMES) / sizeof(CA_NAMES[0]) ? CA_NAMES[ca] : "unknown";
}

static double now_monotonic(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double now_realtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Escape the few characters that would break a JSON string. Labels come from the
 * caller (provider:arm), so they are tame, but a quote in one would silently produce
 * a corrupt line that a parser reports far from here. */
static void json_escape(const char *in, char *out, size_t cap)
{
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 2 < cap; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c >= 0x20) {
            out[o++] = (char)c;
        }
        /* control characters are dropped rather than escaped: nothing legitimate
         * puts one in a label, and \u escaping here would be dead code */
    }
    out[o] = '\0';
}

/* --- destination matching ------------------------------------------------- */

static int parse_dsts(const char *csv)
{
    char *copy = strdup(csv);
    if (!copy) return -1;
    for (char *tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
        while (*tok == ' ') tok++;
        if (!*tok) continue;
        if (g_ndst >= MAX_DSTS) {
            fprintf(stderr, "cwnd_monitor: more than %d destinations, ignoring rest\n",
                    MAX_DSTS);
            break;
        }
        struct dst d;
        memset(&d, 0, sizeof d);
        if (inet_pton(AF_INET, tok, d.addr) == 1) {
            d.family = AF_INET;
        } else if (inet_pton(AF_INET6, tok, d.addr) == 1) {
            d.family = AF_INET6;
        } else {
            fprintf(stderr, "cwnd_monitor: not an IP address: %s\n", tok);
            free(copy);
            return -1;
        }
        g_dsts[g_ndst++] = d;
    }
    free(copy);
    return 0;
}

/* A socket is ours if its peer port matches and -- when destinations were given --
 * its peer address is one of them. With no destinations the port alone decides, which
 * is deliberately loose: a run whose DNS answer changed mid-flight should still be
 * observed, noisily, rather than silently producing an empty CSV.
 *
 * v4-mapped v6 addresses (::ffff:a.b.c.d) are compared against v4 destinations too,
 * because a socket opened to an IPv4 host on a dual-stack box is reported by the
 * kernel in either shape depending on how the client resolved it. */
static int dst_matches(int family, const unsigned char *addr, unsigned port)
{
    if ((int)port != g_port) return 0;
    if (g_ndst == 0) return 1;

    static const unsigned char V4MAP[12] = {0,0,0,0,0,0,0,0,0,0,0xff,0xff};
    const unsigned char *v4 = NULL;
    if (family == AF_INET) {
        v4 = addr;
    } else if (family == AF_INET6 && memcmp(addr, V4MAP, sizeof V4MAP) == 0) {
        v4 = addr + 12;
    }

    for (int i = 0; i < g_ndst; i++) {
        const struct dst *d = &g_dsts[i];
        if (d->family == family &&
            memcmp(d->addr, addr, family == AF_INET ? 4 : 16) == 0)
            return 1;
        if (d->family == AF_INET && v4 && memcmp(d->addr, v4, 4) == 0)
            return 1;
    }
    return 0;
}

/* --- tracked sockets ------------------------------------------------------ */

/* The sockets we have already found, so later ticks can ask for them by name instead
 * of asking for everything.
 *
 * This is the difference between a tick that costs 2.4 ms and one that costs 3 us.
 * A dump makes the kernel walk the entire established hash table -- every bucket,
 * sized from system memory, not from how many sockets are actually open -- and hand
 * back every TCP socket for us to filter in userspace. Measured on a 16 GB box with
 * 25 sockets open: 2410 us per dump, and identical whether 11 sockets matched the
 * filter or none did, because the walk is the cost, not the answer.
 *
 * The same request carrying a 4-tuple and *without* NLM_F_DUMP is a hash lookup:
 * 3.0 us, measured over 2000 queries on the same box. 800x. That is what makes a 2 ms
 * period affordable, and 2 ms is what it takes to see an idle reset on a path whose
 * RTT is 3 ms -- there the restarted window doubles back out of sight inside a single
 * 10 ms sample, so the event is real and the sampler simply steps over it.
 *
 * The cookie is kept and sent back with each query. It identifies the socket
 * *instance*, so when a connection closes and the kernel hands its ephemeral port to
 * something else, the answer is ENOENT rather than a stranger's congestion window
 * filed under the old socket's name.
 */
#define MAX_TRACKED 64

/* How often to dump anyway, to notice sockets that appeared since the last one. Every
 * arm opens a fresh connection (the runner drops the pooled one so each arm starts
 * from a handshake), and a socket nobody is tracking is a socket nobody is measuring.
 * At 2410 us per dump this costs about 5% of a core -- still fifteen times cheaper
 * than dumping every tick, and it bounds how long a new connection can go unseen. */
#define REDISCOVER_SECONDS 0.100

/* How soon a dump may follow the discovery that a followed socket has gone.
 *
 * Losing a socket is the one moment worth looking again quickly: the connection that
 * replaces it opens with a handshake and an initial window, which is the part worth
 * seeing. But "look again immediately" turns into a loop during teardown -- a closing
 * socket sits in FIN_WAIT long enough to be re-found by the dump, fails its exact query
 * on the next tick, forces another dump, and so on. Measured: 117 dumps in 1017 ticks
 * across three reconnects, against 20 in steady state. A floor of 20 ms bounds that at
 * one dump per 10 ticks while still picking a new connection up almost at once. */
#define REDISCOVER_AFTER_LOSS 0.020

struct tracked {
    int family;
    struct inet_diag_sockid id;
    int have_cookie;    /* the kernel's, learned from the first reply */
};

static struct tracked g_tracked[MAX_TRACKED];
static int g_ntracked = 0;
static unsigned long g_announced = 0;   /* four-tuples the client handed us */

/* Sockets announced on stdin, before the kernel has been asked about anything.
 *
 * Discovery by dump has a floor: a socket born after the last dump is invisible until
 * the next one. Measured with a 100 ms rediscovery timer, five connections that pushed
 * data as soon as they opened: every one of them was noticed between 4 and 22 ms late,
 * and by then the window had already climbed from 10 to somewhere between 48 and 75.
 * The initial congestion window -- the thing a reset returns the connection *to* -- was
 * never once recorded. A pcap holding the three-way handshake beside a cwnd log holding
 * nothing is what that looks like from the outside.
 *
 * Shortening the timer does not fix it: a dump costs 2.4 ms, so a 10 ms timer spends a
 * quarter of a core and still arrives late on a 3 ms path.
 *
 * The client, though, knows the four-tuple the instant connect() returns -- core.wire
 * already wraps the socket to count its bytes. So it says so, on this process's stdin,
 * and the very next tick queries the new socket directly. No dump, no lag.
 *
 * Announced sockets carry no cookie: the client knows the addresses, not the kernel's
 * instance id. INET_DIAG_NOCOOKIE tells the kernel to match on the four-tuple alone,
 * and the first reply carries the real cookie, which is kept from then on -- so the
 * window where a recycled port could alias is one query wide, not the whole run.
 */
static void track_add_announced(int family, const struct inet_diag_sockid *id);

static void track_add(int family, const struct inet_diag_sockid *id)
{
    for (int i = 0; i < g_ntracked; i++) {
        if (g_tracked[i].family == family &&
            g_tracked[i].id.idiag_sport == id->idiag_sport &&
            g_tracked[i].id.idiag_dport == id->idiag_dport &&
            memcmp(g_tracked[i].id.idiag_src, id->idiag_src,
                   sizeof id->idiag_src) == 0 &&
            memcmp(g_tracked[i].id.idiag_dst, id->idiag_dst,
                   sizeof id->idiag_dst) == 0)
            return;
    }
    if (g_ntracked >= MAX_TRACKED) return;
    g_tracked[g_ntracked].family = family;
    g_tracked[g_ntracked].id = *id;
    g_tracked[g_ntracked].have_cookie = 1;   /* a dump answer carries the real one */
    g_ntracked++;
}

static void track_add_announced(int family, const struct inet_diag_sockid *id)
{
    int before = g_ntracked;
    track_add(family, id);
    if (g_ntracked > before) {
        g_tracked[g_ntracked - 1].have_cookie = 0;
        g_announced++;
    }
}

/* Parse one announcement: "track <src-ip> <sport> <dst-ip> <dport>".
 *
 * Deliberately dumb. This reads a pipe from the process that started it, not a network,
 * and a line it cannot parse is dropped with a note rather than treated as an attack or
 * as a reason to stop sampling.
 */
static void handle_command(char *line)
{
    char verb[16], src[64], dst[64];
    unsigned sport, dport;
    if (sscanf(line, "%15s %63s %u %63s %u", verb, src, &sport, dst, &dport) != 5) {
        fprintf(stderr, "cwnd_monitor: cannot parse command: %s\n", line);
        return;
    }
    if (strcmp(verb, "track") != 0) {
        fprintf(stderr, "cwnd_monitor: unknown command: %s\n", verb);
        return;
    }

    struct inet_diag_sockid id;
    memset(&id, 0, sizeof id);
    int family;
    if (inet_pton(AF_INET, src, id.idiag_src) == 1 &&
        inet_pton(AF_INET, dst, id.idiag_dst) == 1) {
        family = AF_INET;
    } else if (inet_pton(AF_INET6, src, id.idiag_src) == 1 &&
               inet_pton(AF_INET6, dst, id.idiag_dst) == 1) {
        family = AF_INET6;
    } else {
        fprintf(stderr, "cwnd_monitor: not an address pair: %s / %s\n", src, dst);
        return;
    }
    id.idiag_sport = htons((uint16_t)sport);
    id.idiag_dport = htons((uint16_t)dport);
    id.idiag_cookie[0] = INET_DIAG_NOCOOKIE;
    id.idiag_cookie[1] = INET_DIAG_NOCOOKIE;
    track_add_announced(family, &id);
}

/* Drain whatever announcements are waiting, without blocking.
 *
 * Called once per tick. Blocking here would hand the client the power to stall the
 * sampler by not writing, which is the opposite of the arrangement: the client is
 * being measured, not trusted to keep time. */
static void read_commands(void)
{
    static char pending[512];
    static size_t plen = 0;

    for (;;) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        if (poll(&pfd, 1, 0) <= 0) return;
        if (!(pfd.revents & POLLIN)) return;

        char buf[512];
        ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
        if (n <= 0) return;                  /* EOF: the client said all it will say */

        for (ssize_t i = 0; i < n; i++) {
            if (buf[i] == '\n') {
                pending[plen] = '\0';
                if (plen) handle_command(pending);
                plen = 0;
            } else if (plen + 1 < sizeof pending) {
                pending[plen++] = buf[i];
            } else {
                plen = 0;                    /* overlong line: drop it whole */
            }
        }
    }
}

/* --- netlink -------------------------------------------------------------- */

static int diag_open(void)
{
    int fd = socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_SOCK_DIAG);
    if (fd < 0) return -1;

    /* A dump of every TCP socket on a busy box is large, and a short receive buffer
     * turns that into ENOBUFS -- which arrives as a lost dump, not an error at the
     * call site. Ask for room; ignore failure, since the default is usually enough. */
    int rcvbuf = RECV_BUF;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);

    /* Never block forever. If the kernel goes quiet we want to miss a tick and carry
     * on, not hang holding a run open. */
    struct timeval tv = {.tv_sec = 1, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return fd;
}

/* One request. With `id`, it asks for that one socket and the kernel does a hash
 * lookup; without, it is a dump and the kernel walks the table. Same message type
 * either way -- NLM_F_DUMP and a filled-in sockid are the whole difference. */
static int diag_request(int fd, int family, uint32_t seq,
                        const struct inet_diag_sockid *id)
{
    struct {
        struct nlmsghdr nlh;
        struct inet_diag_req_v2 req;
    } msg;
    memset(&msg, 0, sizeof msg);

    msg.nlh.nlmsg_len = sizeof msg;
    msg.nlh.nlmsg_type = SOCK_DIAG_BY_FAMILY;
    msg.nlh.nlmsg_flags = NLM_F_REQUEST | (id ? 0 : NLM_F_DUMP);
    msg.nlh.nlmsg_seq = seq;

    msg.req.sdiag_family = (uint8_t)family;
    msg.req.sdiag_protocol = IPPROTO_TCP;
    msg.req.idiag_ext = (1 << (INET_DIAG_INFO - 1));
    if (id) {
        msg.req.id = *id;
        /* Every state, for an exact query: the socket we are following may have moved
         * to FIN_WAIT or CLOSE_WAIT since we found it, and a state mask that excluded
         * where it went would read as "gone" and trigger a needless rediscovery. */
        msg.req.idiag_states = 0xffffffffu;
    } else {
        msg.req.idiag_states = WANTED_STATES;
    }

    struct sockaddr_nl nladdr = {.nl_family = AF_NETLINK};
    return sendto(fd, &msg, sizeof msg, 0, (struct sockaddr *)&nladdr, sizeof nladdr)
           < 0 ? -1 : 0;
}

static void fmt_addr(int family, const unsigned char *raw, char *out, size_t cap)
{
    if (!inet_ntop(family, raw, out, (socklen_t)cap))
        snprintf(out, cap, "?");
}

/* One socket, one line. `info` has already been copied into a full-size, zeroed
 * struct tcp_info by the caller, so every field below is safe to read even on a
 * kernel whose tcp_info is shorter than ours -- the missing tail reads as zero, and
 * info_len in the meta line tells the reader how much was real. */
static void emit_sample(double t_rel, double wall, int family,
                        const struct inet_diag_msg *m,
                        const struct tcp_info *ti)
{
    char lbuf[INET6_ADDRSTRLEN], rbuf[INET6_ADDRSTRLEN];
    fmt_addr(family, (const unsigned char *)m->id.idiag_src, lbuf, sizeof lbuf);
    fmt_addr(family, (const unsigned char *)m->id.idiag_dst, rbuf, sizeof rbuf);

    printf("{\"type\":\"sample\""
           ",\"t_ms\":%.3f,\"wall\":%.6f"
           ",\"local\":\"%s:%u\",\"remote\":\"%s:%u\""
           ",\"state\":\"%s\",\"ca_state\":\"%s\""
           ",\"snd_cwnd\":%u,\"snd_ssthresh\":%u,\"rcv_ssthresh\":%u"
           ",\"rtt_us\":%u,\"rttvar_us\":%u,\"min_rtt_us\":%u"
           ",\"snd_mss\":%u,\"rcv_mss\":%u,\"advmss\":%u,\"pmtu\":%u"
           ",\"unacked\":%u,\"sacked\":%u,\"lost\":%u,\"retrans\":%u"
           ",\"total_retrans\":%u,\"reordering\":%u"
           ",\"bytes_sent\":%llu,\"bytes_acked\":%llu,\"bytes_received\":%llu"
           ",\"bytes_retrans\":%llu"
           ",\"segs_out\":%u,\"segs_in\":%u"
           ",\"delivered\":%u,\"delivery_rate\":%llu,\"pacing_rate\":%llu"
           ",\"snd_wnd\":%u,\"rwnd_limited_us\":%llu,\"sndbuf_limited_us\":%llu"
           ",\"busy_time_us\":%llu"
           ",\"last_data_sent_ms\":%u,\"last_data_recv_ms\":%u,\"last_ack_recv_ms\":%u"
           ",\"rto_us\":%u,\"ato_us\":%u"
           ",\"inode\":%u}\n",
           t_rel * 1000.0, wall,
           lbuf, ntohs(m->id.idiag_sport), rbuf, ntohs(m->id.idiag_dport),
           state_name(m->idiag_state), ca_name(ti->tcpi_ca_state),
           ti->tcpi_snd_cwnd, ti->tcpi_snd_ssthresh, ti->tcpi_rcv_ssthresh,
           ti->tcpi_rtt, ti->tcpi_rttvar, ti->tcpi_min_rtt,
           ti->tcpi_snd_mss, ti->tcpi_rcv_mss, ti->tcpi_advmss, ti->tcpi_pmtu,
           ti->tcpi_unacked, ti->tcpi_sacked, ti->tcpi_lost, ti->tcpi_retrans,
           ti->tcpi_total_retrans, ti->tcpi_reordering,
           (unsigned long long)ti->tcpi_bytes_sent,
           (unsigned long long)ti->tcpi_bytes_acked,
           (unsigned long long)ti->tcpi_bytes_received,
           (unsigned long long)ti->tcpi_bytes_retrans,
           ti->tcpi_segs_out, ti->tcpi_segs_in,
           ti->tcpi_delivered,
           (unsigned long long)ti->tcpi_delivery_rate,
           (unsigned long long)ti->tcpi_pacing_rate,
           ti->tcpi_snd_wnd,
           (unsigned long long)ti->tcpi_rwnd_limited,
           (unsigned long long)ti->tcpi_sndbuf_limited,
           (unsigned long long)ti->tcpi_busy_time,
           ti->tcpi_last_data_sent, ti->tcpi_last_data_recv, ti->tcpi_last_ack_recv,
           ti->tcpi_rto, ti->tcpi_ato,
           m->idiag_inode);
}

/* Read the answer to one request and emit what it holds.
 *
 * `is_dump` says which shape to expect. A dump is multipart and ends with NLMSG_DONE,
 * and every socket in it has to be filtered against the destination. An exact reply is
 * one message, has no NLMSG_DONE to wait for -- waiting would block for the receive
 * timeout on every single tick -- and needs no filtering, because the 4-tuple was the
 * question. On a dump, matching sockets are also remembered, so subsequent ticks can
 * ask for them directly.
 *
 * Returns the number of sockets emitted; 0 when an exact query found nothing (the
 * socket closed, or its port was recycled and the cookie no longer matches), or -1
 * when the answer could not be read at all.
 */
static int diag_drain(int fd, int family, double t_rel, double wall,
                      char *buf, size_t buflen, int *info_len_out, int is_dump,
                      struct tracked *asked_about)
{
    int emitted = 0;

    for (;;) {
        ssize_t n = recv(fd, buf, buflen, 0);
        if (n < 0) {
            if (errno == EINTR) {
                if (g_stop) return emitted;
                continue;
            }
            return emitted > 0 ? emitted : -1;
        }
        if (n == 0) return emitted;

        for (struct nlmsghdr *h = (struct nlmsghdr *)buf;
             NLMSG_OK(h, (unsigned)n); h = NLMSG_NEXT(h, n)) {

            if (h->nlmsg_type == NLMSG_DONE) return emitted;
            if (h->nlmsg_type == NLMSG_ERROR) {
                struct nlmsgerr *e = (struct nlmsgerr *)NLMSG_DATA(h);
                /* ENOENT on an exact query is the ordinary way a followed socket ends.
                 * Reporting it would put one line on stderr per tick per closed
                 * connection, which is how a diagnostic channel becomes unreadable. */
                if (!(!is_dump && e->error == -ENOENT))
                    fprintf(stderr, "cwnd_monitor: netlink error %d (family %d)\n",
                            e->error, family);
                return is_dump ? -1 : 0;
            }

            struct inet_diag_msg *m = (struct inet_diag_msg *)NLMSG_DATA(h);
            if (!is_dump && asked_about && !asked_about->have_cookie) {
                /* An announced socket was matched on its four-tuple alone. Now that
                 * the kernel has named the instance, keep the name: a later query
                 * then cannot be answered by whatever inherits the port. */
                memcpy(asked_about->id.idiag_cookie, m->id.idiag_cookie,
                       sizeof m->id.idiag_cookie);
                asked_about->have_cookie = 1;
            }
            if (is_dump) {
                unsigned dport = ntohs(m->id.idiag_dport);
                if (!dst_matches(family, (const unsigned char *)m->id.idiag_dst, dport))
                    continue;
                track_add(family, &m->id);
            }

            /* Zeroed full-size copy: a kernel older than this build's headers returns
             * a shorter tcp_info, and reading past it would be a genuine overread. */
            struct tcp_info ti;
            memset(&ti, 0, sizeof ti);
            int have_info = 0;

            int rtalen = (int)(h->nlmsg_len - NLMSG_LENGTH(sizeof(*m)));
            for (struct rtattr *a = (struct rtattr *)(m + 1);
                 RTA_OK(a, rtalen); a = RTA_NEXT(a, rtalen)) {
                if (a->rta_type != INET_DIAG_INFO) continue;
                int len = (int)RTA_PAYLOAD(a);
                if (info_len_out) *info_len_out = len;
                if (len > (int)sizeof ti) len = (int)sizeof ti;
                memcpy(&ti, RTA_DATA(a), (size_t)len);
                have_info = 1;
            }
            if (!have_info) continue;   /* a socket with no INET_DIAG_INFO has nothing to say */

            emit_sample(t_rel, wall, family, m, &ti);
            emitted++;
        }
        /* A single-part reply is complete when its one message has been read. Looping
         * for a NLMSG_DONE that is never coming would stall the tick until the receive
         * timeout, which at a 2 ms period is the whole run. */
        if (!is_dump) return emitted;
    }
}

/* --- main ----------------------------------------------------------------- */

static void usage(void)
{
    fprintf(stderr,
        "usage: cwnd_monitor [--dst IP[,IP...]] [--port N] [--interval-ms N]\n"
        "                    [--max-seconds N] [--label STR]\n");
}

int main(int argc, char **argv)
{
    /* 2 ms, not 10: on a path whose RTT is 3 ms a restarted window is back where it
     * started within one 10 ms sample. Affordable only because a tick is now hash
     * lookups rather than a table walk. */
    long interval_ms = 2;
    /* 0 = run until signalled. Fractional, so a caller can ask for a 20 ms run purely
     * to find out whether netlink answers at all -- which is how core.cwnd decides
     * whether to offer monitoring, and it cannot spend a second doing it. */
    double max_seconds = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--dst") && v)              { if (parse_dsts(v)) return 2; i++; }
        else if (!strcmp(a, "--port") && v)        { g_port = atoi(v); i++; }
        else if (!strcmp(a, "--interval-ms") && v) { interval_ms = atol(v); i++; }
        else if (!strcmp(a, "--max-seconds") && v) { max_seconds = atof(v); i++; }
        else if (!strcmp(a, "--label") && v)       { g_label = v; i++; }
        else { usage(); return 2; }
    }
    if (interval_ms < 1) interval_ms = 1;

    /* Line buffered, so the reader on the other end of the pipe sees each sample as it
     * is taken. Full buffering would deliver the whole run in one burst at exit and
     * make live progress impossible. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);   /* the reader may go away first; that is not a crash */

    int fd = diag_open();
    if (fd < 0) {
        fprintf(stderr, "cwnd_monitor: netlink open failed: %s\n", strerror(errno));
        return 1;
    }

    char *buf = malloc(RECV_BUF);
    if (!buf) {
        fprintf(stderr, "cwnd_monitor: out of memory\n");
        close(fd);
        return 1;
    }

    char label[256];
    json_escape(g_label, label, sizeof label);

    int info_len = 0;
    double t0 = now_monotonic();

    printf("{\"type\":\"meta\",\"label\":\"%s\",\"port\":%d,\"dsts\":%d"
           ",\"interval_ms\":%ld,\"pid\":%d,\"wall_start\":%.6f"
           ",\"tcp_info_build\":%zu}\n",
           label, g_port, g_ndst, interval_ms, (int)getpid(), now_realtime(),
           sizeof(struct tcp_info));

    /* Absolute-deadline sleeping, so the period does not drift by however long a dump
     * took. A relative sleep of 10 ms after 3 ms of work gives a 13 ms period, and over
     * a multi-minute run that skew would misplace the idle gap this tool exists to
     * time. */
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    unsigned long ticks = 0, samples = 0, dumps = 0, exacts = 0;
    uint32_t seq = 1;
    double last_dump = -1e9;

    while (!g_stop) {
        double t = now_monotonic();
        double t_rel = t - t0;
        if (max_seconds > 0 && t_rel >= max_seconds) break;

        /* Announcements first, so a socket the client just opened is queried on
         * this tick rather than after the next dump. That gap is where the initial
         * congestion window lives. */
        read_commands();

        double wall = now_realtime();
        int got = 0;

        /* Dump only to discover, and never more often than the timer allows -- not
         * even when nothing is being followed. "Nothing to follow" is a normal state,
         * not an emergency: it is every gap between arms, the whole of a mock run, and
         * the moment after a connection closes. Letting an empty table bypass the
         * timer meant dumping flat out through all of them, which is 2.4ms of kernel
         * work per 2ms tick -- a monitor that costs the most exactly when it has
         * nothing to measure. Every other tick is hash lookups. */
        int need_dump = (t - last_dump >= REDISCOVER_SECONDS);

        if (need_dump) {
            g_ntracked = 0;
            for (int fi = 0; fi < 2; fi++) {
                int family = fi == 0 ? AF_INET : AF_INET6;
                if (diag_request(fd, family, seq++, NULL) < 0) continue;
                int n = diag_drain(fd, family, t_rel, wall, buf, RECV_BUF,
                                   &info_len, 1, NULL);
                if (n > 0) got += n;
            }
            last_dump = t;
            dumps++;
        } else {
            /* Copy the table first: a query that comes back empty drops that socket,
             * and rewriting the array underneath the loop would skip its neighbour. */
            struct tracked live[MAX_TRACKED];
            int nlive = 0;
            for (int i = 0; i < g_ntracked; i++) {
                if (diag_request(fd, g_tracked[i].family, seq++,
                                 &g_tracked[i].id) < 0)
                    continue;
                int n = diag_drain(fd, g_tracked[i].family, t_rel, wall, buf,
                                   RECV_BUF, &info_len, 0, &g_tracked[i]);
                exacts++;
                if (n > 0) {
                    got += n;
                    live[nlive++] = g_tracked[i];
                }
            }
            if (nlive != g_ntracked) {
                /* Something closed. Keep what is left for this tick and rediscover on
                 * the next one -- a closed connection is often replaced immediately,
                 * and the replacement's handshake and initial window are the part
                 * worth seeing. */
                memcpy(g_tracked, live, (size_t)nlive * sizeof *live);
                g_ntracked = nlive;
                /* Look again soon, but not on the very next tick -- see
                 * REDISCOVER_AFTER_LOSS. Never push the deadline later than it
                 * already was. */
                double soon = t + REDISCOVER_AFTER_LOSS - REDISCOVER_SECONDS;
                if (soon < last_dump) last_dump = soon;
            }
        }
        samples += (unsigned)got;
        ticks++;

        next.tv_nsec += (long)(interval_ms * 1000000L);
        while (next.tv_nsec >= 1000000000L) {
            next.tv_nsec -= 1000000000L;
            next.tv_sec++;
        }
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL) == EINTR
               && !g_stop)
            ;
    }

    /* dumps and exact_queries are in the trailer because a run's cost is
     * dumps*2.4ms + exacts*3us. A monitor that ended up dumping every tick -- because
     * the socket it followed kept being replaced -- should show up as a number here
     * rather than as unexplained CPU.  */
    printf("{\"type\":\"end\",\"ticks\":%lu,\"samples\":%lu,\"seconds\":%.3f"
           ",\"dumps\":%lu,\"exact_queries\":%lu,\"tracked\":%d"
           ",\"announced\":%lu"
           ",\"tcp_info_len\":%d}\n",
           ticks, samples, now_monotonic() - t0, dumps, exacts, g_ntracked,
           g_announced, info_len);
    fflush(stdout);

    free(buf);
    close(fd);
    return 0;
}
