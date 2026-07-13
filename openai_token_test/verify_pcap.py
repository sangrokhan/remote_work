"""Check the socket byte counter against the packet capture.

wire.py counts our own bytes with our own code. That is convenient and it is also
exactly the kind of number that can be quietly wrong. A pcap is an outside
witness: real packets, captured by tcpdump, which has no idea what we claim.

For each captured arm this sums the TCP payload bytes travelling client -> server
and compares them to the arm's wire_sent. They will not match exactly, and should
not:

  pcap  = TLS records: the HTTP bytes plus per-record framing (5-byte header +
          AEAD tag, ~22 B per record) plus retransmissions
  socket= the HTTP bytes handed to the TLS layer

So expect pcap slightly ABOVE socket. A pcap far below socket means the capture
dropped packets; a pcap far above means something else was talking on that
connection. Either way you want to know before you publish the number.

Usage:  python verify_pcap.py [exec_id]     (default: the most recent run)
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import capture as cap_mod
import store

# tcpdump -nn prints "... length 517" for data-bearing packets; pure ACKs are
# "length 0". Without -q, which strips the field entirely — the mistake that made
# the first version of this script report zero for everything.
_LEN = re.compile(r"length (\d+)$")


def pcap_client_bytes(path: Path, ips: list[str]) -> tuple[int, int]:
    """(payload bytes client->server, packet count) for the captured host."""
    if not ips:
        return 0, 0
    expr = f"tcp and ({' or '.join(f'dst host {ip}' for ip in ips)})"
    out = subprocess.run(["tcpdump", "-r", str(path), "-nn", expr],
                         capture_output=True, text=True).stdout
    total = packets = 0
    for line in out.splitlines():
        m = _LEN.search(line.strip())
        if not m:
            continue
        n = int(m.group(1))
        if n:
            total += n
            packets += 1
    return total, packets


def main() -> int:
    exec_id = sys.argv[1] if len(sys.argv) > 1 else None
    if exec_id:
        doc = store.load_run(exec_id)
    else:
        runs = sorted(glob.glob(str(store.RUNS_DIR / "*" / "run.json")))
        doc = json.loads(Path(runs[-1]).read_text()) if runs else None

    if not doc:
        print("no run found")
        return 1
    caps = doc.get("captures") or []
    if not caps:
        print(f"run {doc['exec_id']} has no packet captures — re-run with --capture")
        return 1

    socket_sent: dict[tuple[str, int], int] = {}
    for r in doc["runs"]:
        key = (r["arm"], r["repeat"])
        socket_sent[key] = (sum(t["wire_sent"] for t in r["turns"])
                            + (r["setup"] or {}).get("wire_sent", 0))

    print(f"run {doc['exec_id']} · {doc['config']['model']} · "
          f"{doc['config']['turns']} turns\n")
    hdr = f"{'arm':<22}{'socket':>10}{'pcap':>10}{'pcap/socket':>13}{'pkts':>7}{'drop':>6}"
    print(hdr)
    print("-" * len(hdr))

    ok = True
    for c in caps:
        if not c.get("ok"):
            print(f"{c['arm']:<22}  capture failed: {c.get('error') or c.get('note')}")
            ok = False
            continue
        path = cap_mod.PCAP_DIR / c["file"]
        got, packets = pcap_client_bytes(path, c["ips"])
        s = socket_sent.get((c["arm"], c.get("repeat", 1)), 0)
        ratio = got / s if s else 0
        flag = "" if 1.0 <= ratio <= 1.15 else "  <-- check"
        if flag:
            ok = False
        print(f"{c['arm']:<22}{s:>10,}{got:>10,}{ratio:>12.2f}x{packets:>7}"
              f"{c.get('dropped', 0):>6}{flag}")

    print("\npcap should sit a few percent above socket: TLS adds ~22 B of framing "
          "per record.\nFar below means dropped packets; far above means the "
          "connection carried something else.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
