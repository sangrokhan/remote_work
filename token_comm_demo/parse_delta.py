#!/usr/bin/env python3
"""Parse Wireshark 'packet list as text' export.

Extract packets matching source A -> dest B (by IP), compute delta time
in MICROSECONDS between consecutive A->B packets.

Export source in Wireshark:
  File > Export Packet Dissections > As Plain Text...
  (Packet summary line only; columns: No. Time Source Destination Protocol Length Info)

Usage:
  python3 parse_delta.py capture.txt --src 192.168.1.10 --dst 192.168.1.20
  python3 parse_delta.py capture.txt --src A --dst B --csv out.csv
"""
import argparse
import csv
import re
import sys

# IPv4 (good enough for capture exports).
IPV4 = r"\d{1,3}(?:\.\d{1,3}){3}"
# Time: clock HH:MM:SS.ffffff  OR  plain decimal seconds.
TIME = r"(?:\d{1,2}:\d{2}:\d{2}\.\d+|\d+\.\d+)"
LINE_RE = re.compile(
    r"^\s*(?:(?P<no>\d+)\s+)?"            # optional packet number
    r"(?P<time>" + TIME + r")\s+"        # time (clock or seconds)
    r"(?P<src>" + IPV4 + r")\s+"          # source IP
    r"(?P<dst>" + IPV4 + r")\s+"          # dest IP
    r"(?P<proto>\S+)\s+"                  # protocol
    r"(?P<size>\d+)\b"                    # packet size (bytes)
)


def to_seconds(ts):
    """Convert 'HH:MM:SS.ffffff' or 'SSS.ffffff' to float seconds."""
    if ":" in ts:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(ts)


def parse_all(path):
    """Yield (no, time_s, size, src, dst) for every parseable line, in order."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            no = int(m.group("no")) if m.group("no") else None
            rows.append((no, to_seconds(m.group("time")), int(m.group("size")),
                         m.group("src"), m.group("dst")))
    return rows


def dominant_pair(rows):
    """Return (src, dst) of the most frequent directed IP pair."""
    counts = {}
    for _, _, _, src, dst in rows:
        counts[(src, dst)] = counts.get((src, dst), 0) + 1
    return max(counts, key=counts.get)


def parse(path, src, dst):
    """Return (no, time_s, size) rows matching src->dst, plus resolved (src,dst).

    If src or dst is None, auto-detect the dominant directed pair.
    """
    allrows = parse_all(path)
    if src is None or dst is None:
        if not allrows:
            return [], (src, dst)
        src, dst = dominant_pair(allrows)
    rows = [(no, t, size) for no, t, size, s, d in allrows if s == src and d == dst]
    return rows, (src, dst)


def deltas(rows):
    """Return list of (no, time_s, delta_us, size). First packet delta = None."""
    out = []
    prev = None
    for no, t, size in rows:
        dt = None if prev is None else (t - prev) + (86400 if prev is not None and t < prev else 0)
        d = None if dt is None else round(dt * 1_000_000, 3)
        out.append((no, t, d, size))
        prev = t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="Wireshark plain-text packet-list export")
    ap.add_argument("--src", help="source IP (A); auto-detected if omitted")
    ap.add_argument("--dst", help="dest IP (B); auto-detected if omitted")
    ap.add_argument("--csv", help="write results to CSV file")
    args = ap.parse_args()

    rows, (src, dst) = parse(args.file, args.src, args.dst)
    if not rows:
        print(f"No packets matched {src} -> {dst}", file=sys.stderr)
        sys.exit(1)
    print(f"src={src}  dst={dst}", file=sys.stderr)

    result = deltas(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["no", "time_s", "delta_us", "size"])
            for no, t, d, size in result:
                w.writerow([no, f"{t:.6f}", "" if d is None else f"{d:.3f}", size])
        print(f"Wrote {len(result)} rows -> {args.csv}")
    else:
        print(f"{'No':>7}  {'Time(s)':>14}  {'Delta(us)':>14}  {'Size':>7}")
        for i, (no, t, d, size) in enumerate(result, 1):
            ds = "" if d is None else f"{d:,.3f}"
            print(f"{no if no is not None else i:>7}  {t:>14.6f}  {ds:>14}  {size:>7}")

    # stats over deltas (skip first None) and sizes
    vals = [d for _, _, d, _ in result if d is not None]
    sizes = [s for _, _, _, s in result]
    if vals:
        print(f"\nmatched={len(result)}  deltas={len(vals)}  "
              f"min={min(vals):,.3f}us  max={max(vals):,.3f}us  "
              f"mean={sum(vals)/len(vals):,.3f}us", file=sys.stderr)
    if sizes:
        print(f"bytes total={sum(sizes):,}  min={min(sizes)}  max={max(sizes)}  "
              f"mean={sum(sizes)/len(sizes):,.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
