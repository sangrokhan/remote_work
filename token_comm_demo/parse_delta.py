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


def parse(path, src, dst):
    """Yield (no, time_s, size) for each line matching src->dst, in file order."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            if m.group("src") == src and m.group("dst") == dst:
                no = int(m.group("no")) if m.group("no") else None
                rows.append((no, to_seconds(m.group("time")), int(m.group("size"))))
    return rows


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
    ap.add_argument("--src", required=True, help="source IP (A)")
    ap.add_argument("--dst", required=True, help="dest IP (B)")
    ap.add_argument("--csv", help="write results to CSV file")
    args = ap.parse_args()

    rows = parse(args.file, args.src, args.dst)
    if not rows:
        print(f"No packets matched {args.src} -> {args.dst}", file=sys.stderr)
        sys.exit(1)

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
