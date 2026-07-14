"""Run the lab from a terminal, and see the bill before it is paid.

The web UI is for reading a run; this is for launching one on a machine that has the
network but no browser. It exists mostly so the confirmation step exists: `--dry-run`
is the default, because two paid APIs sit behind a monthly cap and a comparison run
across six Gemini arms and three OpenAI ones is between fifty and a hundred calls.
Nothing goes out until `--go` says so.

    python cli.py                              # what would run, and what it would cost
    python cli.py --go --measure bytes         # run it
    python cli.py --go --providers gemini --arms stateless,cached --turns 3
    python cli.py --serve                      # the web UI instead
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import capture as pcap        # noqa: E402
from core import config, metrics, runner, scenario, store  # noqa: E402
from providers import base              # noqa: E402


def _selection(args) -> dict | None:
    if not args.providers:
        return None
    arms = [a.strip() for a in (args.arms or "").split(",") if a.strip()]
    names = [p.strip() for p in args.providers.split(",") if p.strip()]
    if arms and len(names) > 1:
        raise SystemExit("--arms names arms of one provider; pass one --providers")
    return {n: (arms or None) for n in names}


def _progress(event: dict) -> None:
    if event.get("phase") == "pause":
        print(f"  … pausing {event['remaining']}s", end="\r", file=sys.stderr)
        return
    print(f"  {event['provider']}:{event['arm']} [{event['phase']}] "
          f"turn {event['turn']}/{event['turns']}", file=sys.stderr)


def _report(run: dict) -> None:
    summary = run["summary"]
    print(f"\nexec_id: {run['exec_id']}  measure: {run['params']['measure']}"
          f"{'  (MOCK)' if run.get('mock') else ''}\n")
    head = f"{'arm':<32} {'up B':>9} {'down B':>9} {'in tok':>8} {'ttft ms':>8} {'tail ms':>8}"
    print(head)
    print("-" * len(head))
    for key in summary["keys"]:
        t = summary["totals"][key]
        marks = t["marks"]
        print(f"{key:<32} {t['wire_sent']:>9} {t['wire_recv']:>9} "
              f"{t['input_tokens']:>8} {marks['ttft_ms']['median']:>8} "
              f"{marks['store_tail_ms']['median']:>8}")
    for key, p in (summary.get("prep") or {}).items():
        print(f"  prep {key}: {p['calls']} call(s), {p['wire_sent']} B up "
              f"— excluded from the totals above")
        # One line per kind. A prep phase is not one kind of call, and a single rolled-up
        # row invites a reader to add an inference call's input tokens to a cache's size.
        for b in p.get("by_kind") or []:
            cost = (f"{b['input_tokens']} in tok billed" if b.get("billed")
                    else "0 tok billed")
            size = f", cache {b['cache_tokens']} tok" if b.get("cache_tokens") else ""
            print(f"      {b['kind']:<20} {b['calls']} call(s), "
                  f"{b['wire_sent']} B up, {cost}{size}")
            if b.get("note"):
                print(f"      {'':<20} └ {b['note']}")
    for f in summary["failures"]:
        print(f"  ! {f['key']} turn {f['turn']}: {f['error']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--providers", help="comma-separated; default: all")
    ap.add_argument("--arms", help="comma-separated arms of a single provider")
    ap.add_argument("--measure", default="bytes", choices=runner.MEASURES)
    ap.add_argument("--fixture", default=scenario.DEFAULT, choices=scenario.names())
    ap.add_argument("--turns", type=int, help="truncate the thread")
    ap.add_argument("--capture", action="store_true", help="pcap per arm (needs tcpdump)")
    ap.add_argument("--no-cache-bust", dest="cache_bust", action="store_false",
                    default=None,
                    help="let the arms share a prefix, so each one can be answered "
                         "from the cache the last one left warm (default: off)")
    ap.add_argument("--pause", type=float, default=0, metavar="SEC",
                    help="wait between arms, to stay under a rate limit")
    ap.add_argument("--go", action="store_true",
                    help="actually run; without it, only report what would be called")
    ap.add_argument("--serve", action="store_true", help="start the web UI instead")
    args = ap.parse_args(argv)

    if args.serve:
        from core import app as web
        web.main()
        return 0

    providers = _selection(args)
    pairs = runner.plan(providers)
    fixture = scenario.load(args.fixture, args.turns)
    passes = 2 if args.measure == "both" else 1
    turns = len(fixture["steps"])
    calls = len(pairs) * turns * passes
    # Per provider, not per run: with one mocked and one live, a single flag either
    # hides a real bill or invents one. A run with any synthetic call in it is filed as
    # mock, because its numbers cannot be charted against measured ones.
    billable = sum(turns * passes for p, _ in pairs if not config.is_mock(p))
    mock = any(config.is_mock(p) for p, _ in pairs)

    print(f"fixture {fixture['name']}: {turns} turns")
    print(f"arms: {', '.join(f'{p}:{a}' for p, a in pairs)}")
    print(f"measure: {args.measure} → {calls} API call(s), {billable} billable"
          f"{' (MOCK)' if mock else ''}")
    for warning in runner.warnings_for(pairs, args.measure):
        print(f"⚠ {warning}")
    for name in {p for p, _ in pairs}:
        ok, reason = base.get(name).ready()
        if not ok:
            print(f"⚠ {name} not ready: {reason}")
    if args.capture:
        ok, reason = pcap.available()
        print(f"capture: {'ready' if ok else f'unavailable — {reason}'}")

    if not args.go:
        print("\nDry run. Nothing was called. Add --go to run it.")
        return 0

    timestamp = datetime.now(timezone.utc).isoformat()
    run = runner.run(providers, system=fixture["system"], steps=fixture["steps"],
                     measure=args.measure, want_capture=args.capture,
                     pause_seconds=args.pause, timestamp=timestamp,
                     cache_bust=args.cache_bust, on_progress=_progress)
    run["timestamp"] = timestamp
    run["mock"] = mock
    run["params"]["fixture"] = fixture["name"]
    run["summary"] = metrics.summarize(run)
    saved = store.save_run(run)
    run["exec_id"] = saved["exec_id"]

    _report(run)
    print(f"\nsaved: {saved['path']}")
    return 1 if run["summary"]["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
