"""Turn a saved experiment into a CSV and two charts.

The two charts are the argument:

  1. cumulative upload bytes  — stateless curves up, stateful is a flat line
  2. cumulative input tokens  — both curve up, on top of each other

Same conversation, same model. The bytes can be saved; the billing cannot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import metrics

RESULTS_DIR = Path(__file__).parent / "results"

LABELS = {
    "chat_stateless": "chat_stateless (full history)",
    "responses_stateless": "responses_stateless (full history)",
    "responses_stateful": "responses_stateful (server holds it)",
}
COLORS = {
    "chat_stateless": "#c0392b",
    "responses_stateless": "#e67e22",
    "responses_stateful": "#2980b9",
}


def write_csv(summary: dict, path: Path) -> Path:
    rows = []
    for arm, s in summary["arms"].items():
        for k in range(s["turns"]):
            rows.append({
                "arm": arm,
                "turn": k + 1,
                "upload_bytes": round(s["per_turn"]["req_payload_bytes"][k]),
                "wire_sent": round(s["per_turn"]["wire_sent"][k]),
                "download_bytes": round(s["per_turn"]["resp_payload_bytes"][k]),
                "input_tokens": round(s["per_turn"]["input_tokens"][k]),
                "cached_tokens": round(s["per_turn"]["cached_tokens"][k]),
                "billed_uncached_tokens": round(s["per_turn"]["billed_uncached_tokens"][k]),
                "output_tokens": round(s["per_turn"]["output_tokens"][k]),
                "latency_ms": round(s["per_turn"]["latency_ms"][k]),
                "cum_upload_bytes": round(s["cum_req_bytes"][k]),
                "cum_input_tokens": round(s["cum_input_tokens"][k]),
            })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def write_charts(summary: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cfg = summary["config"]

    for arm, s in summary["arms"].items():
        turns = range(1, s["turns"] + 1)
        label = LABELS.get(arm, arm)
        color = COLORS.get(arm)
        # the stateful arm's upfront conversation-create upload is included, so
        # the flat line starts honestly above zero
        offset = s["setup_req_bytes"]
        ax1.plot(turns, [offset + v for v in s["cum_req_bytes"]],
                 marker="o", label=label, color=color)
        ax2.plot(turns, s["cum_input_tokens"], marker="o", label=label, color=color)

    ax1.set_title("Cumulative upload bytes")
    ax1.set_xlabel("turn")
    ax1.set_ylabel("bytes sent by client")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Cumulative input tokens (billed)")
    ax2.set_xlabel("turn")
    ax2.set_ylabel("input tokens")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{cfg['model']} · {cfg['fixture']} · {cfg['turns']} turns — "
                 f"server-side state saves bytes, not tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    args = ap.parse_args()

    exp = json.loads((RESULTS_DIR / f"{args.name}.json").read_text())
    summary = metrics.summarize(exp)

    (RESULTS_DIR / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2))
    csv_path = write_csv(summary, RESULTS_DIR / f"{args.name}.csv")
    png_path = write_charts(summary, RESULTS_DIR / f"{args.name}.png")

    metrics.print_summary(summary)
    print(f"\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
