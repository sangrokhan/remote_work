import json
import statistics

d = json.load(open("/tmp/idle_reset_exp/results_realistic.json"))

for phase in ("web_client_enabled", "web_client_disabled"):
    print(f"\n=== {phase} ===")
    by_turn = {}
    for res in d[phase]:
        for t in res["run"]["turns"]:
            by_turn.setdefault(t["turn"], []).append(t["response_raw"].get("recv_ms"))
    for turn, vals in sorted(by_turn.items()):
        wire = d[phase][0]["run"]["turns"][turn]["wire_sent"]
        print(f"  turn {turn} (wire_sent={wire:6d}B): {[round(v,3) for v in vals]}  mean={statistics.mean(vals):.2f}ms")

print("\n=== turns 1-9 pooled (post-idle only, excl. turn 0) ===")
enabled_all = []
disabled_all = []
for res in d["web_client_enabled"]:
    enabled_all.extend([t["response_raw"].get("recv_ms") for t in res["run"]["turns"][1:]])
for res in d["web_client_disabled"]:
    disabled_all.extend([t["response_raw"].get("recv_ms") for t in res["run"]["turns"][1:]])

print(f"enabled:  n={len(enabled_all)} mean={statistics.mean(enabled_all):.2f}ms median={statistics.median(enabled_all):.2f}ms min={min(enabled_all):.2f} max={max(enabled_all):.2f}")
print(f"disabled: n={len(disabled_all)} mean={statistics.mean(disabled_all):.2f}ms median={statistics.median(disabled_all):.2f}ms min={min(disabled_all):.2f} max={max(disabled_all):.2f}")
ratio = statistics.mean(enabled_all) / statistics.mean(disabled_all)
print(f"ratio (enabled/disabled): {ratio:.1f}x")

print("\n=== turn 0 (baseline, both conditions right after connect) ===")
for phase in ("web_client_enabled", "web_client_disabled"):
    vals = [res["run"]["turns"][0]["response_raw"].get("recv_ms") for res in d[phase]]
    print(f"  {phase}: {vals} mean={statistics.mean(vals):.2f}ms")
