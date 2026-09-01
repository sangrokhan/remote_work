import json
import statistics

d = json.load(open("/tmp/idle_reset_exp/results_upload_webclient.json"))

for phase in ("web_client_enabled", "web_client_disabled"):
    print(f"\n=== {phase} ===")
    by_turn = {}
    for res in d[phase]:
        for t in res["run"]["turns"]:
            by_turn.setdefault(t["turn"], []).append(t["response_raw"].get("recv_ms"))
    for turn, vals in sorted(by_turn.items()):
        print(f"  turn {turn}: {[round(v,1) for v in vals]}  mean={statistics.mean(vals):.1f}ms")

print("\n=== turns 1-5 pooled (post-idle uploads only, excl. turn 0 which follows connect not idle) ===")
enabled_all = []
disabled_all = []
for res in d["web_client_enabled"]:
    enabled_all.extend([t["response_raw"].get("recv_ms") for t in res["run"]["turns"][1:]])
for res in d["web_client_disabled"]:
    disabled_all.extend([t["response_raw"].get("recv_ms") for t in res["run"]["turns"][1:]])

print(f"enabled:  n={len(enabled_all)} mean={statistics.mean(enabled_all):.1f}ms median={statistics.median(enabled_all):.1f}ms stdev={statistics.stdev(enabled_all):.1f}")
print(f"disabled: n={len(disabled_all)} mean={statistics.mean(disabled_all):.1f}ms median={statistics.median(disabled_all):.1f}ms stdev={statistics.stdev(disabled_all):.1f}")
diff = statistics.mean(enabled_all) - statistics.mean(disabled_all)
print(f"diff (enabled - disabled): {diff:+.1f}ms ({diff/statistics.mean(disabled_all)*100:+.1f}%)")
