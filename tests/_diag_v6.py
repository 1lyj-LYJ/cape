import json

print("== per-round uniqueness/votes ==")
for line in open(r'runs/gate2_v6_s0/log.jsonl', encoding='utf-8'):
    d = json.loads(line)
    print(f"round {d['round']}: unique={d['n_unique_candidates']} best_votes={d['best_votes']:.0f}")

print("\n== cloud report ==")
print(open(r'runs/gate2_v6_s0/uniform_sample_report.json', encoding='utf-8').read())
