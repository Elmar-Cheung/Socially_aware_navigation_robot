import json, numpy as np
for line in open("trap_debug/events.jsonl"):
    ev = json.loads(line); hs = ev["history"]
    d = [s["humans"][0]["dist_to_robot"] for s in hs]
    safe = np.mean([s["safety_triggered"] for s in hs])
    crosses = np.sum(np.diff(np.array(d) > 2.0) != 0)
    print(f"ep {ev['episode']}: dist {min(d):.2f}-{max(d):.2f} m, "
          f"safety {safe:.0%}, crosses 2 m {crosses}x, modes {set(s['planner_mode'] for s in hs)}")