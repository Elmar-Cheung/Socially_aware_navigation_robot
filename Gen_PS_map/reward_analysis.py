import numpy as np
from social_nav_env_no_obst import SocialNavEnv as Emo
from dist_nav_env_no_obst import SocialNavEnv as Dist
from collections import Counter
emo_events, react, react_gt_walk = Counter(), Counter(), Counter()
event_dist = {}
prev = {}
name = "dist"
Env = Dist
env = Env(width=5, height=5, num_humans=1)
stats = dict(inside_steps=0, depth=0.0, front_inside=0, robot_closing=0)
for size in (0.25, 0.5, 1.0, 1.5, 2.0):
    totals, steps = {}, 0
    obs, _ = env.reset(seed=0)
    for _ in range(100):                          # episodes
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(np.full(env.action_space.shape, size, np.float32))
            for hid, e in info["reactions"].items():
                react[e] += 1
                if info["gt_responses"][hid][0] == "walk":
                    react_gt_walk[e] += 1                     # detected while the human was really just walking
            for h in env.humans:
                label = h.behaviour_type
                if label != "regular" and label != prev.get(h.human_id):
                    emo_events[label] += 1
                    event_dist.setdefault(label, []).append(float(np.linalg.norm(h.pos - env.robot.pos)))
                prev[h.human_id] = label
            for k in ("move_reward", "tot_soc_pen", "collision_reward", "base_shaping_reward", "human_reaction_reward"):
                totals[k] = totals.get(k, 0.0) + info[k]
            steps += 1; done = term or trunc
        obs, _ = env.reset()
    print(name, size, {k: round(v / 100, 2) for k, v in totals.items()}, "steps/ep", steps // 100)
    precision = {e: 1 - react_gt_walk[e] / react[e] for e in react}
    print(name, size, "precision", precision)