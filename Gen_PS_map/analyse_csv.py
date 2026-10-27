import numpy as np
from collections import Counter
from social_nav_env_no_obst import SocialNavEnv

env = SocialNavEnv(width=5, height=5, num_humans=1)
obs, _ = env.reset(seed=0)
lengths, kinds, seen, ep_len = [], Counter(), 0, 0
for _ in range(5000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    ep_len += 1
    seen += int(info["human_nearby"])
    if term or trunc:
        lengths.append(ep_len); kinds[env.encounter] += 1; ep_len = 0
        obs, _ = env.reset()
print("episodes:", len(lengths), "mean length:", np.mean(lengths), "min:", min(lengths))
print("kinds:", kinds, "human in view:", seen / 5000)