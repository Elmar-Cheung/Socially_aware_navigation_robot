import numpy as np
from social_nav_env_no_obst import SocialNavEnv
env = SocialNavEnv(width=5, height=5, num_humans=1)
d = []
for s in range(10_000, 10_100):
    env.reset(seed=s)
    if env.encounter == "overtake":
        d.append(np.linalg.norm(env.humans[0].goal - env.robot.goal))
print(len(d), np.round(np.percentile(d, [0, 50, 100]), 2))