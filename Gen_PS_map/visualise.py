import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from social_nav_env import SocialNavEnv, draw
from stable_baselines3 import SAC
import glob


# Load environment
env = SocialNavEnv(width=10, height=10, num_humans=2)

model = SAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/no_human", env=env)

# Reset to fresh episode
obs, _ = env.reset()

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#1a1a2e')

def animate(frame):
    global obs
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminate, _, info = env.step(action)
    draw(env, ax)
    if terminate:
        obs, _ = env.reset()

animation = FuncAnimation(
    fig, animate,
    frames=1000,
    interval=100,
    repeat=False,
    blit=False
)
plt.tight_layout()
plt.show(block=True)