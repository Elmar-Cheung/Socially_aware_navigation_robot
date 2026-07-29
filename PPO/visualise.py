import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from social_nav_env import SocialNavEnv, draw
from stable_baselines3 import PPO
import glob
import os

# Load environment
env = SocialNavEnv(width=15, height=15, num_humans=10)
  
model = PPO.load("90-fast-mountain.zip", env=env)

# Reset to fresh episode
obs, _ = env.reset()

fig, ax = plt.subplots(figsize=(8, 8))
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
    frames=2000,
    interval=100,
    repeat=False,
    blit=False
)
plt.tight_layout()
plt.show(block=True)