import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/.env"))
os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")

import wandb
import matplotlib
from social_nav_env import SocialNavEnv, draw, HumanOnlyReplayBuffer
#from dist_navigation_rl import SocialNavEnv, draw, HumanOnlyReplayBuffer
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import matplotlib.pyplot as plt
matplotlib.use('TkAgg', force=True)
from matplotlib.animation import FuncAnimation
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")
import stable_baselines3
import affective_buffer as ab
import numpy as np


# Now login and init
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

wandb.init(
    project="social_nav",
    settings=wandb.Settings(
        api_key=os.environ["WANDB_API_KEY"],
        mode="online"
    ),
    config={
        "algorithm": "SAC",
        "total_timesteps": 50000,
        "num_humans": 2,
        "width": 6,
        "height": 6,
    }
)

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Removed wandb.init from here to avoid multiprocess memory leaks on initialization

    def _on_step(self):
        wandb.log({
            'reward':           self.locals['rewards'][0],
            'dist_to_goal':     self.locals['infos'][0].get('dist_to_goal', 0),
            'time_used':        self.locals['infos'][0].get('time_used', 0),
            'tot_soc_pen':      self.locals['infos'][0].get('tot_soc_pen', 0),
            'move_reward':      self.locals['infos'][0].get('move_reward', 0),
            'social_reward':    self.locals['infos'][0].get('social_reward', 0),
            'orientation':      self.locals['infos'][0].get('orientation', 0),
            'orientation_unwrapped': self.locals['infos'][0].get('orientation_unwrapped', 0),
            'speed':            self.locals['infos'][0].get('speed', 0),
            'arrive':           self.locals['infos'][0].get('human_arrived', 0),
            'episode_step':     self.locals['infos'][0].get('episode_step', 0),
            'avg_pen':     self.locals['infos'][0].get('avg_pen', 0),
            'avg_trigger':     self.locals['infos'][0].get('avg_trigger', 0),
        })
        return True
    
    def _on_rollout_end(self):
        if self.logger.name_to_value:
            wandb.log({
                "train/actor_loss":   self.logger.name_to_value.get('train/actor_loss', 0),
                "train/critic_loss":  self.logger.name_to_value.get('train/critic_loss', 0),
                "train/ent_coef":     self.logger.name_to_value.get('train/ent_coef', 0),
                "train/ent_coef_loss":self.logger.name_to_value.get('train/ent_coef_loss', 0),
                "time/total_timesteps": self.num_timesteps,
            })
    
    def _on_training_end(self):
        pass # Handle closing cleanly at the end of the global script instead

BASE_DIR = os.path.join("C:\\", "Users", "Elmar", "Desktop", "MTP", "All_models", "Gen_PS_map")
os.makedirs(os.path.join(BASE_DIR, "tensorboard_logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)

# --- Initialization ---
# Initialize wandb directly in the main thread execution context

env = SocialNavEnv(width=6, height=6, num_humans=2)

model = SAC(
    "MultiInputPolicy", env,
    replay_buffer_class=HumanOnlyReplayBuffer,
    verbose=1,
    device="cuda",
    tensorboard_log=os.path.join(BASE_DIR, "tensorboard_logs"),
    learning_rate=3e-4,
    buffer_size=50000,
    learning_starts=5000,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    target_entropy='auto'
)

checkpoint = CheckpointCallback(
    save_freq=2500,
    save_path=os.path.join(BASE_DIR, "checkpoints"),
    name_prefix='social_nav_sac'
)

# --- Training Loop ---
model.learn(total_timesteps=50000, callback=[checkpoint, WandbCallback()])

# --- Secure Shutdown and Save ---
print("Training finished. Safely synchronizing and closing W&B threads...")
wandb.finish() # Crucial: This unlocks the training process stream

# Save to absolute path to guarantee visualize.py accesses the same location
final_save_path = os.path.join(BASE_DIR, "no_human.zip")
if os.path.exists(final_save_path):
    os.remove(final_save_path)

model.save(final_save_path)
print
print(f"SUCCESS: Model saved securely to: {final_save_path}")

#----- Visualisation -----
"""
model = SAC.load("no_human.zip", env=env)

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
"""