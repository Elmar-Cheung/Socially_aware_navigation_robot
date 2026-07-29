import os
import matplotlib
from social_nav_env import SocialNavEnv, draw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wandb
import torch
import stable_baselines3
import affective_buffer as ab
from affectivityML import affectivityML

matplotlib.use('TkAgg')

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        wandb.init(
            project="social_nav",
            config={
                "algorithm": "PPO",
                "total_timesteps": 250000,
                "num_humans": 25,
                "width": 25,
                "height": 30,
            }
        )

    def _on_step(self):
        wandb.log({
            'reward':          self.locals['rewards'][0],
            'dist_to_goal':    self.locals['infos'][0].get('dist_to_goal', 0),
            'time_used':       self.locals['infos'][0].get('time_used', 0),
            'tot_soc_pen':     self.locals['infos'][0].get('tot_soc_pen', 0),
            'move_reward':     self.locals['infos'][0].get('move_reward', 0),
            'social_reward':   self.locals['infos'][0].get('social_reward', 0)
        })
        return True
    
    def _on_rollout_end(self):
        # PPO metrics are only available after rollout ends, not every step
        if self.logger.name_to_value:
            wandb.log({
                "train/entropy_loss":       self.logger.name_to_value.get('train/entropy_loss', 0),
                "train/explained_variance": self.logger.name_to_value.get('train/explained_variance', 0),
                "train/loss":               self.logger.name_to_value.get('train/loss', 0),
                "train/policy_gradient_loss": self.logger.name_to_value.get('train/policy_gradient_loss', 0),
                "train/value_loss":         self.logger.name_to_value.get('train/value_loss', 0),
                "train/approx_kl":          self.logger.name_to_value.get('train/approx_kl', 0),
                "train/clip_fraction":      self.logger.name_to_value.get('train/clip_fraction', 0),
                "time/total_timesteps":     self.num_timesteps,
            })
    
    def _on_training_end(self):
        wandb.finish()


# --- Training ---
env = SocialNavEnv(width=25, height=25, num_humans=30)

# Add this before model = PPO(...)
os.makedirs("tensorboard_logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

model = PPO(
    "MultiInputPolicy", env,
    verbose=1,
    device="cuda",
    tensorboard_log="./tensorboard_logs/",
    n_steps=2048,
    batch_size=128,
    n_epochs=5,
    learning_rate=0.0001,
    target_kl=0.02
)

checkpoint = CheckpointCallback(
    save_freq=5000,
    save_path='./checkpoints/',
    name_prefix='social_nav'
)

model.learn(total_timesteps=250000, callback=[checkpoint, WandbCallback()])

model.save("fix_approach_goal.zip")

buffer = ab.get_buffer()
analyser = affectivityML()
analyser.analyse(buffer)
#----- Visualisation -----

model = PPO.load("fix_approach_goal.zip", env=env)

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