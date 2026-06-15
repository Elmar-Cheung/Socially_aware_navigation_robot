import os
import matplotlib
from social_nav_env import SocialNavEnv, draw
from stable_baselines3 import SAC
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
                "algorithm": "SAC",
                "total_timesteps": 10000,
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
        if self.logger.name_to_value:
            wandb.log({
                "train/actor_loss":   self.logger.name_to_value.get('train/actor_loss', 0),
                "train/critic_loss":  self.logger.name_to_value.get('train/critic_loss', 0),
                "train/ent_coef":     self.logger.name_to_value.get('train/ent_coef', 0),
                "train/ent_coef_loss":self.logger.name_to_value.get('train/ent_coef_loss', 0),
                "time/total_timesteps": self.num_timesteps,
            })
    
    def _on_training_end(self):
        wandb.finish()


# --- Training ---
env = SocialNavEnv(width=25, height=25, num_humans=30)

# Add this before model = PPO(...)
os.makedirs("tensorboard_logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

model = SAC(
    "MultiInputPolicy", env,
    verbose=1,
    device="cuda",
    tensorboard_log="./tensorboard_logs/",
    learning_rate=3e-4,
    buffer_size=1000,
    learning_starts=2500,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    target_entropy='auto'
)

checkpoint = CheckpointCallback(
    save_freq=2500,
    save_path='./checkpoints/',
    name_prefix='social_nav_sac'
)

model.learn(total_timesteps=10000, callback=[checkpoint, WandbCallback()])

model.save("1.zip")

buffer = ab.get_buffer()
analyser = affectivityML()
analyser.analyse(buffer)
#----- Visualisation -----

model = SAC.load("fix_approach_goal.zip", env=env)

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