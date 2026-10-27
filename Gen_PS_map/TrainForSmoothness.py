import os
import gc
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/.env"))
os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")

import wandb
import matplotlib
matplotlib.use('TkAgg', force=True)

from social_nav_env_no_obst import SocialNavEnv, HumanOnlyReplayBuffer
from buffer import EventWindowReplayBuffer
from eval_utils import DeterministicEvalCallback

from smooth_sac import SmoothSAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import obs_as_tensor, set_random_seed
import torch
import numpy as np

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

# ------------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------------
RUN_PREFIX      = "em_based_no_obstacle_new_buffer"
SEEDS           = [1, 2, 3, 4, 5]
TOTAL_TIMESTEPS = 50000

BASE_DIR = os.path.join("C:\\", "Users", "Elmar", "Desktop", "MTP", "All_models", "Gen_PS_map")
TB_DIR   = os.path.join(BASE_DIR, "tensorboard_logs")
os.makedirs(TB_DIR, exist_ok=True)


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        last_obs = self.model._last_obs
        obs_tensor = obs_as_tensor(last_obs, self.model.device)

        with torch.no_grad():
            actions, _ = self.model.policy.actor.action_log_prob(obs_tensor)
            q_values_tuple = self.model.critic(obs_tensor, actions)
            q_values = torch.min(q_values_tuple[0], q_values_tuple[1])
            mean_q = q_values.mean().item()
            max_q = q_values.max().item()

        grad_norms = [p.grad.detach().norm(2).item()
                      for p in self.model.policy.critic.parameters() if p.grad is not None]
        critic_grad_norm = float(np.linalg.norm(grad_norms)) if grad_norms else 0.0

        raw_rewards = self.locals["rewards"]
        info = self.locals['infos'][0]

        wandb.log({
            'mean_q': mean_q,
            'max_q': max_q,
            'critic_grad_norm': critic_grad_norm,
            'raw_reward_distribution': wandb.Histogram(raw_rewards),
            '0reward': raw_rewards,
            'dist_to_goal':          info.get('dist_to_goal', 0),
            'time_used':             info.get('time_used', 0),
            'tot_soc_pen':           info.get('tot_soc_pen', 0),
            'move_reward':           info.get('move_reward', 0),
            'collision_reward':      info.get('collision_reward', 0),
            'base_shaping_reward':   info.get('base_shaping_reward', 0),
            'orientation':           info.get('orientation', 0),
            'orientation_unwrapped': info.get('orientation_unwrapped', 0),
            'speed':                 info.get('speed', 0),
            'arrive':                info.get('human_arrived', 0),
            'episode_step':          info.get('episode_step', 0),
            'avg_pen':               info.get('avg_pen', 0),
            'avg_trigger':           info.get('avg_trigger', 0),
            'cascade_cp':            info.get('cascade_cp', 0),
            'cascade_mp':            info.get('cascade_mp', 0),
            'cascade_sp':            info.get('cascade_sp', 0),
            'last_unsafe':           info.get('last_unsafe', 0),
            'action_front':          info.get('action_front', 0),
            'action_left':           info.get('action_left', 0),
            'action_right':          info.get('action_right', 0),
            'action_back':           info.get('action_back', 0),
            'action_delta_front':    info.get('action_delta_front', 0),
            'action_delta_left':     info.get('action_delta_left', 0),
            'action_delta_right':    info.get('action_delta_right', 0),
            'action_delta_back':     info.get('action_delta_back', 0),
        })
        return True

    def _on_rollout_end(self):
        v = self.logger.name_to_value
        if v:
            wandb.log({
                "train/actor_loss":      v.get('train/actor_loss', 0),
                "train/critic_loss":     v.get('train/critic_loss', 0),
                "train/ent_coef":        v.get('train/ent_coef', 0),
                "train/ent_coef_loss":   v.get('train/ent_coef_loss', 0),
                "train/smooth_penalty":  v.get('train/smooth_penalty', 0),
                "train/presence_mean":   v.get('train/presence_mean', 0),
                "train/gap_mean":        v.get('train/gap_mean', 0),
                "train/sat_frac":        v.get('train/sat_frac', 0),
                "train/active_frac":     v.get('train/active_frac', 0),
                "train/actor_grad_norm": v.get('train/actor_grad_norm', 0),
                "time/total_timesteps":  self.num_timesteps,
                "event_buffer_size": self.model.replay_buffer.event.size(),
                "n_events": self.model.replay_buffer.n_events,
            })


def train_one(seed: int):
    run_name = f"{RUN_PREFIX}_s{seed}"
    print(f"\n========== Starting {run_name} ==========")

    # Seeds python `random`, numpy and torch (CPU + CUDA)
    set_random_seed(seed)

    run = wandb.init(
        project="social_nav",
        name=run_name,
        group=RUN_PREFIX,          # groups the 5 runs together in the W&B UI
        settings=wandb.Settings(mode="online"),
        config={
            "algorithm": "SAC+L2C2",
            "seed": seed,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_humans": 1,
            "width": 5,
            "height": 5,
            "smooth_coef": 0.01,
            "smooth_lipschitz": 5.0,
            "smooth_noise_std": 0.02,
            "temporal_coef": 0.1,
        },
    )
    cfg = run.config

    # Fresh environments every run
    env      = SocialNavEnv(width=5, height=5, num_humans=1, residual=False)
    eval_env = SocialNavEnv(width=5, height=5, num_humans=1, residual=False)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)   # different stream from the training env

    model = SmoothSAC(
        "MultiInputPolicy", env,
        replay_buffer_class=EventWindowReplayBuffer,
        verbose=1,
        device="cuda",
        seed=seed,
        tensorboard_log=TB_DIR,
        learning_rate=3e-4,
        buffer_size=20000,
        learning_starts=5000,
        train_freq=1,
        gradient_steps=1,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        target_entropy='auto',
        smooth_coef=cfg["smooth_coef"],
        smooth_lipschitz=cfg["smooth_lipschitz"],
        smooth_noise_std=cfg["smooth_noise_std"],
        temporal_coef=cfg["temporal_coef"],
    )

    # Separate checkpoint folder per seed so runs don't overwrite each other
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = CheckpointCallback(save_freq=2500, save_path=ckpt_dir, name_prefix=run_name)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=run_name,
        callback=[checkpoint, WandbCallback(), DeterministicEvalCallback(eval_env, eval_freq=5000)],
    )

    final_save_path = os.path.join(BASE_DIR, f"{run_name}.zip")
    model.save(final_save_path)   # overwrites if it exists
    print(f"SUCCESS: {run_name} saved to {final_save_path}")

    wandb.finish()

    # Clean up before the next seed
    env.close()
    eval_env.close()
    del model, env, eval_env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for s in SEEDS:
        train_one(s)
    print("\nAll seeds finished.")