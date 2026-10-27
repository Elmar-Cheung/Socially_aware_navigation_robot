import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from stable_baselines3 import SAC

from render import draw
#from social_nav_env import SocialNavEnv
#from dist_based_env import SocialNavEnv
from social_nav_env_no_obst import SocialNavEnv
#from dist_nav_env_no_obst import SocialNavEnv

# MODEL_PATH = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emo_based_obstacle"
MODEL_PATH = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emo_based_no_obstacle"

# MODEL_PATH = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/dist_based_obstacle"
# MODEL_PATH = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/dist_based_no_obstacle"

CSV_PATH = "live_simulation_results_emo_no_obs.csv"
N_EPISODES = 100

COLUMNS = ["episode", "reached_goal", "dist_per_step", "closest_dist", "avg_valence", "avg_intensity",
           "steps", "human_interactions", "reactions", "safety_triggers", "mean_jerk", "p95_jerk"]


class EpisodeLogger:
    """Collects per-step data during an episode and turns it into one row of metrics."""

    def __init__(self, env):
        self.env = env
        self.rows = []
        self.start(env)

    def start(self, env):
        self.start_pos = env.robot.pos.copy()
        self.goal = env.robot.goal.copy()     # robot.step assigns the next goal on arrival
        self.vels = []
        self.valence = self.intensity = 0.0

    def record(self):
        env = self.env
        self.vels.append(env.robot.measured_vel(env.dt))   # actual motion, not the command
        # Ground-truth emotion of the human the robot observes this step
        seen = set(env._obs['agents'][:, 7].astype(int)) - {0}
        #seen = set(env._obs['agents'][:, 5].astype(int)) - {0}
        for h in env.humans:
            if h.human_id in seen:
                self.valence += h.emotion[0]
                self.intensity += h.emotion[1]

    def finish(self, reached_goal):
        env, s = self.env, self.env.stats
        n = max(env.current_step, 1)
        vels = np.array(self.vels)
        if len(vels) >= 3:
            jerk = np.linalg.norm(np.diff(vels, n=2, axis=0) / env.dt ** 2, axis=1)
            mean_jerk, p95_jerk = float(jerk.mean()), float(np.percentile(jerk, 95))
        else:
            mean_jerk = p95_jerk = 0.0
        row = dict(
            episode=len(self.rows) + 1,
            reached_goal=int(reached_goal),
            dist_per_step=float(np.linalg.norm(self.goal - self.start_pos)) / n,
            closest_dist=s['min_human_dist'] if np.isfinite(s['min_human_dist']) else float('nan'),
            avg_valence=self.valence / n,
            avg_intensity=self.intensity / n,
            steps=env.current_step,
            human_interactions=s['human_interaction'],
            reactions=s['reactions'],
            safety_triggers=s['safety_triggers'],
            mean_jerk=mean_jerk,
            p95_jerk=p95_jerk,
        )
        self.rows.append(row)
        return row


def append_csv(path, row):
    new_file = not os.path.exists(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def plot_final_metrics(rows):
    """One subplot per metric, against episode number."""
    if not rows:
        return
    panels = [("dist_per_step", "Distance versus Step per Episode", "m / step", 'blue', 'o'),
              ("closest_dist", "Closest Distance per Episode", "Distance (m)", 'orange', 's'),
              ("avg_valence", "Average Valence per Episode", "Valence", 'green', '^'),
              ("avg_intensity", "Average Intensity per Episode", "Intensity", 'red', 'd'),
              ("reactions", "Detected Human Reactions per Episode", "Count", 'purple', 'x'),
              ("steps", "Steps per Episode", "Steps", 'brown', 'o')]
    episodes = [r["episode"] for r in rows]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 17), sharex=True)
    for ax, (key, title, ylabel, color, marker) in zip(axes, panels):
        ax.plot(episodes, [r[key] for r in rows], marker=marker, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Episode")
    plt.tight_layout()
    plt.show()


def main():
    env = SocialNavEnv(width=5, height=5, num_humans=1)
    model = SAC.load(MODEL_PATH, env=env)
    obs, _ = env.reset()
    logger = EpisodeLogger(env)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#1a1a2e')

    def animate(_frame):
        nonlocal obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        logger.record()
        draw(env, ax)
        if not (terminated or truncated):
            return

        row = logger.finish(reached_goal=terminated)
        append_csv(CSV_PATH, row)
        print(f"Episode {row['episode']}: {'goal reached' if terminated else 'timed out'} in {row['steps']} steps | "
              f"dist/step {row['dist_per_step']:.3f} | closest {row['closest_dist']:.2f} m | "
              f"valence {row['avg_valence']:+.3f} | intensity {row['avg_intensity']:.3f} | "
              f"interactions {row['human_interactions']} | reactions {row['reactions']}")

        obs, _ = env.reset()
        logger.start(env)
        if row['episode'] >= N_EPISODES:
            print(f"Target episode count reached ({N_EPISODES}). Stopping animation.")
            animation.event_source.stop()
            plt.close(fig)

    animation = FuncAnimation(fig, animate, frames=10_000_000, interval=1, repeat=False, blit=False)
    plt.tight_layout()
    plt.show(block=True)        # blocks until the animation window is closed
    plot_final_metrics(logger.rows)


if __name__ == "__main__":
    main()