import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from social_nav_env import SocialNavEnv
#from dist_navigation_rl import SocialNavEnv, draw
#from dist_nav_env_no_obst import SocialNavEnv, draw
from social_nav_env_no_obst import SocialNavEnv, draw
#from social_nav_env_2H import SocialNavEnv, draw
from stable_baselines3 import SAC
from render import draw
import csv
import numpy as np


# Load environment
env = SocialNavEnv(width=7, height=4, num_humans=1)
model = SAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emo_based_obstacle", env=env)
#model = SAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emo_based_no_obstacle", env=env)
# Reset to fresh episode
obs, _ = env.reset()

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#1a1a2e')

global reset_cnt
reset_cnt = 0

dist_vs_step_history = []
closest_dist_history = []
avg_valence_history = []
avg_intensity_history = []
step_per_per_episode = []
human_interactions_history = []
episode_vel_history = []
mean_jerk_history = []
p95_jerk_history = []

with open("live_simulation_results_emo.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "current_episode_num", "Distance versus Step per Episode", "Closest Distance per Episode", 
                "Average Valence per Episode", "Average Intensity per Episode", "Steps per Episode",
                "mean_jerk", "p95_jerk"
            ])

def animate(frame):
    global reset_cnt
    global obs
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminate, _, info = env.step(action)
    draw(env, ax)
    episode_vel_history.append(env.robot.vel.copy())

    if terminate:
        num_step = env.current_step
        
        # Calculate current episode metrics
        d_vs_s = env.robot.goal_dist / num_step
        c_dist = env.min_human_dist
        a_val = env.cascade_val / num_step
        a_int = env.cascade_int / num_step
        #a_val = env.valence_history / num_step
        #a_int = env.intensity_history / num_step
        h_int = env.human_interaction
        s_per_ep = num_step
        
        # Append metrics to the global tracking lists
        vels = np.array(episode_vel_history)          # shape (T, 2)
        if len(vels) >= 3:
            accel = np.diff(vels, axis=0) / env.dt
            jerk = np.diff(accel, axis=0) / env.dt
            jerk_mag = np.linalg.norm(jerk, axis=1)
            mean_jerk = float(jerk_mag.mean())
            p95_jerk = float(np.percentile(jerk_mag, 95))
        else:
            mean_jerk = 0.0
            p95_jerk = 0.0

        mean_jerk_history.append(mean_jerk)
        p95_jerk_history.append(p95_jerk)

        dist_vs_step_history.append(d_vs_s)
        closest_dist_history.append(c_dist)
        avg_valence_history.append(a_val)
        avg_intensity_history.append(a_int)
        human_interactions_history.append(h_int)
        step_per_per_episode.append(s_per_ep)

        current_episode_num = len(dist_vs_step_history)
        with open("live_simulation_results_emo.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_episode_num, d_vs_s, c_dist, a_val, a_int, s_per_ep, mean_jerk, p95_jerk
            ])

        print("Dist versus step: ", d_vs_s)
        print("Closest dist: ", c_dist)
        print("Average valence: ", a_val)
        print("Average intensity: ", a_int)
        print("Number of human interactions: ", h_int)

        episode_vel_history.clear()
        obs, _ = env.reset()
        reset_cnt += 1
        print(f"Episode reset. Total resets: {reset_cnt}")

        if reset_cnt >= 100:
            print(f"Target reset count reached ({reset_cnt}). Stopping animation.")
            animation.event_source.stop()
            plt.close(fig) # Closes the animation window automatically to let the final plot take over

        
def plot_final_metrics():
    """Generates 5 individual plots arranged in a grid at the end of the simulation."""
    fig_final, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)
    episodes = range(1, len(dist_vs_step_history) + 1)
    
    # Plot 1: Distance vs Step
    axes[0].plot(episodes, dist_vs_step_history, marker='o', color='blue')
    axes[0].set_title("Distance versus Step per Episode")
    axes[0].set_ylabel("Ratio")
    
    # Plot 2: Closest Distance
    axes[1].plot(episodes, closest_dist_history, marker='s', color='orange')
    axes[1].set_title("Closest Distance per Episode")
    axes[1].set_ylabel("Distance")
    
    # Plot 3: Average Valence
    axes[2].plot(episodes, avg_valence_history, marker='^', color='green')
    axes[2].set_title("Average Valence per Episode")
    axes[2].set_ylabel("Valence")
    
    # Plot 4: Average Intensity
    axes[3].plot(episodes, avg_intensity_history, marker='d', color='red')
    axes[3].set_title("Average Intensity per Episode")
    axes[3].set_ylabel("Intensity")
    """
    # Plot 5: Human Interactions
    axes[4].plot(episodes, human_interactions_history, marker='x', color='purple')
    axes[4].set_title("Number of Human Interactions per Episode")
    axes[4].set_ylabel("Count")
    axes[4].set_xlabel("Episode")
    """
    # Plot 6: Steps per Episode
    axes[4].plot(episodes, step_per_per_episode, marker='o', color='brown')
    axes[4].set_title("Steps per Episode")
    axes[4].set_ylabel("Steps")
    axes[4].set_xlabel("Episode")

    plt.tight_layout()
    plt.show() # <--- Essential step to render and block execution for the final graph

animation = FuncAnimation(
    fig, animate,
    frames=10000000,
    interval=1,
    repeat=False,
    blit=False
)

plt.tight_layout()
plt.show(block=True) # Script execution blocks here until the animation window is stopped/closed

# Once animation finishes, script proceeds to render the metrics
plot_final_metrics()
