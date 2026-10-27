from smooth_sac import SmoothSAC
from social_nav_env_no_obst import SocialNavEnv
from trace_episode import trace_episode, plot_trace, plot_trajectory
import pandas as pd

BASE = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map"
RESULTS = f"{BASE}/results/base_100"
REP = "learned_s2"                                    # median seed by intrusion_steps

env = SocialNavEnv(width=5, height=5, num_humans=1)
model = SmoothSAC.load(f"{BASE}/emotion_based_no_obstacle_s2", device="cpu")
df_rep = pd.read_csv(f"{RESULTS}/{REP}.csv")

for kind, g in df_rep.groupby("kind"):
    ep_seed = int(g.iloc[(g["steps"] - g["steps"].median()).abs().argsort().iloc[0]]["seed"])
    trace, k = trace_episode(model, env, seed=ep_seed)            # no [0]
    plot_title = f"{kind} Seed {ep_seed}"
    plot_trace(trace, k, title=plot_title, path=f"{RESULTS}/trace_{REP}_{k}_{ep_seed}.png")
    plot_trajectory(trace, k, every=0.5, title=plot_title, path=f"{RESULTS}/traj_{REP}_{k}_{ep_seed}.png")
    print("saved", k, ep_seed)