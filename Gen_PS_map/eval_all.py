# eval_all.py
import pandas as pd
from smooth_sac import SmoothSAC
from social_nav_env_no_obst import SocialNavEnv
from eval_utils import evaluate_to_csv
from compare_baselines import ConstantPolicy   # or copy the class here
import numpy as np
BASE = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map"
RESULTS = f"{BASE}/results/base_100"
N = 100

env_abs = SocialNavEnv(width=5, height=5, num_humans=1, residual=False)
env_res = SocialNavEnv(width=5, height=5, num_humans=1, residual=True)
"""
methods = {f"learned_s{k}": SmoothSAC.load(f"{BASE}/em_based_no_obstacle_s{k}", device="cpu")
           for k in range(1, 6)}
methods.update({f"residual_s{k}": SmoothSAC.load(f"{BASE}/em_based_no_obstacle_r{k}", device="cpu")
           for k in range(1, 6)})"""
methods = {f"new_buffer_learned_s{k}": SmoothSAC.load(f"{BASE}/em_based_no_obstacle_new_s{k}", device="cpu")
           for k in range(1, 6)}
methods.update({
    "const_0.5":  ConstantPolicy([0.5] * 4),
    "const_1.0":  ConstantPolicy([1.0] * 4),
    "const_asym": ConstantPolicy([1.5, 0.8, 0.8, 0.5]),
    "const_asym_back1": ConstantPolicy([1.5, 0.8, 0.8, 1.0]),
})
for s in (1.0, 1.1, 1.2, 1.3, 1.5):
    methods[f"prior_x{s}"] = ConstantPolicy(list(s * np.array([1.5, 0.8, 0.8, 1.0])))
    
for name, policy in methods.items():
    env = env_res if name.startswith("residual") else env_abs
    evaluate_to_csv(policy, env, f"{RESULTS}/{name}.csv", name, "base", n_episodes=N)
    print("done", name)

cols = ["steps", "success", "intrusion_steps", "neg_arousal", "reactions", "triggers", "disc_ret"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

#learned  = pd.concat([pd.read_csv(f"{RESULTS}/learned_s{k}.csv") for k in range(1, 6)])
#residual = pd.concat([pd.read_csv(f"{RESULTS}/residual_s{k}.csv") for k in range(1, 6)])
learned  = pd.concat([pd.read_csv(f"{RESULTS}/new_buffer_learned_s{k}.csv") for k in range(1, 6)])
prior    = pd.read_csv(f"{RESULTS}/const_asym_back1.csv")

per_seed = learned.groupby("method")[cols].mean()
print("From scratch, per seed:");        print(per_seed.round(2))
print("From scratch, mean ± std:");      print(per_seed.agg(["mean", "std"]).round(2))

#both = pd.concat([residual, prior])
both = pd.concat([learned, prior])
print("\nNew buffer vs its prior:");       print(both.groupby("method")[cols].mean().round(2))
print("\nPer type:");                    print(both.groupby(["kind", "method"])[cols].mean().round(2))

