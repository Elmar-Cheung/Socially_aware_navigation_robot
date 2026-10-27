# compare_baselines.py
import numpy as np
import pandas as pd
from smooth_sac import SmoothSAC
from social_nav_env_no_obst import SocialNavEnv
from eval_utils import evaluate_to_csv, summarize
from scipy.stats import wilcoxon

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

RESULTS = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base"
N = 50

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class ConstantPolicy:
    def __init__(self, sizes):
        self.a = np.asarray(sizes, dtype=np.float32).reshape(1, 4)
    def predict(self, obs, deterministic=True):
        return self.a.copy(), None


env = SocialNavEnv(width=5, height=5, num_humans=1)
methods = {
    "learned":       SmoothSAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/em_based_no_obstacle_s1", device="cpu"),
    "const_0.5":     ConstantPolicy([0.5] * 4),
    "const_1.0":     ConstantPolicy([1.0] * 4),
    "const_1.5":     ConstantPolicy([1.5] * 4),
    "const_2.0":     ConstantPolicy([2.0] * 4),
    "const_asym":    ConstantPolicy([1.5, 0.8, 0.8, 0.5]),   # front, left, right, back
}

summary = []
for name, policy in methods.items():
    df = evaluate_to_csv(policy, env, f"{RESULTS}/{name}.csv", name, "base", n_episodes=N)
    summary.append({"method": name, **summarize(df, "m")})

table = pd.DataFrame(summary).set_index("method")
table.columns = [c.split("/", 1)[1] for c in table.columns]
print(table.round(3).to_string())
table.to_csv(f"{RESULTS}/summary.csv")

df = pd.read_csv(f"{RESULTS}/learned.csv")
print(df["steps"].describe(), df["steps"].nunique())

for s in range(5):
    env.reset(seed=10_000 + s)
    h = env.humans[0]
    print(s, round(h.base_speed, 3), h.pos, h.goal, env.robot.pos, env.robot.goal)

all_df = pd.concat([pd.read_csv(f"{RESULTS}/{name}.csv").assign(method=name) for name in methods.keys()])
all_df.groupby("method")[["intrusion_steps", "reactions", "neg_arousal", "triggers", "disc_ret"]].mean()

a = pd.read_csv(f"{RESULTS}/learned.csv").sort_values("seed")
b = pd.read_csv(f"{RESULTS}/const_1.0.csv").sort_values("seed")
print(wilcoxon(a["intrusion_steps"].values, b["intrusion_steps"].values))

METRICS = ["steps", "intrusion_steps", "neg_arousal", "reactions", "triggers", "disc_ret"]

def paired_tests(results_dir, a="learned", baselines=("const_1.0", "const_asym", "const_0.5")):
    A = pd.read_csv(f"{results_dir}/{a}.csv").set_index("seed").sort_index()
    rows = []
    for b in baselines:
        B = pd.read_csv(f"{results_dir}/{b}.csv").set_index("seed").sort_index()
        for m in METRICS:
            d = (A[m] - B[m]).values
            nz = d[d != 0]
            p = wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan
            rows.append(dict(baseline=b, metric=m, n_nonzero=len(nz),
                             median_diff=np.median(d), a_better=(d < 0).sum(),
                             b_better=(d > 0).sum(), p=p))
    df = pd.DataFrame(rows)
    ok = df["p"].notna()
    df.loc[ok, "p_holm"] = multipletests(df.loc[ok, "p"], method="holm")[1]
    return df.round(4)

print(paired_tests(RESULTS).to_string())

import pandas as pd
R = RESULTS
asym = pd.read_csv(f"{R}/const_asym.csv")
c10  = pd.read_csv(f"{R}/const_1.0.csv")
oracle = pd.concat([asym[asym["kind"].isin(["crossing", "head_on"])],
                    c10[c10["kind"] == "overtake"]]).sort_values("seed")
oracle["method"] = "oracle"
oracle.to_csv(f"{R}/oracle.csv", index=False)

all_df = pd.concat([pd.read_csv(f"{R}/{m}.csv") for m in
                    ["learned", "oracle", "const_1.0", "const_asym", "const_0.5"]])
print(all_df.groupby(["kind", "method"])[["steps", "intrusion_steps", "neg_arousal",
                                          "reactions", "disc_ret"]].mean().round(2))
print(paired_tests(R, baselines=("oracle", "const_1.0", "const_asym")).to_string())
print(df.groupby("kind")[["ps_front", "ps_left", "ps_right", "ps_back"]].mean().round(2))