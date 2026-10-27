import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

RESULTS = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base_100"
METRICS = ["steps", "intrusion_steps", "neg_arousal", "reactions", "triggers", "disc_ret"]
COLS = ["success", "steps", "intrusion_steps", "neg_arousal", "reactions", "triggers", "disc_ret",
        "startles", "jolts", "freezes", "gt_stops", "gt_sidesteps"]    
LEARNED = [f"learned_s{k}" for k in range(1, 6)]
CONSTS = ["const_0.5", "const_1.0", "const_asym", "const_asym_back1"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

seeds = pd.concat([pd.read_csv(f"{RESULTS}/{m}.csv") for m in LEARNED])
consts = pd.concat([pd.read_csv(f"{RESULTS}/{m}.csv") for m in CONSTS])

# --- 1. Oracle: per type, the constant with the fewest intrusions among those with >= 95% success
by_type = consts.groupby(["kind", "method"])[["success", "intrusion_steps"]].mean()
choice = {}
for kind, g in by_type.groupby(level="kind"):
    ok = g[g["success"] >= 0.95]
    choice[kind] = ok["intrusion_steps"].idxmin()[1]
print("oracle picks:", choice)
oracle = pd.concat([consts[(consts.method == m) & (consts.kind == k)] for k, m in choice.items()])
oracle = oracle.assign(method="oracle").sort_values("seed")
oracle.to_csv(f"{RESULTS}/oracle.csv", index=False)

# --- 2. Overall tables
print("\nConstants + oracle:")
print(pd.concat([consts, oracle]).groupby("method")[COLS].mean().round(2))
per_seed = seeds.groupby("method")[COLS].mean()
print("\nLearned, per seed:"); print(per_seed.round(2))
print("\nLearned, mean ± std across seeds:"); print(per_seed.agg(["mean", "std"]).round(2))

# --- 3. Per encounter type (learned averaged over seeds, plus every baseline)
learned_avg = seeds.groupby(["kind", "seed"])[COLS].mean().reset_index().assign(method="learned")
everything = pd.concat([learned_avg, consts, oracle])
print("\nPer encounter type:")
print(everything.groupby(["kind", "method"])[COLS].mean().round(2))

# --- 4. Paired tests, learned averaged over the 5 seeds per eval episode
def seed_avg_tests(R, baselines):
    L = seeds.groupby("seed")[METRICS].mean()
    rows = []
    for b in baselines:
        B = pd.read_csv(f"{R}/{b}.csv").set_index("seed")[METRICS].loc[L.index]
        for m in METRICS:
            d = (L[m] - B[m]).values
            nz = d[d != 0]
            rows.append(dict(baseline=b, metric=m, n_nonzero=len(nz), median_diff=np.median(d),
                             learned_better=(d < 0).sum(), base_better=(d > 0).sum(),
                             p=wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan))
    df = pd.DataFrame(rows); ok = df["p"].notna()
    df.loc[ok, "p_holm"] = multipletests(df.loc[ok, "p"], method="holm")[1]
    return df.round(4)

print("\nPaired tests:")
print(seed_avg_tests(RESULTS, ["oracle"] + CONSTS).to_string())

resp_cols = ["startles", "jolts", "freezes", "gt_stops", "gt_sidesteps", "intrusion_steps", "neg_arousal"]
groups = {"prior": ["const_asym_back1"],
          "from_scratch": [f"learned_s{k}" for k in range(1, 6)],
          "residual": [f"residual_s{k}" for k in range(1, 6)]}
resp = pd.concat([pd.read_csv(f"{RESULTS}/{m}.csv").assign(group=g)
                  for g, ms in groups.items() for m in ms])
print("\nGround-truth human responses per episode:")
print(resp.groupby(["kind", "group"])[resp_cols].mean().round(2))