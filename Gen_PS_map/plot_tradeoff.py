import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base_100"
PRIORS = [1.0, 1.1, 1.2, 1.3, 1.5]
GROUPS = {
    "from scratch": ([f"learned_s{k}" for k in range(1, 6)], "tab:blue", "o"),
    "residual":     ([f"residual_s{k}" for k in range(1, 6)], "tab:orange", "^"),
}
METRICS = [("intrusion_steps", "PS intrusion steps / episode"),
           ("neg_arousal", "negative arousal / episode")]


def means(method):
    d = pd.read_csv(f"{RESULTS}/{method}.csv")
    return d["steps"].mean(), {m: d[m].mean() for m, _ in METRICS}


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (metric, label) in zip(axes, METRICS):
    # scaled prior: what a fixed shape can achieve
    pts = [means(f"prior_x{s}") for s in PRIORS]
    xs, ys = [p[0] for p in pts], [p[1][metric] for p in pts]
    ax.plot(xs, ys, "k-o", ms=4, label="scaled prior (fixed shape)")
    for s, x, y in zip(PRIORS, xs, ys):
        ax.annotate(f"×{s}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    # other constants, for context
    for c in ["const_0.5", "const_1.0", "const_asym"]:
        x, m = means(c)
        ax.plot(x, m[metric], "s", color="grey", ms=5)
        ax.annotate(c.replace("const_", ""), (x, m[metric]), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color="grey")

    # learned: each seed small, mean ± std across seeds large
    for name, (methods, color, marker) in GROUPS.items():
        pts = [means(m) for m in methods]
        sx = np.array([p[0] for p in pts]); sy = np.array([p[1][metric] for p in pts])
        ax.plot(sx, sy, marker, color=color, alpha=0.35, ms=5)
        ax.errorbar(sx.mean(), sy.mean(), xerr=sx.std(ddof=1), yerr=sy.std(ddof=1),
                    fmt=marker, color=color, ms=9, capsize=3, label=f"{name} (mean ± std, 5 seeds)")

    ax.set_xlabel("time to goal (steps, 0.1 s each)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Efficiency vs. social cost — homogeneous humans, 100 episodes")
fig.tight_layout()
fig.savefig(f"{RESULTS}/tradeoff.png", dpi=200)
fig.savefig(f"{RESULTS}/tradeoff.pdf")          # vector version for the thesis
print("saved", f"{RESULTS}/tradeoff.png")