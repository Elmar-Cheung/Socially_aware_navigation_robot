
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from smooth_sac import SmoothSAC
from social_nav_env_no_obst import SocialNavEnv
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

RESULTS = "C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base"

def trace_episode(model, env, seed, deterministic=True):
    obs, _ = env.reset(seed=seed)
    rows, done, t = [], False, 0
    while not done:
        seen = bool(obs["agents"][0, env.ID_COL] != 0)
        action, _ = model.predict(obs, deterministic=deterministic)
        a = np.asarray(action, dtype=np.float32).reshape(env.TOP_N, env.ACTION_WIDTH)[0]
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

        h = env.humans[0]
        lx, ly = h._to_local(env.robot.pos) if h.orientation is not None else (np.nan, np.nan)
        rows.append(dict(
            t=t * env.dt, seen=seen,
            ps_front=a[0], ps_left=a[1], ps_right=a[2], ps_back=a[3],
            dist=float(np.linalg.norm(h.pos - env.robot.pos)),
            robot_x_in_human=lx,          # >0: robot in front of human, <0: behind
            robot_y_in_human=ly,          # >0: robot on human's left
            robot_r=h.robot_r,            # <1: inside the human's true PS
            true_front=h.ps_front, true_back=h.ps_back, true_left=h.ps_left, true_right=h.ps_right,
            response=h.response, valence=h.emotion[0], intensity=h.emotion[1],
            robot_x=env.robot.pos[0], robot_y=env.robot.pos[1], human_x=h.pos[0], human_y=h.pos[1],
        ))
        t += 1
    return pd.DataFrame(rows), env.encounter

def plot_trace(df, kind, path=None):
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for c in ["ps_front", "ps_left", "ps_right", "ps_back"]:
        ax[0].plot(df.t, df[c], label=c.replace("ps_", ""))
    ax[0].set_ylabel("predicted PS (m)"); ax[0].legend(ncol=4, fontsize=8)

    ax[1].plot(df.t, df.dist, label="distance")
    ax[1].plot(df.t, df.robot_x_in_human, label="robot ahead(+)/behind(−)")
    ax[1].axhline(0, color="grey", lw=0.5); ax[1].set_ylabel("m"); ax[1].legend(fontsize=8)

    ax[2].plot(df.t, df.robot_r, label="robot_r")
    ax[2].axhline(1, color="red", ls="--", lw=0.8, label="PS boundary")
    ax[2].set_ylim(0, 4); ax[2].set_ylabel("PS-normalised dist"); ax[2].legend(fontsize=8)

    ax[3].plot(df.t, -df.valence.clip(upper=0) * df.intensity, label="neg. arousal")
    ax[3].set_ylabel("arousal"); ax[3].set_xlabel("time (s)")
    for i, (resp, g) in enumerate(df.groupby((df.response != df.response.shift()).cumsum())):
        r = g.response.iloc[0]
        if r != "walk":
            ax[3].axvspan(g.t.iloc[0], g.t.iloc[-1], alpha=0.15, label=r)
    ax[3].legend(fontsize=8)
    fig.suptitle(f"{kind}"); fig.tight_layout()
    if path: fig.savefig(path, dpi=200)
    plt.show()

def plot_trajectory(df, kind, every=1.0, path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df.robot_x, df.robot_y, "b-", label="robot")
    ax.plot(df.human_x, df.human_y, "r-", label="human")
    step = max(1, int(every / (df.t.iloc[1] - df.t.iloc[0])))
    for _, row in df.iloc[::step].iterrows():
        if not row.seen:
            continue
        # simple symmetric approximation: width = left+right, height = front+back
        ax.add_patch(Ellipse((row.human_x, row.human_y), row.ps_left + row.ps_right,
                             row.ps_front + row.ps_back, fill=False, alpha=0.3, color="orange"))
        ax.plot([row.robot_x, row.human_x], [row.robot_y, row.human_y], "k:", lw=0.5)
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.legend()
    ax.set_title(kind)
    if path: fig.savefig(path, dpi=200)
    plt.show()

env = SocialNavEnv(width=5, height=5, num_humans=1)

methods = {
    "learned1":       SmoothSAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emotion_based_no_obstacle_s1", device="cpu"),
    "learned2":       SmoothSAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emotion_based_no_obstacle_s2", device="cpu"),
    "learned3":       SmoothSAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emotion_based_no_obstacle_s3", device="cpu"),
    "learned4":       SmoothSAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emotion_based_no_obstacle_s4", device="cpu"),
    "learned5":       SmoothSAC.load("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/emotion_based_no_obstacle_s5", device="cpu"),
    }

summary = []
for name, model in methods.items():
    df, kind = trace_episode(model, env, seed=0, deterministic=True)

    plot_trace(df, kind, path="C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base/trace.png")
    plot_trajectory(df, kind, every=0.5, path="C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base/traj.png")


METRICS = ["steps", "intrusion_steps", "neg_arousal", "reactions", "triggers", "discounted return"]


pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
consts = pd.concat([pd.read_csv(f"{RESULTS}/{m}.csv") for m in
                    ["const_0.5", "const_1.0", "const_asym", "const_asym_back1"]])
print(consts.groupby("method")[METRICS].mean().round(2))
print(pd.concat([methods, consts]).groupby(["kind", "method"])[METRICS].mean().round(2))

def seed_avg_tests(R, baselines, seeds=range(1, 6)):
    L = (pd.concat([pd.read_csv(f"{R}/learned_s{k}.csv") for k in seeds])
           .groupby("seed")[METRICS].mean())                      # one row per eval episode
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

print(seed_avg_tests(RESULTS, ["oracle", "const_1.0", "const_asym", "const_asym_back1"]).to_string())
