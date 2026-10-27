# eval_utils.py
import csv

import os
import numpy as np
import pandas as pd
import wandb
from stable_baselines3.common.callbacks import BaseCallback

NAMES = ['front', 'left', 'right', 'back']

path = "Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/results/base/learned.csv"
new_file = not os.path.exists(path)

def run_episodes(model, env, n_episodes=10, deterministic=True, seed0=10_000):

    records = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + episode)          # seed -> world reset, same scenarios every eval
        ep = dict(seed=seed0 + episode, steps=0, ret=0.0, success=0, triggers=0, reactions=0,
            human_steps=0, intrusion_steps=0, neg_arousal=0.0, min_dist=np.inf,
            startles=0, jolts=0, freezes=0, gt_stops=0, gt_sidesteps=0)
        prev_resp = {}
        ep_deltas, prev_a, prev_seen, done = [], None, False, False
        ep["kind"] = env.encounter
        ps_sum, ps_n = np.zeros(4), 0
        t, disc_ret = 0, 0.0
        while not done:
            seen = bool(obs['agents'][0, env.ID_COL] != 0)
            action, _ = model.predict(obs, deterministic=deterministic)
            a = np.asarray(action, dtype=np.float32).reshape(env.TOP_N, env.ACTION_WIDTH)
            if seen and prev_seen:                    # only where the action actually shapes the costmap
                ep_deltas.append(np.abs(a - prev_a)[0])
            prev_a, prev_seen = a, seen
            if seen:
                ps_sum += a[0]; ps_n += 1

            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            disc_ret += (0.99 ** t) * float(r)
            t += 1
            for key in ("soc_pen", "move", "collision", "shaping", "human_reaction", "emo_shaping"):
                ep[f"r_{key}"] = ep.get(f"r_{key}", 0.0) + float(info.get(f"r_{key}", 0.0))

            ep["steps"] += 1
            ep["ret"] += float(r)
            ep["human_steps"] += int(seen)
            ep["triggers"] += int(info["collision_reward"] < 0)
            ep["reactions"] += len(info["reactions"])
            for h in env.humans:                         # ground truth, not perception
                ep["min_dist"] = min(ep["min_dist"], float(np.linalg.norm(h.pos - env.robot.pos)))
                ep["intrusion_steps"] += int(h.robot_r < 1.0)
                v_h, i_h = h.emotion
                ep["neg_arousal"] += max(-v_h, 0.0) * i_h

                now, before = h.response, prev_resp.get(h.human_id, "walk")
                if now != before:
                    if now == "startle":
                        ep["startles"] += 1
                        ep["jolts"] += int(h.startle_reaction == "jolt")
                        ep["freezes"] += int(h.startle_reaction == "freeze")
                    elif now == "stop" and before != "startle":   # a startle always ends in "stop"; don't double count
                        ep["gt_stops"] += 1
                    elif now == "sidestep":
                        ep["gt_sidesteps"] += 1
                prev_resp[h.human_id] = now
        ep["disc_ret"] = disc_ret
        for i, n in enumerate(NAMES):
            ep[f"ps_{n}"] = float(ps_sum[i] / ps_n) if ps_n else np.nan
        ep["success"] = int(term)
        d = np.array(ep_deltas) if ep_deltas else np.zeros((0, len(NAMES)))
        ep["delta_mean"] = float(d.mean()) if d.size else 0.0
        ep["jump_frac"] = float((d > 0.5).mean()) if d.size else 0.0
        for i, n in enumerate(NAMES):
            ep[f"delta_{n}"] = float(d[:, i].mean()) if d.size else 0.0
        records.append(ep)

    return pd.DataFrame(records)

def summarize(df, tag):
    """Episode table -> scalar metrics (for wandb / quick comparison)."""
    per100 = lambda col: 100 * df[col].sum() / max(df["human_steps"].sum(), 1)
    return {
        f"{tag}/return": df["ret"].mean(),
        f"{tag}/success_rate": df["success"].mean(),
        f"{tag}/steps": df["steps"].mean(),
        f"{tag}/delta_mean": df["delta_mean"].mean(),
        f"{tag}/jump_frac": df["jump_frac"].mean(),
        f"{tag}/intrusions_per_100": per100("intrusion_steps"),
        f"{tag}/neg_arousal_per_100": per100("neg_arousal"),
        f"{tag}/reactions_per_100": per100("reactions"),
        f"{tag}/triggers_per_100": per100("triggers"),
        f"{tag}/min_dist": df["min_dist"].replace(np.inf, np.nan).mean(),
        f"{tag}/discounted_return": df["disc_ret"].mean(),
    }

def evaluate_to_csv(model, env, path, method, scenario, **kw):
    """Run episodes, save the per-episode table, return it."""
    df = run_episodes(model, env, **kw)
    N = kw.get("n_episodes", 50)
    assert len(df) == N and df["seed"].nunique() == N, (len(df), df["seed"].nunique())
    df.insert(0, "method", method)
    df.insert(1, "scenario", scenario)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

class DeterministicEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_episodes=10):
        super().__init__()
        self.eval_env, self.eval_freq, self.n_episodes = eval_env, eval_freq, n_episodes

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            metrics = {"time/total_timesteps": self.num_timesteps}
            for det, tag in ((True, "eval_det"), (False, "eval_stoch")):
                df = run_episodes(self.model, self.eval_env, self.n_episodes, det)
                metrics.update(summarize(df, tag))
            wandb.log(metrics)
        return True