"""Track a HumanAgent's personal space (PS) under controlled conditions.

Part 1 - steady-state sweeps: vary one factor at a time (walking speed, emotion,
turning, corridor width, crowding, awareness) and let the smoothed PS settle,
to see how each factor scales front / left / right / back.

Part 2 - encounters: a full HumanAgent.step() simulation against a scripted
robot (head-on, offset pass, crossing, overtaking, parked in the path), logging
the PS, distance, the human's response and emotion every tick.

The "back" reported everywhere is the *effective* back the human's perception
uses: once aware of the robot, the rear is guarded as much as the sides.
The policy's action cap (largest PS it can predict) is drawn as a dashed line.

Run from the project folder:  python ps_tracker.py
Outputs: ps_tracking/ (PNG figures, one CSV per encounter, printed summary)
"""
import csv
import math
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from human_agent import HumanAgent, _Circle

OUT = "ps_tracking"
ACTION_CAP = 2.0            # upper bound of the policy's action space (m)
DT = 0.1
WIDTH, HEIGHT = 10.0, 6.0   # roomier than the training map, so walls don't interfere
PS_COLORS = {"front": "#d1495b", "left": "#00798c", "right": "#66a182", "back": "#edae49"}
RESPONSE_COLORS = {"walk": None, "attend": "#2a78d6", "stop": "#eb6834",
                   "sidestep": "#1baf7a", "startle": "#eda100"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_human(speed=1.0, pos=(1.0, 3.0), goal=(9.0, 3.0), seed=0):
    return HumanAgent(pos=pos, goal=goal, group_id=0, human_id=1, group_size=1, speed=speed,
                      width=WIDTH, height=HEIGHT, rng=np.random.default_rng(seed))


def effective_ps(h):
    """(front, left, right, back) as the human's perception uses them."""
    back = max(h.ps_back, h.ps_left, h.ps_right) if h.aware else h.ps_back
    return float(h.ps_front), float(h.ps_left), float(h.ps_right), float(back)


def corridor(gap, x=5.0, y=3.0, r=50.0):
    """Two huge circles whose facing surfaces leave a ``gap`` m wide corridor around y."""
    return [_Circle(np.array([x, y - gap / 2 - r], np.float32), r),
            _Circle(np.array([x, y + gap / 2 + r], np.float32), r)]


# ----------------------------------------------------------------------
# Part 1: steady-state sweeps
# ----------------------------------------------------------------------

def steady_ps(valence=0.0, intensity=0.0, speed_ratio=1.0, turn=0.0, gap=None, crowd=None,
              aware=False, response="walk", iters=80):
    """Hold the situation fixed and iterate the PS update until the smoothing settles."""
    h = make_human()
    h.speed = 1.0                                  # preferred speed, so actual_speed = ratio
    h.actual_speed = speed_ratio
    h.emotion = (valence, intensity)
    h.aware, h.response = aware, response
    h.ori_diff = turn                              # heading change per tick
    obstacles = corridor(gap) if gap is not None else []
    neighbours = [SimpleNamespace(pos=h.pos + np.array([0.0, d * s], np.float32))
                  for d, s in zip(crowd or [], (1.0, -1.0, 1.0, -1.0))]
    others = [h] + neighbours
    edges = h._local_gap_edges(obstacles)
    for _ in range(iters):
        h._update_ps_limits(others, edges)
        h._apply_turn_limits()
        h._apply_speed_limits()
    return effective_ps(h)


SWEEPS = [
    # (title, parameter, values, fixed kwargs, value -> kwarg converter)
    ("Walking speed", "speed_ratio", np.linspace(0.0, 1.5, 16), {}, None),
    ("Aware and stopped (watching the robot)", "speed_ratio", np.linspace(0.0, 1.5, 16),
     {"aware": True, "response": "stop"}, None),
    ("Negative to positive emotion (intensity 1)", "valence", np.linspace(-1.0, 1.0, 21),
     {"intensity": 1.0}, None),
    ("Turning", "turn", np.linspace(0.0, math.pi, 13), {}, None),
    ("Corridor width", "gap", np.linspace(0.8, 3.4, 14), {}, None),
    ("Two neighbours at distance", "crowd", np.linspace(0.3, 1.1, 9), {}, lambda d: [d, d]),
]
X_LABELS = {"speed_ratio": "actual / preferred speed", "valence": "valence",
            "turn": "heading change per tick (rad)", "gap": "corridor width (m)",
            "crowd": "neighbour distance (m)"}


def run_sweeps():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (title, param, values, fixed, conv) in zip(axes.ravel(), SWEEPS):
        ps = np.array([steady_ps(**fixed, **{param: conv(v) if conv else v}) for v in values])
        for j, side in enumerate(PS_COLORS):
            ax.plot(values, ps[:, j], marker="o", ms=3, color=PS_COLORS[side], label=side)
        ax.axhline(ACTION_CAP, color="black", ls="--", lw=1, label="action cap")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(X_LABELS[param]); ax.set_ylabel("PS size (m)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Steady-state personal space by situation (effective back shown)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sweeps.png"), dpi=130)
    plt.close(fig)

    # Front PS over valence x intensity
    vals, ints = np.linspace(-1, 1, 21), np.linspace(0, 1, 11)
    grid = np.array([[steady_ps(valence=v, intensity=i)[0] for v in vals] for i in ints])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma",
                   extent=[vals[0], vals[-1], ints[0], ints[-1]])
    cs = ax.contour(vals, ints, grid, levels=[ACTION_CAP], colors="cyan", linewidths=1.5)
    ax.clabel(cs, fmt="cap %.1f m", fontsize=8)
    fig.colorbar(im, label="front PS (m)")
    ax.set_xlabel("valence"); ax.set_ylabel("intensity")
    ax.set_title("Front PS while walking, by emotion")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "front_ps_emotion.png"), dpi=130)
    plt.close(fig)

    base = steady_ps()
    print("Walking, neutral: front %.2f  left %.2f  right %.2f  back %.2f" % base)
    print("Walking, valence -1 / intensity 1: front %.2f  left %.2f  right %.2f  back %.2f"
          % steady_ps(valence=-1.0, intensity=1.0))
    print("Stopped, aware: front %.2f  left %.2f  right %.2f  back %.2f"
          % steady_ps(speed_ratio=0.0, aware=True, response="stop"))
    print("Stopped, unaware: front %.2f  left %.2f  right %.2f  back %.2f"
          % steady_ps(speed_ratio=0.0))


# ----------------------------------------------------------------------
# Part 2: encounters against a scripted robot
# ----------------------------------------------------------------------

class ScriptedRobot:
    radius = 0.25

    def __init__(self, pos, vel):
        self.pos = np.array(pos, np.float32)
        self.prev_pos = self.pos.copy()
        self.vel = np.array(vel, np.float32)

    def step(self, dt):
        self.prev_pos = self.pos.copy()
        lo, hi = np.array([self.radius] * 2), np.array([WIDTH - self.radius, HEIGHT - self.radius])
        self.pos = np.clip(self.pos + self.vel * dt, lo, hi).astype(np.float32)


ENCOUNTERS = {
    # name: (human kwargs, robot start, robot velocity)
    "head_on":            (dict(speed=1.0), (9.0, 3.0), (-0.8, 0.0)),
    "head_on_offset_1m":  (dict(speed=1.0), (9.0, 4.0), (-0.8, 0.0)),
    "crossing":           (dict(speed=1.0), (5.5, 0.3), (0.0, 0.8)),
    "overtaking":         (dict(speed=0.5, pos=(2.5, 3.0)), (0.3, 3.0), (0.8, 0.0)),
    "parked_in_path":     (dict(speed=1.0), (5.0, 3.2), (0.0, 0.0)),
}


def run_encounter(hkw, rpos, rvel, seconds=12.0, seed=0):
    h = make_human(seed=seed, **hkw)
    robot = ScriptedRobot(rpos, rvel)
    rows = []
    for k in range(int(seconds / DT)):
        robot.step(DT)                         # same order as the env: robot first, then humans
        h.step(DT, [h], robot, [])
        f, l, r, b = effective_ps(h)
        rows.append(dict(t=round((k + 1) * DT, 2), front=f, left=l, right=r, back=b,
                         raw_back=float(h.ps_back),
                         dist=float(np.linalg.norm(h.pos - robot.pos)),
                         robot_r=float(min(h.robot_r, 5.0)), speed=float(h.actual_speed),
                         valence=float(h.emotion[0]), intensity=float(h.emotion[1]),
                         response=h.response, aware=bool(h.aware), event=h.behaviour_type))
        if h.arrived:
            break
    return rows


def shade_responses(ax, t, responses):
    start = 0
    for k in range(1, len(responses) + 1):
        if k == len(responses) or responses[k] != responses[start]:
            color = RESPONSE_COLORS.get(responses[start])
            if color:
                ax.axvspan(t[start] - DT / 2, t[k - 1] + DT / 2, color=color, alpha=0.15, lw=0)
            start = k


def plot_encounter(name, rows):
    t = np.array([r["t"] for r in rows])
    resp = [r["response"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    for side in PS_COLORS:
        ax.plot(t, [r[side] for r in rows], color=PS_COLORS[side], label=side)
    ax.axhline(ACTION_CAP, color="black", ls="--", lw=1, label="action cap")
    ax.set_ylabel("PS size (m)"); ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, [r["dist"] for r in rows], color="black", label="distance (m)")
    ax.plot(t, [r["robot_r"] for r in rows], color="purple", label="robot depth r (<1 = inside PS)")
    ax.axhline(1.0, color="purple", ls=":", lw=1)
    ax.set_ylabel("m / PS units"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, [r["valence"] for r in rows], color="#e8554e", label="valence")
    ax.plot(t, [r["intensity"] for r in rows], color="#555", label="intensity")
    ax.plot(t, [r["speed"] for r in rows], color="#2a78d6", ls="--", label="speed (m/s)")
    ax.set_ylabel("emotion / speed"); ax.set_xlabel("time (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for ax in axes:
        shade_responses(ax, t, resp)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in RESPONSE_COLORS.values() if c]
    labels = [k for k, c in RESPONSE_COLORS.items() if c]
    fig.legend(handles, labels, loc="upper right", fontsize=8, title="response")
    fig.suptitle(f"Encounter: {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"encounter_{name}.png"), dpi=130)
    plt.close(fig)


def summarise(name, rows):
    front = np.array([r["front"] for r in rows])
    inside = np.array([r["robot_r"] < 1.0 for r in rows])
    seen = []
    for r in rows:
        if not seen or seen[-1] != r["response"]:
            seen.append(r["response"])
    print(f"{name:20s} max front {front.max():.2f} m | front > cap {np.mean(front > ACTION_CAP):5.0%} | "
          f"robot inside PS {inside.mean():5.0%} | min dist {min(r['dist'] for r in rows):.2f} m | "
          f"min valence {min(r['valence'] for r in rows):+.2f} | responses {' > '.join(seen)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Part 1: steady-state PS ===")
    run_sweeps()

    print("\n=== Part 2: encounters ===")
    for name, (hkw, rpos, rvel) in ENCOUNTERS.items():
        rows = run_encounter(hkw, rpos, rvel)
        with open(os.path.join(OUT, f"encounter_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_encounter(name, rows)
        summarise(name, rows)
    print(f"\nSaved figures and CSVs to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()