"""Find, classify and record situations where the robot or a human gets stuck.

Runs the environment (with a trained model, a constant action, or random
actions), watches net displacement over a sliding window, and whenever the
robot and/or a human stops making progress it saves:
  * a PNG snapshot of the scene (render.draw)
  * the last few seconds of state history (JSON lines)
At the end it prints a summary and saves a map of where the events happened.

Usage (from the project folder):
    python trap_debugger.py --env emo --model C:/.../emo_based_no_obstacle --episodes 50
    python trap_debugger.py --env dist --action 0.5 --episodes 50
    python trap_debugger.py --env emo --action random --episodes 50
"""
import argparse
import json
import os
from collections import Counter, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WINDOW_S = 2.0            # s, window for the "no net progress" check
ROBOT_STUCK_DIST = 0.15   # m of robot net displacement within the window
HUMAN_STUCK_DIST = 0.10   # m of human net displacement within the window
HISTORY_S = 3.0           # s of state history saved with each event
HUGE_FRONT_PS = 3.0       # m, a human front PS larger than this is flagged


def make_env(kind):
    if kind == "emo":
        from social_nav_env_no_obst import SocialNavEnv
    else:
        from dist_nav_env_no_obst import SocialNavEnv
    return SocialNavEnv(width=5, height=5, num_humans=1)


def make_policy(args, env):
    if args.model:
        from stable_baselines3 import SAC
        model = SAC.load(args.model, env=env)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if args.action == "random":
        return lambda obs: env.action_space.sample()
    value = float(args.action)
    return lambda obs: np.full(env.action_space.shape, value, dtype=np.float32)


def _r(x, n=2):
    return np.round(np.asarray(x, dtype=float), n).tolist()


def snapshot(env):
    robot, planner = env.robot, env.planner
    humans = []
    for h in env.humans:
        humans.append(dict(
            id=h.human_id, pos=_r(h.pos), goal=_r(h.goal), vel=_r(h.vel),
            response=h.response, startle=h.startle_reaction, aware=bool(h.aware),
            robot_r=round(float(h.robot_r), 2), emotion=_r(h.emotion),
            behaviour=h.behaviour_type,
            ps_front_left_right_back=_r((h.ps_front, h.ps_left, h.ps_right, h.ps_back)),
            dist_to_robot=round(float(np.linalg.norm(h.pos - robot.pos)), 2),
        ))
    return dict(
        step=env.current_step, robot_pos=_r(robot.pos), robot_goal=_r(robot.goal),
        robot_vel=_r(robot.measured_vel(env.dt)),
        planner_mode=planner.debug.get("mode"), safety_triggered=bool(planner.safety_triggered),
        trigger_mode=getattr(planner, "trigger_mode", None), stuck_ticks=int(planner.stuck_ticks),
        action=None if env.action is None else _r(env.action), humans=humans,
    )


def classify(robot_stuck, human, robot_pos):
    """Name the trap from who is stuck and what the human is doing."""
    near = float(np.linalg.norm(human.pos - robot_pos)) < 2.5
    human_stuck = human is not None and human.stuck
    if robot_stuck and human_stuck and near:
        return f"standoff (human {human.response})"
    if robot_stuck:
        return "robot_stuck"
    if human.response in ("stop", "sidestep", "startle"):
        return f"human_held_by_robot ({human.response})"
    return f"human_stuck_while_{human.response}"   # forces / goal trap, not the robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["emo", "dist"], default="emo")
    ap.add_argument("--model", default=None, help="path to a trained model (.zip)")
    ap.add_argument("--action", default="random", help="'random' or a constant PS size, e.g. 0.5")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="trap_debug")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from render import draw

    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.env)
    policy = make_policy(args, env)
    window = max(2, int(round(WINDOW_S / env.dt)))
    history_len = max(1, int(round(HISTORY_S / env.dt)))

    events, flags = [], Counter()
    log_path = os.path.join(args.out, "events.jsonl")
    open(log_path, "w").close()
    fig, ax = plt.subplots(figsize=(8, 8))

    obs, _ = env.reset(seed=args.seed)
    for ep in range(1, args.episodes + 1):
        robot_track = deque(maxlen=window)
        human_tracks = {}
        history = deque(maxlen=history_len)
        active = set()                       # trap labels currently ongoing (no repeat logging)
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(policy(obs))
            done = term or trunc
            history.append(snapshot(env))
            robot = env.robot

            # --- invariants: things that should never happen ---
            if not np.all(np.isfinite(robot.pos)):
                flags["robot_pos_nan"] += 1
            for h in env.humans:
                if not np.all(np.isfinite(h.pos)):
                    flags["human_pos_nan"] += 1
                if np.linalg.norm(h.pos - robot.pos) < h.radius + robot.radius:
                    flags["body_overlap"] += 1
                if h.ps_front > HUGE_FRONT_PS:
                    flags["human_front_ps_over_%.1fm" % HUGE_FRONT_PS] += 1

            # --- progress over the window ---
            robot_track.append(robot.pos.copy())
            robot_stuck = (len(robot_track) == window
                           and np.linalg.norm(robot_track[-1] - robot_track[0]) < ROBOT_STUCK_DIST)
            labels = set()
            for h in env.humans:
                tr = human_tracks.setdefault(h.human_id, deque(maxlen=window))
                tr.append(h.pos.copy())
                h.stuck = (not h.arrived and len(tr) == window
                           and np.linalg.norm(tr[-1] - tr[0]) < HUMAN_STUCK_DIST)
                if robot_stuck or h.stuck:
                    labels.add(classify(robot_stuck, h, robot.pos))

            for label in labels - active:    # log each trap once, at its onset
                ev = dict(episode=ep, label=label, robot_pos=_r(robot.pos), history=list(history))
                events.append(ev)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev) + "\n")
                draw(env, ax)
                ax.set_title(f"ep {ep} step {env.current_step}: {label}", color="white")
                fig.savefig(os.path.join(args.out, f"ep{ep:03d}_step{env.current_step:03d}.png"),
                            facecolor=fig.get_facecolor())
            active = labels

        print(f"episode {ep}: {'goal' if term else 'timeout'} after {env.current_step} steps, "
              f"{sum(e['episode'] == ep for e in events)} trap events")
        obs, _ = env.reset()

    # --- summary ---
    print("\n=== trap events by type ===")
    for label, n in Counter(e["label"] for e in events).most_common():
        print(f"{n:5d}  {label}")
    print("\n=== planner mode at trap onset ===")
    for mode, n in Counter(e["history"][-1]["planner_mode"] for e in events).most_common():
        print(f"{n:5d}  {mode}")
    print("\n=== invariant flags (step counts) ===")
    for k, n in flags.most_common() or [("none", 0)]:
        print(f"{n:5d}  {k}")

    if events:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6))
        kinds = sorted({e["label"] for e in events})
        for k in kinds:
            pts = np.array([e["robot_pos"] for e in events if e["label"] == k])
            ax.scatter(pts[:, 0], pts[:, 1], label=k, s=25, alpha=0.7)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal")
        ax.set_title("Robot position at trap onset"); ax.legend(fontsize=7)
        fig.savefig(os.path.join(args.out, "trap_map.png"), dpi=120)
    print(f"\nSaved snapshots, events.jsonl and trap_map.png to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()