"""What the robot can perceive about humans: reactions from tracked positions,
and (optionally noisy) emotion estimates from a camera."""
from collections import deque

import numpy as np


def robot_is_cause(rel, human_vel, robot_vel, near=2.0, cpa_dist=1.0, cpa_time=3.0):
    """True if the robot is plausibly causing whatever the human does now:
    it is close, or both are on course for a near-pass (closest point of approach).
    ``rel`` = human_pos - robot_pos, world-aligned."""
    if np.linalg.norm(rel) < near:
        return True
    rel_vel = human_vel - robot_vel
    vv = float(rel_vel @ rel_vel)
    if vv < 1e-6:
        return False
    t_cpa = -float(rel @ rel_vel) / vv
    return 0.0 < t_cpa < cpa_time and np.linalg.norm(rel + rel_vel * t_cpa) < cpa_dist


class _Track:
    def __init__(self, rel, maxlen):
        self.prev_rel = rel
        self.vels = deque(maxlen=maxlen)   # ego-motion-compensated world velocities
        self.cause = 0                     # ticks the robot-cause gate stays open
        self.refractory = 0                # ticks until another event may fire


class ReactionDetector:
    """Detects stop / sidestep / freeze / jolt reactions from relative positions
    and the robot's own odometry only, and fires only when the robot is a
    plausible cause. Times are in seconds so the detector is independent of dt.
    Call ``update`` exactly once per simulation step."""

    def __init__(self, dt, ref_time=1.0, smooth_time=0.2, cause_hold_time=1.0,
                 stop_speed=0.15, min_ref_speed=0.2, startle_acc=5.0, freeze_drop=0.7,
                 jolt_speed=0.4, side_speed=0.35, side_fwd_ratio=0.6, disrupt_deadband=0.2):
        ticks = lambda t: max(1, int(round(t / dt)))
        self.dt = dt
        self.ref_ticks, self.smooth_ticks = ticks(ref_time), ticks(smooth_time)
        self.cause_hold = ticks(cause_hold_time)
        # Long enough that the next reference window lies entirely after the event,
        # so a human who stays stopped does not fire again.
        self.refractory_ticks = self.ref_ticks + self.smooth_ticks + 2
        self.stop_speed, self.min_ref_speed = stop_speed, min_ref_speed
        self.startle_acc, self.jolt_speed = startle_acc, jolt_speed
        self.freeze_drop = freeze_drop
        self.side_speed, self.side_fwd_ratio = side_speed, side_fwd_ratio
        self.tracks = {}
        self.disrupt_deadband = disrupt_deadband
        self.disruption = {}

    def reset(self):
        self.tracks.clear()

    def update(self, detections, robot_vel):
        """detections: {human_id: rel_pos}; robot_vel: measured robot velocity.
        Returns {human_id: event} for the events that fired this step."""
        for hid in list(self.tracks):
            if hid not in detections:
                del self.tracks[hid]          # lost track: history no longer contiguous
        events = {}
        for hid, rel in detections.items():
            rel = np.asarray(rel, dtype=np.float32)
            tr = self.tracks.get(hid)
            if tr is None:
                self.tracks[hid] = _Track(rel, self.ref_ticks + self.smooth_ticks + 2)
                continue
            tr.vels.append((rel - tr.prev_rel) / self.dt + robot_vel)
            tr.prev_rel = rel
            now = np.mean(list(tr.vels)[-self.smooth_ticks:], axis=0)

            tr.cause = self.cause_hold if robot_is_cause(rel, now, robot_vel) else max(0, tr.cause - 1)
            tr.refractory = max(0, tr.refractory - 1)
            if len(tr.vels) == tr.vels.maxlen and tr.cause > 0:
                ref = np.mean(list(tr.vels)[:self.ref_ticks], axis=0)
                ref_speed = float(np.linalg.norm(ref))
                if ref_speed >= self.min_ref_speed:
                    dev = float(np.linalg.norm(now - ref)) / ref_speed
                    self.disruption[hid] = max(0.0, dev - self.disrupt_deadband)
            if len(tr.vels) < tr.vels.maxlen or tr.cause == 0 or tr.refractory > 0:
                continue

            event = self._classify(np.array(tr.vels), now, rel)
            if event:
                tr.refractory = self.refractory_ticks
                events[hid] = event
        return events

    def _classify(self, vels, now, rel):
        ref = vels[:self.ref_ticks].mean(axis=0)
        ref_speed = float(np.linalg.norm(ref))
        if ref_speed < self.min_ref_speed:
            return None
        k = self.smooth_ticks
        acc = np.linalg.norm(vels[-1] - vels[-k - 1]) / (k * self.dt)
        heading = ref / ref_speed
        dist = float(np.linalg.norm(rel))
        away = rel / (dist + 1e-6)                              # robot -> human
        lateral = np.array([-heading[1], heading[0]])
        robot_ahead = float(-away @ heading) > 0.2              # robot roughly in front of the walking direction
        speed = float(np.linalg.norm(now))
        fwd = float(now @ heading)
        lat_away = float(now @ lateral) * np.sign(float(away @ lateral))   # > 0: sideways motion away from the robot

        if acc > self.startle_acc and float(now @ away) > self.jolt_speed and dist < 1.2:
            return 'jolt'
        if not robot_ahead or dist > 2.0:
            return None
        if speed < self.stop_speed:
            before = float(np.linalg.norm(vels[-k - 1]))
            return 'freeze' if before > 0 and (before - speed) / before > self.freeze_drop else 'stop'
        if lat_away > self.side_speed and fwd < self.side_fwd_ratio * ref_speed:
            return 'sidestep'
        return None


def perceive_emotion(valence, intensity, rel_orientation, rng):
    """Camera emotion estimate. Recognition accuracy (~65 % for SOTA) falls off
    as the face turns away and is zero beyond 75°. ``rel_orientation`` is the
    angle between the human's heading and the direction to the robot."""
    abs_deg = np.degrees(abs(rel_orientation))
    if valence == 0.0 or abs_deg > 75.0:
        return 0.0, 0.0
    factor = max(abs_deg - 30.0, 0.0) / 45.0              # 0 at 30°, 1 at 75°
    accuracy, jitter = 0.65 - 0.45 * factor, 0.15 + 0.1 * factor
    if rng.random() < accuracy:                           # correct, with jitter
        valence += rng.uniform(-jitter, jitter)
        intensity += rng.uniform(-jitter, jitter)
    elif abs(valence) <= 0.7:                             # mild emotion read as neutral
        valence, intensity = rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.3)
    else:                                                 # strong emotion read with the wrong sign
        valence = -valence + rng.uniform(-jitter, jitter)
        intensity += rng.uniform(-jitter, jitter)
    return float(np.clip(valence, -1.0, 1.0)), float(np.clip(intensity, 0.0, 1.0))
