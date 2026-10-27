from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Emotion = Tuple[float, float]  # (valence in [-1, 1], intensity in [0, 1])
NEUTRAL: Emotion = (0.0, 0.0)
_RNG = np.random.default_rng()


# ---------------------------------------------------------------------------
# Scenario overrides
# ---------------------------------------------------------------------------
SCENARIO_OVERRIDES: dict = {
    "all": {"stability": -2.1, "plasticity": -1.5, "ps_init": (1.7, 0.65, 0.75)},
    2: {"stability": 1.1, "plasticity": 1.5, "ps_init": (1.5, 0.55, 0.60), "static": True},
}


# ---------------------------------------------------------------------------
# Emotion reactions
# ---------------------------------------------------------------------------
# Each returns (fired, (valence, intensity)); the values are the *peak* of the
# resulting emotion curve. ``approach`` is the robot-blame weight in [0.2, 1].

def norm_personality(val: float) -> float:
    """Map a trait z-score in [-3, 3] to [0, 1]."""
    return (val + 3.0) / 6.0


def _sample_emotion(rng, v_range, i_range, scale, v_max=1.0) -> Emotion:
    valence = rng.uniform(*v_range) * scale
    intensity = rng.uniform(*i_range) * scale
    return float(_clamp(valence, -1.0, v_max)), float(_clamp(intensity, 0.0, 1.0))


def violate_PS(stability, ps_violate, approach, rng=_RNG):
    """Robot inside personal space: likelier and stronger the deeper it is."""
    s = norm_personality(stability)
    scale = _clamp((1.0 - 0.3 * s) * (0.3 + 1.0 * ps_violate), 0.1, 1.5) * approach
    return _sample_emotion(rng, (-0.45, -0.30), (0.45, 0.55), scale)


def collision_reaction(stability, approach, rng=_RNG):
    """Physical contact: always fires, high intensity."""
    s = norm_personality(stability)
    scale = _clamp(1.2 - 0.40 * s, 0.70, 1.80) * approach
    return _sample_emotion(rng, (-0.65, -0.55), (0.65, 0.75), scale, v_max=-0.01)


def disturb_group(stability, approach, triggered_before, rng=_RNG):
    """Robot passes between group members: low probability, low intensity."""
    s = norm_personality(stability)
    a = approach * (1.2 if triggered_before else 1.0)
    if rng.random() >= _clamp(0.35 - 0.15 * s, 0.05, 0.65):
        return False, NEUTRAL
    scale = _clamp(1.0 - 0.30 * s, 0.5, 1.4) * a
    return True, _sample_emotion(rng, (-0.30, -0.15), (0.25, 0.40), scale)


def curious_reaction(stability, plasticity, rng=_RNG):
    """First sight of the robot: mild positive interest."""
    s, p = norm_personality(stability), norm_personality(plasticity)
    if rng.random() >= _clamp(0.3 + 0.5 * p, 0.10, 0.90):
        return False, NEUTRAL
    scale = _clamp(1.0 - 0.30 * s + 0.2 * p, 0.40, 1.6)
    return True, _sample_emotion(rng, (0.25, 0.40), (0.20, 0.50), scale)


def startle_trigger(stability, plasticity, approach, dist_severity, rng=_RNG):
    """Startle response: 'freeze' (stop and stare) or 'jolt' (step away)."""
    s, p = norm_personality(stability), norm_personality(plasticity)
    base = 1.0 - 0.30 * s + 0.4 * dist_severity
    if rng.random() < _clamp(0.6 - 0.4 * p - 0.2 * s, 0.05, 0.95):
        scale = _clamp(base, 0.50, 1.8) * approach
        return "freeze", _sample_emotion(rng, (-0.65, -0.55), (0.55, 0.65), scale)
    scale = _clamp(base + 0.2, 0.6, 2.2) * approach
    return "jolt", _sample_emotion(rng, (-0.60, -0.50), (0.65, 0.75), scale)


@dataclass
class _Percept:
    """What the human perceives of the robot this tick."""
    dt: float
    robot: object
    mates: list
    dist: float          # m, centre to centre
    r: float             # PS-normalised distance: <1 inside the PS ellipse
    in_fov: bool
    moving: bool         # robot speed above ROBOT_MOVING_SPEED
    closing: bool        # robot moving towards the human
    receding: bool       # robot moving away from the human
    robot_vel: np.ndarray
    closing_speed: float  # m/s towards the human
    approach: float      # robot-blame weight in [0.2, 1]


# ---------------------------------------------------------------------------
# HumanAgent
# ---------------------------------------------------------------------------

class HumanAgent:
    # --- Body and locomotion -------------------------------------------------
    RADIUS = 0.25                   # m, physical body radius
    MAX_SPEED = 1.5                 # m/s, before the emotional boost
    ARRIVAL_RADIUS = 0.05           # m
    ENABLE_RANDOM_PAUSES = False    # spontaneous stops unrelated to the robot

    # --- Perception ----------------------------------------------------------
    FOV_HALF_ANGLE = np.deg2rad(90)   # human horizontal visual field is ~200°
    NOTICE_DIST = 4.0               # m, a robot in view this close has been seen (no startle)
    ATTEND_ENTER = 1.5              # PS units: robot noticed at 1.5x the PS ellipse
    ATTEND_EXIT = 1.8               # PS units: hysteresis before walking on
    AWARE_RELEASE_DIST = 2.0        # m, a noticed robot is forgotten only beyond this and receding
    YIELD_EXIT = 1.2                # PS units: robot must be this far out to resume
    STARTLE_MARGIN = 0.4            # m beyond body contact: an unnoticed robot this close
                                    # startles even if the (small) rear PS is not yet entered
    ROBOT_MOVING_SPEED = 0.1        # m/s, slower robots are treated as obstacles
    CLOSING_SPEED = 0.05            # m/s towards the human counts as approaching
    STARTLE_CLOSING = 0.2           # m/s: the robot must be approaching at least this fast to startle

    # --- Responses -----------------------------------------------------------
    ATTEND_SPEED_MULT = 0.4         # walking speed while watching the robot
    TURN_RATE = 3.0                 # rad/s when turning to face the robot
    SIDESTEP_SPEED = 0.6            # m/s
    SIDESTEP_TICKS = (30, 40)       # upper bound; the step ends once clear of the robot's path
    SIDESTEP_CLEARANCE = 0.3        # m of free space kept beyond body contact
    COLLISION_JOLT_PROB = 0.5       # at stability 0
    CONTACT_FORCE_STRENGTH = 10.0

    # --- Emotion dynamics ----------------------------------------------------
    TAU_RISE = 0.3                  # s, onset of a reaction
    TAU_DECAY = 1.0                 # s at stability 0; lower stability lingers longer
    EMOTION_FLOOR = 0.02            # intensity below which the episode ends
    # Share of an event's magnitude that lasts only while its cause persists.
    # The rest (the immediate, reflexive part) always plays out once triggered.
    SUSTAINED_FRACTION = {"violate_ps": 0.6}
    TAU_RELEASE = 0.5               # s, fade of the sustained part once the cause is gone

    # --- Social force model --------------------------------------------------
    GOAL_FORCE_STRENGTH = 2.5
    SOCIAL_FORCE_STRENGTH = 0.5
    SOCIAL_FORCE_RANGE = 1.75       # m
    GROUP_FORCE_STRENGTH = 2.0
    GROUP_FORCE_RANGE = 4.0         # m, tanh saturation scale
    GROUP_DESIRED_DIST = 2.0        # m, comfortable side-by-side spacing
    GROUP_SIDE_BIAS = 1.6           # >1 favours walking abreast over single file
    GROUP_LAG_DECAY = 0.5           # 1/m, how sharply leaders slow for laggards
    WALL_MARGIN = 0.2               # m
    WALL_FORCE_STRENGTH = 10.0
    OBSTACLE_MARGIN_CLOSE = 0.2     # m from surface, hard repulsion zone
    OBSTACLE_MARGIN_FAR = 1.0       # m from surface, soft repulsion zone
    OBSTACLE_FORCE_STRENGTH = 10.0
    CURIOSITY_RANGE = 4.0           # m

    # --- Corridor lane keeping (assumes corridors run along the x-axis) ------
    CORRIDOR_WIDTH = 3.0            # m, narrower gaps count as corridors
    LANE_KEEP_SIDE = -1.0           # fallback side: +1 keep left, -1 keep right
    LANE_KEEP_TARGET_RATIO = 0.2    # preferred offset from near wall, fraction of gap
    LANE_KEEP_MIN_STANDOFF_BUFFER = -0.10
    LANE_KEEP_STRENGTH = 5.0        # must out-muscle the soft obstacle repulsion
    LANE_KEEP_DAMPING = 8.0         # ~critical damping of the lateral spring

    # --- Orientation ---------------------------------------------------------
    ORIENTATION_SMOOTHING = 0.5
    ORIENTATION_MIN_DISP = 1e-3     # m per tick below which heading is not updated
    BACKSTEP_ANGLE = np.deg2rad(100)
    BACKSTEP_SMOOTHING = 0.03       # slower reorientation while stepping backwards
    BACKSTEP_HOLD_TICKS = 25

    # --- Personal space ------------------------------------------------------
    PS_SMOOTHING = 0.30
    PS_SPEED_REF = (2.0, 1.3, 0.9)  # front/side/back scale at normal walking speed (Gérin-Lajoie)
    MIN_SIDE_PS_RATIO = 0.45        # floor on corridor squeeze
    CROWD_RADIUS = 1.0              # m, neighbours closer than this compress PS

    # --- Startle movement and pauses (ticks) ---------------------------------
    FREEZE_DURATION_RANGE = (8, 20)
    JOLT_DURATION_RANGE = (3, 8)
    PAUSE_COOLDOWN_RANGE = (50, 150)
    PAUSE_DURATION_RANGE = (10, 60)
    PAUSE_RAMP = 0.05               # speed-multiplier change per tick

    # --- Turning -------------------------------------------------------------
    TURN_RATE_REF = 1.2             # rad/s at which the full shrink applies (~70°/s)
    TURN_ATTACK_TAU = 0.15          # s, shrink builds quickly once turning starts
    TURN_RELEASE_TAU = 0.8          # s, PS recovers slowly after the turn
    TURN_SHRINK = (0.6, 0.4, 0.5)   # max fractional shrink: front, side, back
    TURN_SHRINK_WHILE_WATCHING = False  # turning to face the robot does not shrink the PS

    def __init__(self, pos, goal, group_id, human_id, group_size, speed, width, height, *,
                 stability: Optional[float] = None, plasticity: Optional[float] = None,
                 ps_init: Optional[Tuple[float, float, float]] = None,
                 static: Optional[bool] = None, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        cfg = {**SCENARIO_OVERRIDES.get("all", {}), **SCENARIO_OVERRIDES.get(human_id, {})}

        # Identity and world
        self.group_id, self.human_id, self.group_size = group_id, human_id, group_size
        self.width, self.height = width, height
        self.radius = self.RADIUS
        self.static = cfg.get("static", False) if static is None else static

        # Kinematics
        self.pos = np.array(pos, dtype=np.float32)
        self.prev_pos = None
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.base_speed = speed         # group's nominal speed
        self.speed = speed              # target speed after group synchronisation
        self.actual_speed = 0.0
        heading = self._goal_heading()
        self.orientation = None if heading is None else math.atan2(heading[1], heading[0])
        self.ori_diff = None            # signed heading change on the last tick
        self.backstep_ticks = 0
        self.arrived = False
        self.turn_level = 0.0           # filtered turn intensity in [0, 1]

        # Personality
        self.stability = stability if stability is not None else cfg.get("stability", self._sample_trait())
        self.plasticity = plasticity if plasticity is not None else cfg.get("plasticity", self._sample_trait())

        # Emotion: ``emotion`` is the felt state, ``_target`` what it moves towards
        self.behaviour_type = "regular" # label of the latest emotional event
        self.emotion: Emotion = NEUTRAL
        self._t_fast: Emotion = NEUTRAL # immediate part of the target, always decays normally
        self._t_sus: Emotion = NEUTRAL  # sustained part, fades fast once its cause is gone
        self.cause_present = False      # is the cause of the current episode still there?
        self.tau_decay = self.TAU_DECAY / _clamp(1.0 + 0.3 * self.stability, 0.2, 2.0)
        self._peak_gain = None          # set on first step, needs dt
        self.triggered_before = False   # sensitised by an earlier negative event
        self.curiosity_rolled = False   # curiosity is rolled once per encounter

        # Response to the robot
        self.response = "walk"          # 'walk' | 'attend' | 'stop' | 'sidestep' | 'startle'
        self.aware = False              # has noticed the robot in the current encounter
        self.in_contact = False
        self._robot_ahead = True
        self.robot_r = math.inf
        self.sidestep_dir = np.zeros(2, dtype=np.float32)
        self.sidestep_ticks = 0
        self.startle_reaction = None    # 'freeze' | 'jolt' | None
        self.startle_phys_timer = 0

        # Spontaneous pauses (disabled unless ENABLE_RANDOM_PAUSES)
        self.pause_mode = None          # 'decelerate' | 'accelerate' | None
        self.pause_speed_mult = 1.0
        self.pause_timer = 0
        self.pause_cooldown = self._randint(self.PAUSE_COOLDOWN_RANGE)

        # Personal space: init_* are the relaxed sizes, temp_* the situational
        # upper limits, ps_* the smoothed values actually used.
        f, s, b = ps_init or cfg.get("ps_init") or self._sample_personal_space()
        self.init_ps_f, self.init_ps_s, self.init_ps_b = f, s, b
        self.ps_front, self.ps_back, self.ps_left, self.ps_right = f, b, s, s
        self.temp_ps_f, self.temp_ps_b, self.temp_ps_left, self.temp_ps_right = f, b, s, s
        # PS-normalised depth at which an intrusion upsets the human. Clamped so
        # that very stable/open humans still react to a deep intrusion.
        self.ps_inner = _clamp(0.6 - self.stability / 3.0 - self.plasticity / 3.0, 0.3, 1.0)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def step(self, dt, other_humans, robot, obstacles):
        if self.arrived:
            return
        self.ori_diff = 0.0
        if self.prev_pos is not None and not self._watching():
            self._update_orientation()
        self.prev_pos = self.pos.copy()

        mates = [h for h in other_humans if h is not self and h.group_id == self.group_id]
        if self.ENABLE_RANDOM_PAUSES:
            self._update_pause_state()

        p = self._perceive(dt, robot, mates)
        self.robot_r = p.r                          # PS-normalised robot distance (for logging)
        self._update_events(p)
        self._update_response(p)
        self._update_emotion(dt, p)
        if self._watching():
            self._face_robot(dt, robot)

        if not self.static:
            gap_edges = self._local_gap_edges(obstacles)
            self._update_ps_limits(other_humans, gap_edges)
            self._move(dt, other_humans, mates, p, obstacles, gap_edges)
        self._apply_turn_limits(dt)
        self._apply_speed_limits()
        if not self.static:
            self._check_arrival()

    def feature_vector(self):
        """Observation row: [x, y, valence, intensity, orientation, speed, human_id]."""
        valence, intensity = self.emotion
        return [float(self.pos[0]), float(self.pos[1]), float(valence), float(intensity),
                float(self.orientation or 0.0), float(self.actual_speed), float(self.human_id or 0)]

    @property
    def speed_mult(self):
        """How much this agent currently holds its group back (1 = not at all)."""
        if self.response in ("stop", "sidestep") or (self.response == "startle" and self.startle_reaction == "freeze"):
            return 0.0
        mult = self.ATTEND_SPEED_MULT if self.response == "attend" else 1.0
        return mult * self.pause_speed_mult

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _perceive(self, dt, robot, mates):
        diff = self.pos - robot.pos
        dist = _norm(diff)
        robot_vel = (np.zeros(2, dtype=np.float32) if robot.prev_pos is None
                     else ((robot.pos - robot.prev_pos) / dt).astype(np.float32))
        unit = diff / dist if dist > 1e-8 else np.zeros(2)
        robot_closing = float(np.dot(robot_vel, unit))      # >0: robot moves towards human
        human_closing = float(np.dot(self.vel, -unit))      # >0: human moves towards robot

        if self.orientation is None:
            r, in_fov = math.inf, False
        else:
            lx, ly = self._to_local(robot.pos)
            # Once aware of the robot, the human guards the space behind as much as its sides.
            back = max(self.ps_back, self.ps_left, self.ps_right) if self.aware else self.ps_back
            sx = self.ps_front if lx >= 0 else back
            sy = self.ps_left if ly >= 0 else self.ps_right
            r = math.hypot(lx / sx, ly / sy)
            in_fov = abs(math.atan2(ly, lx)) <= self.FOV_HALF_ANGLE

        rc, hc = max(robot_closing, 0.0), max(human_closing, 0.0)
        approach = 1.0 if rc + hc < 0.05 else _clamp(rc / (rc + hc), 0.2, 1.0)
        return _Percept(dt=dt, robot=robot, mates=mates, dist=dist, r=r, in_fov=in_fov,
                        moving=_norm(robot_vel) > self.ROBOT_MOVING_SPEED,
                        closing=robot_closing > self.CLOSING_SPEED,
                        receding=robot_closing < -self.CLOSING_SPEED,
                        robot_vel=robot_vel, closing_speed=robot_closing,
                        approach=approach)

    # ------------------------------------------------------------------
    # Events -> emotional targets
    # ------------------------------------------------------------------

    def _update_events(self, p):
        # Awareness: the robot has been seen (in view and near), or noticed
        # earlier and still close. Only an unseen robot can startle.
        if p.in_fov and (p.dist < self.NOTICE_DIST or p.r < self.ATTEND_ENTER):
            self.aware = True
        elif p.r > self.ATTEND_EXIT and p.dist > self.AWARE_RELEASE_DIST and not p.closing:
            self.aware = False

        self._check_collision(p)
        if p.moving and not self.aware and p.closing_speed > self.STARTLE_CLOSING and (p.r < 1.0 or p.dist < self._contact_dist(p.robot) + self.STARTLE_MARGIN):
            self._check_startle(p)
        if p.r < 1.0 and self.aware:
            self._check_violate_ps(p)
        self._check_disturb_group(p)
        self._check_curiosity(p)

    def _check_collision(self, p):
        contact = self._contact_dist(p.robot)
        if p.dist >= contact:
            if p.dist > contact + 0.2:
                self.in_contact = False
            return
        if self.in_contact:
            return
        self.in_contact = self.aware = True
        emotion = collision_reaction(self.stability, p.approach, self.rng)
        self._set_target("collision", emotion)
        jolt_prob = _clamp(self.COLLISION_JOLT_PROB - 0.15 * self.stability, 0.1, 0.9)
        if self.startle_phys_timer <= 0 and self.rng.random() < jolt_prob:
            self._start_startle_motion("jolt", p, intensity= emotion[1])

    def _check_startle(self, p):
        """Robot entered the PS without being noticed (e.g. from behind)."""
        self.aware = True
        severity = _clamp(1.0 - (p.dist - self._contact_dist(p.robot)) / self.STARTLE_MARGIN, 0.0, 1.0)
        reaction, emotion = startle_trigger(self.stability, self.plasticity, p.approach, severity, self.rng)
        self._set_target("startled", emotion)
        self._start_startle_motion(reaction, p, emotion[1])

    def _check_violate_ps(self, p):
        """Re-sampled every tick while the robot is deeper than ``ps_inner``.
        ``_set_target`` keeps the strongest target, so the emotion stays near
        its peak for as long as the intrusion lasts."""
        if p.r <= self.ps_inner:
            d = p.r ** 2
            violation = (math.exp(1.0 - d) - 1.0) / (math.e - 1.0)
            emotion = violate_PS(self.stability, violation, p.approach, self.rng)
            self._set_target("violate_ps", emotion)

    def _check_disturb_group(self, p):
        robot = p.robot
        if robot.prev_pos is None or not any(
                _segments_intersect(robot.prev_pos, robot.pos, self.pos, m.pos) for m in p.mates):
            return
        fired, emotion = disturb_group(self.stability, p.approach, self.triggered_before, self.rng)
        if fired:
            self._set_target("disturb_group", emotion)

    def _check_curiosity(self, p):
        """Rolled once when the robot first comes into view within range."""
        if p.dist >= self.CURIOSITY_RANGE:
            self.curiosity_rolled = False            # encounter over, may roll again next time
            return
        if self.curiosity_rolled or not p.in_fov:
            return
        self.curiosity_rolled = True
        fired, emotion = curious_reaction(self.stability, self.plasticity, self.rng)
        if fired:
            self._set_target("curious", emotion, sensitise=False)

    @property
    def _target(self) -> Emotion:
        """Total emotional target (immediate + sustained parts)."""
        return (self._t_fast[0] + self._t_sus[0], self._t_fast[1] + self._t_sus[1])

    def _set_target(self, label, emotion, sensitise=True):
        """Adopt a new emotional target if it is at least as strong as the
        current one; negative events always override positive ones. The
        target is split into an immediate and a sustained part."""
        v, i = emotion
        tv, ti = self._target
        if abs(v) * i >= abs(tv) * ti or (v < 0 < tv):
            f = self.SUSTAINED_FRACTION.get(label, 0.0)
            self._t_fast = (v * (1.0 - f), i * (1.0 - f))
            self._t_sus = (v * f, i * f)
            self.behaviour_type = label
        if sensitise:
            self.triggered_before = True

    def _cause_present(self, p):
        """Is the cause of the current episode still there?"""
        if self.behaviour_type == "violate_ps":
            return p.r < 1.0 and not p.receding       # robot still inside the PS and not backing off
        return True                                    # other events: no early release (yet)

    # ------------------------------------------------------------------
    # Emotion dynamics
    # ------------------------------------------------------------------

    def _update_emotion(self, dt, p):
        """Felt emotion chases a decaying target: fast rise, slow fall. The
        sustained part of the target fades quickly once its cause is gone."""
        if self._peak_gain is None:
            self._peak_gain = self._compute_peak_gain(dt)
        decay = math.exp(-dt / self.tau_decay)
        self.cause_present = self._cause_present(p)
        sus_decay = decay if self.cause_present else math.exp(-dt / self.TAU_RELEASE)
        rise = 1.0 - math.exp(-dt / self.TAU_RISE)
        tv, ti = self._target
        v, i = self.emotion
        v += (tv * self._peak_gain - v) * rise
        i += (ti * self._peak_gain - i) * rise
        self._t_fast = (self._t_fast[0] * decay, self._t_fast[1] * decay)
        self._t_sus = (self._t_sus[0] * sus_decay, self._t_sus[1] * sus_decay)
        self.emotion = (_clamp(v, -1.0, 1.0), _clamp(i, 0.0, 1.0))
        if ti < self.EMOTION_FLOOR and i < self.EMOTION_FLOOR:
            self.emotion, self._t_fast, self._t_sus = NEUTRAL, NEUTRAL, NEUTRAL
            self.behaviour_type = "regular"

    def _compute_peak_gain(self, dt):
        """Gain so the felt curve peaks at the sampled event magnitude."""
        decay, rise = math.exp(-dt / self.tau_decay), 1.0 - math.exp(-dt / self.TAU_RISE)
        target, felt, peak = 1.0, 0.0, 0.0
        for _ in range(int(20 * self.tau_decay / dt) + 1):
            felt += (target - felt) * rise
            target *= decay
            if felt < peak:
                break
            peak = felt
        return 1.0 / max(peak, 1e-6)

    # ------------------------------------------------------------------
    # Response state machine
    # ------------------------------------------------------------------

    def _update_response(self, p):
        heading = self._goal_heading()
        self._robot_ahead = heading is None or float(np.dot(p.robot.pos - self.pos, heading)) > 0.0
        if self.startle_phys_timer > 0:
            self.response = "startle"
            return
        if not (self.aware and p.moving):
            self.response = "walk"                    # unaware, or robot parked (obstacle)
            return

        if self.response in ("stop", "sidestep") and ((p.r < 1.0 and not p.receding)
                                                     or (p.closing and p.r < self.YIELD_EXIT)):
            if self.response == "sidestep":
                self.sidestep_ticks -= 1
                if self.sidestep_ticks <= 0 or self._clear_of_path(p):
                    self.response = "stop"
        elif p.r < 1.0 and not p.receding:
            self._start_yield(p)
        elif p.closing and p.r < (self.ATTEND_EXIT if self.response != "walk" else self.ATTEND_ENTER):
            self.response = "attend"
        else:
            self.response = "walk"

    def _clear_of_path(self, p):
        """True once the human is far enough from the robot's line of travel."""
        speed = _norm(p.robot_vel)
        if speed < 1e-6:
            return True
        normal = np.array([-p.robot_vel[1], p.robot_vel[0]]) / speed
        offset = abs(float(np.dot(normal, self.pos - p.robot.pos)))
        return offset >= self._contact_dist(p.robot) + self.SIDESTEP_CLEARANCE

    def _start_yield(self, p):
        """Robot inside the PS: stop, or step out of its path."""
        p_side = _clamp(0.5 + 0.12 * self.plasticity + 0.08 * self.stability, 0.1, 0.9)
        speed = _norm(p.robot_vel)
        if speed < 1e-6 or self.rng.random() >= p_side:
            self.response = "stop"
            return
        normal = np.array([-p.robot_vel[1], p.robot_vel[0]], dtype=np.float32) / speed
        side = float(np.dot(normal, self.pos - p.robot.pos))
        sign = (1.0 if side > 0 else -1.0) if abs(side) > 0.05 else float(self.rng.choice((-1.0, 1.0)))
        self.sidestep_dir = normal * sign
        self.sidestep_ticks = self._randint(self.SIDESTEP_TICKS)
        self.response = "sidestep"

    def _start_startle_motion(self, reaction, p, intensity):
        self.startle_reaction = reaction
        self.response = "startle"
        duration = self.FREEZE_DURATION_RANGE if reaction == "freeze" else self.JOLT_DURATION_RANGE
        self.startle_phys_timer = self._randint(duration) * _clamp(1.0 - 0.2 * self.stability, 0.4, 2.0)
        if reaction == "jolt":
            away = self.pos - p.robot.pos
            gain = _clamp(1.0 + 0.1 * self.plasticity, 0.7, 1.3) ** 2 * (0.6 + intensity)
            self.vel = (away / (_norm(away) + 1e-8) * gain).astype(np.float32)

    def _watching(self):
        """True while the body turns to face the robot instead of the walking
        direction. While still walking (attend) the human only turns towards a
        robot ahead of it; a robot behind is tracked without walking backwards."""
        return self.response in ("stop", "sidestep", "startle") or (self.response == "attend" and self._robot_ahead)

    def _face_robot(self, dt, robot):
        diff = robot.pos - self.pos
        if _norm(diff) < 1e-6:
            return
        target = math.atan2(diff[1], diff[0])
        if self.orientation is None or (self.response == "startle" and self.startle_reaction == "freeze"):
            new = target                               # startled stare: snap round
        else:
            turn = _clamp(_wrap(target - self.orientation), -self.TURN_RATE * dt, self.TURN_RATE * dt)
            new = self.orientation + turn
        self.ori_diff = 0.0 if self.orientation is None else _wrap(new - self.orientation)
        self.orientation = float(_wrap(new))
        self.backstep_ticks = 0

    # ------------------------------------------------------------------
    # Pauses
    # ------------------------------------------------------------------

    def _update_pause_state(self):
        if self.pause_timer > 0:                 # fully stopped
            self.pause_timer -= 1
            self.pause_speed_mult = 0.0
            if self.pause_timer <= 0:
                self.pause_mode = "accelerate"
        elif self.startle_phys_timer > 0:        # no new pause mid-startle
            return
        elif self.pause_mode == "decelerate":
            self.pause_speed_mult = max(self.pause_speed_mult - self.PAUSE_RAMP, 0.0)
            if self.pause_speed_mult == 0.0:
                self.pause_timer = self._randint(self.PAUSE_DURATION_RANGE)
        elif self.pause_mode == "accelerate":
            self.pause_speed_mult = min(self.pause_speed_mult + self.PAUSE_RAMP, 1.0)
            if self.pause_speed_mult == 1.0:
                self.pause_mode = None
                self.pause_cooldown = self._randint(self.PAUSE_COOLDOWN_RANGE)
        else:
            self.pause_cooldown -= 1
            if self.pause_cooldown <= 0:
                prob = min(0.15 - 0.08 * self.plasticity - 0.04 * self.stability, 0.95)
                if self.rng.random() < prob:
                    self.pause_mode = "decelerate"

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _move(self, dt, others, mates, p, obstacles, gap_edges):
        robot = p.robot
        robot_body = [_Circle(robot.pos, getattr(robot, "radius", self.RADIUS))]
        if self.response == "startle":
            self._startle_motion(dt, robot)
        else:
            self._sync_group_speed(mates)
            parked_robot = [] if p.moving else [_Circle(robot.pos, getattr(robot, "radius", self.RADIUS))]
            force = (self._drive_force() + self._curiosity_force(robot)
                     + self._detour_force(robot)
                     + self._social_force_humans(others) + self._contact_force(robot)
                     + self._group_force(mates) + self._boundary_force()
                     + self._obstacle_force(list(obstacles) + robot_body)
                     + self._lane_keeping_force(gap_edges))
            valence, intensity = self.emotion
            force = force * (1.0 - valence * intensity)          # upset humans react more strongly
            self.vel = (self.vel + force * dt).astype(np.float32)
            max_speed = self.MAX_SPEED * (1.0 + 0.3 * abs(valence))  # upset humans rush
            speed = _norm(self.vel)
            if speed > max_speed:
                self.vel = (self.vel * (max_speed / speed)).astype(np.float32)
            self.pos = self.pos + self.vel * dt
        self.actual_speed = float(_norm(self.vel))
        self.pos[0] = _clamp(self.pos[0], 0.1, self.width - 0.1)
        self.pos[1] = _clamp(self.pos[1], 0.1, self.height - 0.1)

    def _startle_motion(self, dt, robot):
        """Physical half of the startle response, replacing the SFM while active."""
        if self.startle_reaction == "freeze":
            self.vel = np.zeros(2, dtype=np.float32)
        else:  # jolt: coast away from the initial push, still kept off walls and the robot
            force = self._boundary_force() + self._contact_force(robot)
            self.vel = (self.vel * 0.9 + force * dt).astype(np.float32)
            self.pos = self.pos + self.vel * dt
        self.startle_phys_timer -= 1
        if self.startle_phys_timer <= 0:
            self.startle_reaction = None
            self.response = "stop"                     # regain composure, then re-evaluate

    def _desired_velocity(self):
        """Preferred velocity for the current response."""
        if self.response == "stop":
            return np.zeros(2, dtype=np.float32)
        if self.response == "sidestep":
            return self.sidestep_dir * self.SIDESTEP_SPEED
        heading = self._goal_heading()
        if heading is None:
            return np.zeros(2, dtype=np.float32)
        wander = (_clamp(0.06 + 0.01 * self.stability, 0.03, 0.09)
                  * _clamp(1.0 + 0.2 * self.plasticity, 0.7, 1.3))
        angle = self.rng.normal(0.0, wander)
        c, s = math.cos(angle), math.sin(angle)
        dx, dy = heading
        direction = np.array([dx * c - dy * s, dx * s + dy * c], dtype=np.float32)
        jitter = self.rng.normal(0.0, _clamp(0.03 + 0.01 * self.stability, 0.02, 0.05))
        mult = self.pause_speed_mult * (self.ATTEND_SPEED_MULT if self.response == "attend" else 1.0)
        return direction * max(0.05, self.speed + jitter) * mult

    def _drive_force(self):
        return (self._desired_velocity() - self.vel) * self.GOAL_FORCE_STRENGTH

    def _sync_group_speed(self, mates):
        """Slow down for paused, stopped or lagging group mates."""
        if not mates:
            self.speed += (self.base_speed - self.speed) * 0.1
            return
        hold_mult = min([1.0] + [m.speed_mult for m in mates])
        lag_mult = 1.0
        heading = self._goal_heading()
        if heading is not None:
            for m in mates:
                behind = -np.dot(m.pos - self.pos, heading) - self.GROUP_DESIRED_DIST
                if behind > 0:
                    lag_mult = min(lag_mult, float(np.exp(-self.GROUP_LAG_DECAY * behind)))
        target = self.base_speed * hold_mult * lag_mult
        self.speed += (target - self.speed) * 0.15

    def _curiosity_force(self, robot):
        """Mild pull towards the robot while curious, never inside the front PS."""
        if self.behaviour_type != "curious":
            return np.zeros(2, dtype=np.float32)
        diff = robot.pos - self.pos
        dist = _norm(diff)
        if dist < self.ps_front:
            return np.zeros(2, dtype=np.float32)
        valence, intensity = self.emotion
        return diff / dist * (0.8 * valence * intensity)

    def _detour_force(self, robot):
        """Sideways push around a parked robot blocking the way to the goal.
        Pure radial repulsion deadlocks when the robot is dead ahead."""
        heading = self._goal_heading()
        diff = robot.pos - self.pos
        dist = _norm(diff)
        reach = self._contact_dist(robot) + self.OBSTACLE_MARGIN_FAR
        if heading is None or dist > reach or float(np.dot(diff, heading)) <= 0:
            return np.zeros(2, dtype=np.float32)
        left = np.array([-heading[1], heading[0]], dtype=np.float32)
        side = -1.0 if float(np.dot(diff, left)) > 0 else 1.0    # pass on the side away from the robot
        if abs(float(np.dot(diff, left))) < 0.05:
            side = self.LANE_KEEP_SIDE                            # dead ahead: keep to the default side
        return left * side * self.OBSTACLE_FORCE_STRENGTH * 0.3 * (reach - dist) / reach

    def _contact_force(self, robot):
        """Pushes the bodies apart only when they overlap."""
        diff = self.pos - robot.pos
        dist = _norm(diff)
        contact = self._contact_dist(robot)
        if 0 < dist < contact:
            return diff / dist * self.CONTACT_FORCE_STRENGTH * (contact - dist) / contact
        return np.zeros(2, dtype=np.float32)

    def _social_force_humans(self, others):
        """Exponential repulsion from humans outside the agent's own group."""
        force = np.zeros(2, dtype=np.float32)
        for other in others:
            if other is self or other.group_id == self.group_id:
                continue
            diff = self.pos - other.pos
            dist = _norm(diff)
            if 0 < dist < self.SOCIAL_FORCE_RANGE:
                force += diff / dist * self.SOCIAL_FORCE_STRENGTH * np.exp(-dist / self.SOCIAL_FORCE_RANGE)
        return force

    def _group_force(self, mates):
        """Spring towards GROUP_DESIRED_DIST from each mate, weaker along the
        walking direction so groups walk abreast rather than in single file."""
        force = np.zeros(2, dtype=np.float32)
        heading = self._goal_heading()
        for mate in mates:
            diff = self.pos - mate.pos
            dist = _norm(diff)
            if dist < 1e-6:
                continue
            magnitude = self.GROUP_FORCE_STRENGTH * np.tanh((dist - self.GROUP_DESIRED_DIST) / self.GROUP_FORCE_RANGE)
            direction = -diff / dist
            if heading is not None:
                along = np.dot(direction, heading) * heading
                direction = (direction - along) + along / self.GROUP_SIDE_BIAS
                norm = _norm(direction)
                if norm < 1e-6:
                    continue
                direction = direction / norm
            force += direction * magnitude
        return force

    def _boundary_force(self):
        """Linear repulsion within WALL_MARGIN of the environment edges."""
        upper = np.array([self.width, self.height], dtype=np.float32) - self.WALL_MARGIN
        push = np.maximum(self.WALL_MARGIN - self.pos, 0.0) - np.maximum(self.pos - upper, 0.0)
        return (self.WALL_FORCE_STRENGTH * push / self.WALL_MARGIN).astype(np.float32)

    def _obstacle_force(self, obstacles):
        """Hard repulsion near an obstacle's surface, soft further out."""
        force = np.zeros(2, dtype=np.float32)
        for obs in obstacles:
            diff = self.pos - obs.pos
            dist = _norm(diff)
            surface = dist - obs.radius
            if surface < self.OBSTACLE_MARGIN_CLOSE:
                gain = (self.OBSTACLE_MARGIN_CLOSE - surface) / self.OBSTACLE_MARGIN_CLOSE
            elif surface < self.OBSTACLE_MARGIN_FAR:
                gain = 0.25 * (self.OBSTACLE_MARGIN_FAR - surface) / self.OBSTACLE_MARGIN_FAR
            else:
                continue
            force += diff / (dist + 1e-8) * self.OBSTACLE_FORCE_STRENGTH * gain
        return force

    def _lane_keeping_force(self, gap_edges):
        """In a corridor, a damped lateral spring towards the walker's right side."""
        if self.response in ("stop", "sidestep"):
            return np.zeros(2, dtype=np.float32)       # don't undo a deliberate sidestep
        floor, ceiling = gap_edges
        gap = ceiling - floor
        if gap <= 1e-6 or gap > self.CORRIDOR_WIDTH:
            return np.zeros(2, dtype=np.float32)
        standoff = self.radius + self.OBSTACLE_MARGIN_CLOSE + self.LANE_KEEP_MIN_STANDOFF_BUFFER
        offset = min(max(gap * self.LANE_KEEP_TARGET_RATIO, standoff), max(gap - standoff, gap * 0.5))
        target_y = ceiling - offset if self._corridor_right_sign() > 0 else floor + offset
        activation = _clamp(1.0 - gap / self.CORRIDOR_WIDTH, 0.0, 1.0)
        spring = self.LANE_KEEP_STRENGTH * _clamp((target_y - float(self.pos[1])) / standoff, -1.5, 1.5)
        damping = self.LANE_KEEP_DAMPING * float(self.vel[1])
        return np.array([0.0, activation * (spring - damping)], dtype=np.float32)

    def _local_gap_edges(self, obstacles):
        """(floor, ceiling): nearest obstacle surfaces below/above the agent at its x."""
        px, py = float(self.pos[0]), float(self.pos[1])
        floor, ceiling = 0.0, float(self.height)
        for obs in obstacles:
            dx = px - obs.pos[0]
            if abs(dx) >= obs.radius:
                continue
            dy = math.sqrt(max(obs.radius ** 2 - dx ** 2, 0.0))
            top, bottom = obs.pos[1] + dy, obs.pos[1] - dy
            if bottom >= py:
                ceiling = min(ceiling, bottom)
            elif top <= py:
                floor = max(floor, top)
        return floor, ceiling

    def _corridor_right_sign(self):
        """+1 if the walker's right-hand side is +y (heading west), else -1."""
        dx = float(self.goal[0] - self.pos[0])
        if abs(dx) <= 1e-3 and self.orientation is not None:
            dx = math.cos(self.orientation)
        if abs(dx) <= 1e-3:
            return self.LANE_KEEP_SIDE
        return 1.0 if dx < 0 else -1.0

    def _update_orientation(self):
        """Smoothed heading from displacement; reorients slowly when stepping backwards."""
        disp = self.pos - self.prev_pos
        if _norm(disp) < self.ORIENTATION_MIN_DISP:
            return
        target = math.atan2(disp[1], disp[0])
        if self.orientation is None:
            self.orientation = target
            return

        if abs(_wrap(target - self.orientation)) > self.BACKSTEP_ANGLE:
            self.backstep_ticks += 1
            hold = min(self.backstep_ticks / self.BACKSTEP_HOLD_TICKS, 1.0)
            smoothing = self.BACKSTEP_SMOOTHING + hold * (self.ORIENTATION_SMOOTHING - self.BACKSTEP_SMOOTHING)
        else:
            self.backstep_ticks = 0
            smoothing = self.ORIENTATION_SMOOTHING

        bx = (1.0 - smoothing) * math.cos(self.orientation) + smoothing * math.cos(target)
        by = (1.0 - smoothing) * math.sin(self.orientation) + smoothing * math.sin(target)
        new = target if math.hypot(bx, by) < 1e-8 else math.atan2(by, bx)
        self.ori_diff = _wrap(new - self.orientation)
        self.orientation = new

    def _check_arrival(self):
        if _norm(self.pos - self.goal) < self.ARRIVAL_RADIUS:
            self.arrived = True
            self.vel = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Personal space
    # ------------------------------------------------------------------

    def _update_ps_limits(self, others, gap_edges):
        """Reset temp_* limits from crowding and corridor width."""
        f, s, b = self.init_ps_f, self.init_ps_s, self.init_ps_b

        near = sorted(d for d in (_norm(o.pos - self.pos) for o in others if o is not self)
                      if d < self.CROWD_RADIUS)
        if len(near) >= 2:
            k = (2.4 - near[0] - near[1]) / 2.0
            self.temp_ps_f = _clamp(f * k, 0.3, f)
            self.temp_ps_b = _clamp(b * k, 0.15, b)
        else:
            self.temp_ps_f, self.temp_ps_b = f, b

        floor, ceiling = gap_edges
        gap = ceiling - floor
        if gap > self.CORRIDOR_WIDTH:
            self.temp_ps_left = self.temp_ps_right = s
            return
        squeeze = _clamp((gap / self.CORRIDOR_WIDTH) ** 2, self.MIN_SIDE_PS_RATIO, 1.0)
        self.temp_ps_left = self.temp_ps_right = s * squeeze
        self.temp_ps_f = min(self.temp_ps_f, f * squeeze)
        self.temp_ps_b = min(self.temp_ps_b, b * squeeze)

    def _apply_turn_limits(self, dt):
        """Shrink the limits while turning. The turn rate (rad/s) is filtered
        with a fast attack and slow release, so the PS stays small through the
        whole turn rather than for the one or two ticks of peak heading change."""
        omega = abs(self.ori_diff or 0.0) / dt
        if self._watching() and not self.TURN_SHRINK_WHILE_WATCHING:
            omega = 0.0                              # turning to face the robot: attention is on it
        x = _clamp(omega / self.TURN_RATE_REF, 0.0, 1.0)
        tau = self.TURN_ATTACK_TAU if x > self.turn_level else self.TURN_RELEASE_TAU
        self.turn_level += (x - self.turn_level) * (1.0 - math.exp(-dt / tau))

        sf, ss, sb = self.TURN_SHRINK
        kf, ks, kb = (1.0 - sf * self.turn_level, 1.0 - ss * self.turn_level, 1.0 - sb * self.turn_level)
        self.temp_ps_f = min(self.init_ps_f * kf, self.temp_ps_f)
        self.temp_ps_b = min(self.init_ps_b * kb, self.temp_ps_b)
        self.temp_ps_left = min(self.init_ps_s * ks, self.temp_ps_left)
        self.temp_ps_right = min(self.init_ps_s * ks, self.temp_ps_right)

    def _apply_speed_limits(self):
        """Front stretches and back shrinks with speed; negative emotion enlarges
        the space. Smooth towards the result. A human watching the robot keeps a
        walking-size PS even when slow or stopped, so slowing down does not
        shrink the space it is guarding."""
        ref_f, ref_s, ref_b = self.PS_SPEED_REF
        v, i = self.emotion
        k = 1.0 + 0.25 * max(-v, 0.0) * i - 0.03 * max(v, 0.0) * i
        ratio = self.actual_speed / max(self.speed, 1e-6)
        if self.aware and self.response != "walk":
            ratio = max(ratio, 1.0)
        targets = (
            min(self.init_ps_f * ref_f * max(ratio, 0.25), self.temp_ps_f),
            min(self.init_ps_b * ref_b * max(1.0 - 0.5 * ratio, 0.1), self.temp_ps_b),
            min(self.init_ps_s * ref_s, self.temp_ps_left),
            min(self.init_ps_s * ref_s, self.temp_ps_right),
        )
        a = self.PS_SMOOTHING
        self.ps_front, self.ps_back, self.ps_left, self.ps_right = (
            cur + (k * tgt - cur) * a
            for cur, tgt in zip((self.ps_front, self.ps_back, self.ps_left, self.ps_right), targets)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_local(self, point):
        """Point in the agent's body frame: (+x forward, +y left)."""
        dx, dy = point[0] - self.pos[0], point[1] - self.pos[1]
        c, s = math.cos(self.orientation), math.sin(self.orientation)
        return dx * c + dy * s, -dx * s + dy * c

    def _contact_dist(self, robot):
        return self.radius + getattr(robot, "radius", self.RADIUS)

    def _goal_heading(self):
        diff = self.goal - self.pos
        norm = _norm(diff)
        return diff / norm if norm > 1e-6 else None

    def _randint(self, bounds):
        return int(self.rng.integers(*bounds))

    def _sample_trait(self):
        return _clamp(float(self.rng.normal(0.0, 1.0)), -3.0, 3.0)

    def _sample_personal_space(self):
        """Placeholder mapping from personality to PS size. There is no agreed
        empirical relationship; plasticity is the primary driver here."""
        front = 1.2 - 0.1 * self.plasticity + 0.05 * self.stability + self.rng.uniform(-0.1, 0.1)
        side = 0.2 + 0.5 * front + self.rng.uniform(-0.1, 0.1)
        back = 0.2 + 0.5 * front + self.rng.uniform(-0.1, 0.3)
        return front, side, back


@dataclass
class _Circle:
    """Minimal obstacle stand-in (used for a parked robot)."""
    pos: np.ndarray
    radius: float


def _norm(v):
    """Length of a 2-vector (much faster than np.linalg.norm for 2 elements)."""
    return math.hypot(v[0], v[1])


def _clamp(x, lo, hi):
    """Scalar clip (much faster than np.clip for Python/NumPy scalars)."""
    return lo if x < lo else hi if x > hi else x


def _wrap(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 properly crosses segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0