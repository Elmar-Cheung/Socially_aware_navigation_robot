from math import dist

import numpy as np
import random

# Emotion reaction functions (carried over from Mesa version)
# ------------------------------------------------------------------
def norm_personality(val):
    return (val + 3.0) / 6.0

def violate_PS(stability, ps_violate, approach):
    # Depend on how close, but does not exceed collision reaction
    a = approach
    x = ps_violate / (ps_violate + 5.0)
    stability = norm_personality(stability)
    
    trigger_prob = np.clip(0.5 - 0.2 * stability + 0.3 * x, 0.10, 0.90)
    if random.random() < trigger_prob:
        mag_scale = np.clip(1.0 - 0.3 * stability + 0.4 * x, 0.1, 1.5)
        valence = random.uniform(-0.45, -0.30) * mag_scale * a
        intensity = random.uniform(0.45, 0.55) * mag_scale * a
        return True, (np.clip(valence, -1.0, 1.0), np.clip(intensity, 0.0, 1.0))
    return False, (0.0, 0.0)

def collision_reaction(stability, approach, Triggered_before):
    # high probability and high intensity
    stability = norm_personality(stability)
    a = approach
    if Triggered_before:
        a = a * 1.4

    trigger_prob = np.clip(0.85 - 0.25 * stability, 0.40, 0.98)
    if random.random() < trigger_prob:
        mag_scale = np.clip(1.2 - 0.40 * stability, 0.70, 1.80)
        valence = random.uniform(-0.65, -0.55) * mag_scale * a
        intensity = random.uniform(0.65, 0.75) * mag_scale * a
        return True, (np.clip(valence, -1.0, -0.01), np.clip(intensity, 0.0, 1.0))
    return False, (0.0, 0.0)
    
def disturb_group(stability, approach, Triggered_before):
    # low probability and low intensity
    stability = norm_personality(stability)
    a = approach
    if Triggered_before:
        a = a * 1.2

    trigger_prob = np.clip(0.35 - 0.15 * stability, 0.05, 0.65)
    if random.random() < trigger_prob:
        mag_scale = np.clip(1.0 - 0.30 * stability, 0.5, 1.4)
        valence = random.uniform(-0.30, -0.15) * mag_scale * a
        intensity = random.uniform(0.25, 0.40) * mag_scale * a
        return True, (np.clip(valence, -1.0, 1.0), np.clip(intensity, 0.0, 1.0))
    return False, (0.0, 0.0)

def curious_reaction(stability, plasticity):
    """First-encounter curiosity trigger."""
    stability = norm_personality(stability)
    plasticity = norm_personality(plasticity)

    base_prob = np.clip(0.3 + 0.5 * plasticity, 0.10, 0.90)
    if random.random() > base_prob:
        return False, (0.0, 0.0)
    
    mag_scale = np.clip(1.0 - 0.30 * stability + 0.2 * plasticity, 0.40, 1.6)
    valence = random.uniform(0.25, 0.40) * mag_scale
    intensity = random.uniform(0.20, 0.50) * mag_scale
    return True, (np.clip(valence, -1.0, 1.0), np.clip(intensity, 0.0, 1.0))

def startle_trigger(stability, plasticity, approach, dist_severity):
    stability = norm_personality(stability)
    plasticity = norm_personality(plasticity)
    a = approach

    # Determine if the reaction is freeze
    pause_prob = np.clip(0.6 - 0.4 * plasticity - 0.2 * stability, 0.05, 0.95)
    is_pause = random.random() < pause_prob
    mag_base = 1.0 - 0.30 * stability + 0.4 * dist_severity
    if is_pause:
        mag_scale = np.clip(mag_base, 0.50, 1.8)
        valence = random.uniform(-0.65, -0.55) * mag_scale * a
        intensity = random.uniform(0.55, 0.65) * mag_scale * a
        return 'freeze', (np.clip(valence, -1.0, 1.0), np.clip(intensity, 0.0, 1.0))
    
    jolt_mag_scale = np.clip(mag_base + 0.2, 0.6, 2.2)

    valence = random.uniform(-0.60, -0.50) * jolt_mag_scale * a
    intensity = random.uniform(0.65, 0.75) * jolt_mag_scale * a
    return 'jolt', (np.clip(valence, -1.0, 1.0), np.clip(intensity, 0.0, 1.0))


# HumanAgent
# ------------------------------------------------------------------

class HumanAgent:
    # SFM parameters
    DESIRED_SPEED       = 1.0    # m/s base, overridden by group speed
    GOAL_FORCE_STRENGTH = 3.0
    SOCIAL_FORCE_STRENGTH = 1.0
    SOCIAL_FORCE_RANGE  = 1.5    # metres — how far social repulsion acts
    GROUP_FORCE_RANGE = 4.0
    CURIOSITY_RANGE = 4.0
    ROBOT_FORCE_STRENGTH = 0.8
    GROUP_FORCE_STRENGTH = 2.0   # Group members want to be next to each other in a zone
    ROBOT_FORCE_RANGE   = 1.0
    MAX_SPEED           = 2.0
    ARRIVAL_RADIUS      = 0.05    # metres — how close = arrived

    # Startle response
    STARTLE_DIST_THRESHOLD  = 1.3    # metres — robot inside this range can startle
    STARTLE_SPEED_THRESHOLD = 0.6    # m/s — robot must be moving at least this fast
    FREEZE_DURATION_RANGE = (8, 20)
    JOLT_DURATION_RANGE   = (3, 8)

    def __init__(self, pos, goal, group_id, human_id, group_size, speed, width, height):
        self.prev_pos       = None
        self.pos            = np.array(pos,  dtype=np.float32)
        self.vel            = np.zeros(2,    dtype=np.float32)
        self.goal           = np.array(goal, dtype=np.float32)
        self.group_id       = group_id
        self.human_id       = human_id
        self.group_size     = group_size
        self.base_speed     = speed          # shared group speed
        self.speed          = speed          # individual speed, updated to match group
        self.width          = width
        self.height         = height

        # Timers
        self.emotion_timer  = None
        self.event_timer    = None
        self.pause_timer    = None

        self.pause_mode      = None
        self.pause_speed_mult = 1.0
        self.pause_timer     = 0
        self.pause_cooldown  = np.random.randint(50, 150)
        self.triggered_before = False

        self.stability      = np.clip(np.random.normal(0.0, 1.0), -3, 3)
        self.plasticity     = np.clip(np.random.normal(0.0, 1.0), -3, 3)

        self.behaviour_type = 'regular'
        self.emotion        = (0.0, 0.0)
        self.arrived        = False
        self.orientation    = None
        self.prev_emotion   = None
        self.met_robot = False   # Check if met robot before
        self.prev_vel       = np.zeros(2, dtype=np.float32)
        self.startle_reaction  = None   # 'freeze' | 'jolt' | None
        self.startle_phys_timer = 0     # ticks left of physical override
        self.init_ps_f, self.init_ps_s, self.init_ps_b = self._get_personal_space_size()
        self.ps_front, self.ps_side, self.ps_back = self.init_ps_f, self.init_ps_s, self.init_ps_b
        self.temp_ps_f, self.temp_ps_s, self.temp_ps_b = self.ps_front, self.ps_side, self.ps_back
        # track speed change
        self.actual_speed = 0.0

    def _get_personal_space_size(self):
        """
        There is no widely agreed relationship between plasticity/stability with
        personal space size. This is just for the convenience of the simulation.
        Plasticity - primary driver for personal space size, stability - secondary driver.
        """
        ps_front = 1.0 - 0.1 * self.plasticity + 0.05 * self.stability + np.random.uniform(-0.1, 0.1)
        ps_side = ps_front * 0.5 + np.random.uniform(-0.05, 0.05)
        ps_back = ps_front * 0.5 + np.random.uniform(-0.1, 0.5)
        return ps_front, ps_side, ps_back


    # Main step — called every simulation tick
    # ------------------------------------------------------------------

    def step(self, dt, other_humans, robot):
        if self.arrived:
            return
        if self.prev_pos is not None and self.startle_reaction != 'freeze':
            self.orientation = np.arctan2(self.pos[1] - self.prev_pos[1], self.pos[0] - self.prev_pos[0])

        self.prev_pos = self.pos
        self._update_behaviour(dt, other_humans, robot)
        self._apply_sfm(dt, other_humans, robot)
        self._speed_scaled_ps()
        self._check_arrival()

    # Impact of speed on the size of personal space
    def _speed_scaled_ps(self):
        # Gérin-Lajoie reference values at normal walking speed (~1.0-1.3 m/s)
        front_ref = 2.0
        side_ref  = 0.6
        
        speed_ratio = self.actual_speed / self.speed  # e.g. reference_speed = 1.2

        self.ps_front = min(self.init_ps_f * front_ref * max(speed_ratio, 0.25), self.temp_ps_f)  # front stretches/shrinks with speed
        self.ps_side  = min(self.init_ps_s * side_ref, self.temp_ps_s)   # side stays close to reference regardless of speed
        self.ps_back  = min(self.init_ps_b * side_ref * max(1.0 - 0.5*speed_ratio, 0.1), self.temp_ps_b) 


    # Behaviour / emotion update
    # ------------------------------------------------------------------

    def _check_collision(self, dt, other_humans, robot, robot_dist, trigger, approach, ps_violate):
        if robot_dist >= 0.1:
            return False
        else:
            self.behaviour_type = 'collision'
            self.triggered_before, self.emotion = collision_reaction(self.stability, approach, self.triggered_before)
            self.emotion_timer = np.random.randint(50, 100)
            return True

    def _check_startle(self, dt, other_humans, robot, robot_dist, trigger, approach, ps_violate):
        dist_severity = np.clip(1.0 - robot_dist / self.STARTLE_DIST_THRESHOLD, 0.0, 1.0)
        if self._detect_startle(dt, robot, dist_severity):
            self.behaviour_type = 'startled'
            self.startle_reaction, self.emotion = startle_trigger(self.stability, self.plasticity, approach, dist_severity)
            self.emotion_timer = np.random.randint(40, 90)
            duration_scale = np.clip(1.0 - 0.2 * self.stability, 0.4, 2.0)
            self.triggered_before = True
            if self.startle_reaction == 'freeze':
                self.startle_phys_timer = np.random.randint(*self.FREEZE_DURATION_RANGE) * duration_scale
            else:
                self.startle_phys_timer = np.random.randint(*self.JOLT_DURATION_RANGE) * duration_scale
                away = self.pos - robot.pos
                away_dist = np.linalg.norm(away) + 1e-8
                mag_scale = np.clip(1.0 + 0.1 * self.plasticity, 0.7, 1.3)
                jolt_strength = np.clip(1.0 + 0.1 * self.plasticity, 0.7, 1.3) * mag_scale * (0.6 + self.emotion[1])
                self.vel = (away / away_dist) * jolt_strength
            return True
        else:
            return False

    def _check_disturb_group(self, dt, other_humans, robot, robot_dist, trigger, approach, ps_violate):
        if self._robot_crosses_group(robot, other_humans):
            self.behaviour_type = 'disturb_group'
            self.triggered_before, self.emotion = disturb_group(self.stability, approach, self.triggered_before)
            self.emotion_timer = np.random.randint(30, 80)
            return True
        else:
            return False

    def _check_curiosity(self, robot, robot_dist):
            if not self.met_robot and robot_dist < self.CURIOSITY_RANGE:
                self.met_robot = True
                triggered, self.emotion = curious_reaction(self.stability, self.plasticity)
                if triggered:
                    self.behaviour_type = 'curious'
                    self.emotion_timer = np.random.randint(40, 100)
                else:
                    self.behaviour_type = 'regular'

    def _check_violate_ps(self, dt, other_humans, robot, robot_dist, trigger, approach, ps_violate):
        if trigger:
            self.behaviour_type = 'violate_ps'
            self.triggered_before, self.emotion = violate_PS(self.stability, ps_violate, approach)
            self.emotion_timer = np.random.randint(20, 60)
            return True
        else:
            return False

    # Emotion decay
    def _decay_emotion(self):
        self.emotion_timer -= 1
        if self.emotion_timer <= 0:
            self.behaviour_type = 'regular'
            self.emotion = (0.0, 0.0)
            self.emotion_timer = None
        else:
            decay_rate = np.clip(0.05 + 0.015 * self.stability, 0.01, 0.1)
            decay_i = decay_rate * 2
            current_valence = self.emotion[0]
            current_intensity = self.emotion[1]
            new_valence = current_valence + decay_rate * (0.1 - current_valence)
            new_intensity = current_intensity + decay_i * (0.1 - current_intensity)
            self.emotion = (new_valence, new_intensity)
        return

    def _update_pause_state(self, dt):
        if self.pause_timer > 0:
            # currently mid-pause — count it down, then release
            self.pause_timer -= 1
            if self.pause_timer <= 0:
                self.pause_mode = 'accelerate'
            return

        if self.startle_phys_timer > 0:
            return  # don't start a new pause mid-startle-response

        if self.pause_mode == 'decelerate':
            # deceleration phase — reduce speed gradually, then enter full pause
            self.pause_speed_mult -= 0.05
            if self.pause_speed_mult <= 0.0:
                self.pause_speed_mult = 0.0
                self.pause_timer = np.random.randint(10, 60)  # full pause duration
            return

        if self.pause_mode == 'accelerate':
            # acceleration phase — increase speed gradually, then return to normal
            self.pause_speed_mult += 0.05
            if self.pause_speed_mult >= 1.0:
                self.pause_speed_mult = 1.0
                self.pause_mode = None
                self.pause_cooldown = np.random.randint(50, 150)  # next pause interval
            return
        
        self.pause_cooldown -= 1
        if self.pause_cooldown <= 0:
            prob = 0.15 - 0.08 * self.plasticity - 0.04 * self.stability
            pause_triggered = random.random() < min(prob, 0.95)
            if pause_triggered:
                self.pause_mode = 'decelerate'

    REACTIVE_TRIGGERS = [
        ('collision',      _check_collision),
        ('startled',       _check_startle),
        ('disturb_group',  _check_disturb_group),
        ('violate_ps',     _check_violate_ps),
    ]

    def _update_reactive_state(self, dt, other_humans, robot):
        if self.emotion_timer is not None:
            self._decay_emotion()
            return

        approach = self._get_approach_factor(robot)
        robot_dist = np.linalg.norm(self.pos - robot.pos)
        ps_violate, trigger = self._violate_personal_space(robot, robot_dist)

        for name, check_fn in self.REACTIVE_TRIGGERS:
            if check_fn(self, dt, other_humans, robot, robot_dist, trigger, approach, ps_violate):
                return
        self._check_curiosity(robot, robot_dist)
        
    def _update_behaviour(self, dt, other_humans, robot):
        self._update_pause_state(dt)          # dispositional layer, independent
        self._update_reactive_state(dt, other_humans, robot)  # trigger table, priority-ordered

    def _violate_personal_space(self, robot, dist):
        if self.orientation is None:
            return 0.0, False

        dy_world = robot.pos[1] - self.pos[1]
        dx_world = robot.pos[0] - self.pos[0]

        local_x = dx_world * np.cos(self.orientation) + dy_world * np.sin(self.orientation)
        local_y = -dx_world * np.sin(self.orientation) + dy_world * np.cos(self.orientation)

        sigma = self.ps_front if local_x >= 0 else self.ps_back
        
        term_front = (local_x ** 2) / (2 * sigma ** 2)
        term_side = (local_y ** 2) / (2 * self.ps_side ** 2)
        
        elliptical_dist = 2.0 * (term_front + term_side)
        # Compute final reward penalty
        if elliptical_dist >= 1.0:
            ps_violate = 0.0
            trigger = False
        else:
            raw_curve = np.exp(1.0 - elliptical_dist) - 1.0
            max_scale = np.exp(1.0) - 1.0

            ps_violate = raw_curve / max_scale
            trigger = np.random.random() < ps_violate
        
        if self.triggered_before:
            ps_violate = ps_violate * 1.2

        if trigger:
            self.triggered_before = True
        
        return ps_violate, trigger
        

    def _detect_startle(self, dt, robot, dist_severity):
        if robot.prev_pos is None or self.orientation is None:
            return False

        dist = np.linalg.norm(self.pos - robot.pos)
        if dist > self.STARTLE_DIST_THRESHOLD:
            return False

        robot_disp = robot.pos - robot.prev_pos
        robot_speed = np.linalg.norm(robot_disp) / dt
        if robot_speed < self.STARTLE_SPEED_THRESHOLD:
            return False

        to_human = self.pos - robot.pos
        to_human_dist = np.linalg.norm(to_human) + 1e-8
        closing_speed = np.dot(robot_disp / dt, to_human / to_human_dist)
        if closing_speed <= 0:
            return False  # robot moving away — not startling

        dx_world = robot.pos[0] - self.pos[0]
        dy_world = robot.pos[1] - self.pos[1]
        local_x = dx_world * np.cos(self.orientation) + dy_world * np.sin(self.orientation)
        startle_scale = np.clip(1.0 + 0.1 * self.stability, 0.7, 1.3)
        behind_mult = startle_scale if local_x < 0 else 1.0

        prob = np.clip(0.5 - 0.15 * self.stability + 0.2 * dist_severity * behind_mult, 0.05, 0.95)
        return random.random() < prob

    def _robot_crosses_group(self, robot, other_humans):
        """
        Checks if the robot's recent trajectory crosses between
        any two members of this group — continuous version of
        the Mesa intersection check.
        """
        if robot.prev_pos is None:
            return False

        group_mates = [
            h for h in other_humans
            if h.group_id == self.group_id and h is not self
        ]
        for mate in group_mates:
            if self._segments_intersect(
                robot.prev_pos, robot.pos, self.pos, mate.pos
            ):
                return True
        return False

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        """Returns True if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False


    # Social Force Model
    # ------------------------------------------------------------------

    """
    Includes:
        goal attraction
        curiosity attraction
        social repulsion (humans)
        social repulsion (robot)
        group cohesion
        boundary repulsion

    """
    def _apply_sfm(self, dt, other_humans, robot):
        if self.startle_phys_timer > 0:
            self._apply_startle_physical(dt, robot)
            self.startle_phys_timer -= 1
            if self.startle_phys_timer <= 0:
                self.startle_reaction = None
            return

        self._sync_group_speed(other_humans)

        force = np.zeros(2, dtype=np.float32)
        force += self._goal_force()
        force += self._curiosity_force(robot)
        force += self._social_force_humans(other_humans)
        force += self._social_force_robot(robot)
        force += self._group_force(other_humans)
        force += self._boundary_force()

        # Emotion modulates social sensitivity — upset humans move more erratically
        valence   = self.emotion[0] if self.emotion else 0.0
        intensity = self.emotion[1] if self.emotion else 0.0
        emotion_scale = 1.0 - valence * intensity  # upset = stronger reactions
        force[:2] *= emotion_scale

        self.vel  = self.vel + force * dt
        speed     = np.linalg.norm(self.vel)
        max_speed = self.MAX_SPEED * (1.0 + abs(valence) * 0.3)  # upset humans rush
        if speed > max_speed:
            self.vel = self.vel / speed * max_speed
        speed     = np.linalg.norm(self.vel)
        
        self.actual_speed = speed
        self.pos = self.pos + self.vel * dt
        # Hard clamp to environment bounds
        self.pos[0] = np.clip(self.pos[0], 0.1, self.width  - 0.1)
        self.pos[1] = np.clip(self.pos[1], 0.1, self.height - 0.1)

    def _apply_startle_physical(self, dt, robot):
        """Physical half of the startle response, run instead of normal SFM
        while startle_phys_timer is active."""
        if self.startle_reaction == 'freeze':
            diff = robot.pos - self.pos
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                self.orientation = float(np.arctan2(diff[1], diff[0]))
            self.vel = np.zeros(2, dtype=np.float32)
            # position intentionally unchanged

        elif self.startle_reaction == 'jolt':
            force = self._boundary_force() + self._social_force_robot(robot)
            self.vel = self.vel * 0.9 + force * dt
            self.pos = self.pos + self.vel * dt
            self.pos[0] = np.clip(self.pos[0], 0.1, self.width  - 0.1)
            self.pos[1] = np.clip(self.pos[1], 0.1, self.height - 0.1)

    def _curiosity_force(self, robot):
        """Mild attraction toward robot while curious, capped so it never overrides
        personal-space repulsion at close range."""
        if self.behaviour_type != 'curious':
            return np.zeros(2, dtype=np.float32)
        diff = robot.pos - self.pos
        dist = np.linalg.norm(diff)
        if dist < self.ps_front: 
            return np.zeros(2, dtype=np.float32)
        valence, intensity = self.emotion
        magnitude = 0.8 * valence * intensity  # small relative to goal/repulsion forces
        return (diff / dist) * magnitude

    def _goal_force(self):
        diff = self.goal - self.pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        direction = diff / dist

        wander_mult = np.clip(1.0 + 0.2 * self.plasticity, 0.7, 1.3)
        wander_scale = np.clip(0.06 + 0.01 * self.stability, 0.03, 0.09)
        wander = wander_scale * wander_mult
        noise_angle = np.random.normal(0, wander)
        c, s = np.cos(noise_angle), np.sin(noise_angle)
        direction = np.array([
            direction[0] * c - direction[1] * s,
            direction[0] * s + direction[1] * c
        ], dtype=np.float32)

        speed_jitter = np.clip(0.03 + 0.01 * self.stability, 0.02, 0.05)
        speed_noise = np.random.normal(0, speed_jitter)
        effective_speed = max(0.05, self.base_speed + speed_noise) * self.pause_speed_mult

        desired_vel = direction * effective_speed
        return (desired_vel - self.vel) * self.GOAL_FORCE_STRENGTH


    GROUP_DESIRED_DIST = 2.0   # comfortable side-by-side spacing
    GROUP_SIDE_BIAS = 1.6 

    def _group_cohesion_force(self, other, dist, diff):
        delta = dist - self.GROUP_DESIRED_DIST
        magnitude = self.GROUP_FORCE_STRENGTH * np.tanh(delta / self.GROUP_FORCE_RANGE)

        base_dir = -diff / (dist * 0.8)   # attract when far, repel when close (sign fixed from before)

        heading = self.goal - self.pos
        hnorm = np.linalg.norm(heading)
        if hnorm < 1e-6:
            return base_dir * magnitude

        heading_unit = heading / hnorm
        
        along = np.dot(base_dir, heading_unit) * heading_unit
        lateral = base_dir - along

        shaped_dir = lateral + along / self.GROUP_SIDE_BIAS
        shaped_dir = shaped_dir / (np.linalg.norm(shaped_dir) + 1e-8)

        return shaped_dir * magnitude

    def _group_force(self, other_humans):
        force = np.zeros(2, dtype=np.float32)
        for other in other_humans:
            if other is self or other.group_id != self.group_id:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            force += self._group_cohesion_force(other, dist, diff)
        return force
    
    def _sync_group_speed(self, other_humans):
        mates = [h for h in other_humans if h.group_id == self.group_id]
        if not mates:
            self.speed += (self.base_speed - self.speed) * 0.1
            return

        # a paused/startled mate has effective speed ~0 — the group should slow for them,
        # not ignore them and speed back up
        effective_speeds = [
            0.0 if (h.pause_timer > 0 or h.startle_phys_timer > 0) else h.actual_speed
            for h in mates
        ]
        slowest = min(effective_speeds)
        target = slowest if slowest > 0.05 else self.base_speed * 0.05  # near-stop, not full base_speed

        self.speed += (target - self.speed) * 0.1

    def _social_force_humans(self, other_humans):
        """Repulsion from other humans."""
        force = np.zeros(2, dtype=np.float32)
        count = 0
        close_dist1 = 0.0
        close_dist2 = 0.0

        for other in other_humans:
            if other is self:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < 1.0:
                count += 1
                if close_dist1 == 0.0 or dist < close_dist1:
                    close_dist2 = close_dist1
                    close_dist1 = dist
                elif close_dist2 == 0.0 or dist < close_dist2:
                    close_dist2 = dist
            if other.group_id == self.group_id:
                continue  # handled entirely by _group_force now
            if 0 < dist < self.SOCIAL_FORCE_RANGE:
                magnitude = self.SOCIAL_FORCE_STRENGTH * np.exp(-dist / self.SOCIAL_FORCE_RANGE)
                force += (diff / dist) * magnitude
        avg_intimate_dist = (2.4 - close_dist1 - close_dist2) / 2.0
        if count >= 2:
            self.temp_ps_f = np.clip(self.init_ps_f * avg_intimate_dist, 0.3, self.init_ps_f)
            self.temp_ps_s  = np.clip(self.init_ps_s * avg_intimate_dist, 0.15, self.init_ps_s)
            self.temp_ps_b  = np.clip(self.init_ps_b * avg_intimate_dist, 0.15, self.init_ps_b)
        else:
            self.temp_ps_f = self.init_ps_f
            self.temp_ps_s  = self.init_ps_s
            self.temp_ps_b  = self.init_ps_b
        return force

    def _social_force_robot(self, robot):
        """ 
        Medium repulsion from robot, more patience for agent with lower intelegence 
        """
        diff = self.pos - robot.pos
        dist = np.linalg.norm(diff)
        if 0 < dist < self.ROBOT_FORCE_RANGE:
            magnitude = self.ROBOT_FORCE_STRENGTH * np.exp(
                -dist / self.ROBOT_FORCE_RANGE
            )
            return (diff / dist) * magnitude
        return np.zeros(2, dtype=np.float32)

    def _boundary_force(self):
        """ 
        Strong repulsion from walls, strongest of all 
        """
        force     = np.zeros(2, dtype=np.float32)
        margin    = 0.2
        strength  = 10.0

        if self.pos[0] < margin:
            force[0] += strength * (margin - self.pos[0]) / margin
        if self.pos[0] > self.width - margin:
            force[0] -= strength * (self.pos[0] - (self.width - margin)) / margin
        if self.pos[1] < margin:
            force[1] += strength * (margin - self.pos[1]) / margin
        if self.pos[1] > self.height - margin:
            force[1] -= strength * (self.pos[1] - (self.height - margin)) / margin

        return force


    # Check if it is Human approaching
    # ------------------------------------------------------------------

    def _get_approach_factor(self, robot, speed_threshold=0.1):
        # 1. Check if the human is moving at all
        speed = np.linalg.norm(self.vel)
        if speed < speed_threshold:
            return False  # Human is stationary; any collision is the robot's fault

        rel_pos_vector = robot.pos - self.pos
        
        # Normalize it to get a clean unit directional vector
        rel_pos_unit = rel_pos_vector / (np.linalg.norm(rel_pos_vector) + 1e-8)

        approach_alignment = np.dot(self.vel, rel_pos_unit)
        if approach_alignment > 0.0:
            # Human is cutting down the distance
            return 0.2
        else:
            return 1.0

    # Arrival
    # ------------------------------------------------------------------

    def _check_arrival(self):
        if np.linalg.norm(self.pos - self.goal) < self.ARRIVAL_RADIUS:
            self.arrived = True
            self.vel     = np.zeros(2, dtype=np.float32)


    # Feature vector for RL observation
    # ------------------------------------------------------------------

    def feature_vector(self):
        """
        Returns a 6-element list matching the observation format:
        [pos_x, pos_y, speed, valence, intensity, prev_valence, prev_intensity]
        """
        valence     = self.emotion[0]      if self.emotion      else 0.0
        intensity   = self.emotion[1]      if self.emotion      else 0.0
        orientation = self.orientation     if self.orientation  else 0.0
        speed       = self.actual_speed    if self.actual_speed else 0.0
        human_id    = self.human_id        if self.human_id     else 0.0
        return [
            float(self.pos[0]),
            float(self.pos[1]),
            float(valence),
            float(intensity),
            float(orientation),
            float(speed),
            float(human_id)
        ]