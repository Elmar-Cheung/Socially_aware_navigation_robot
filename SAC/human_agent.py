import numpy as np
import random

# Emotion reaction functions (carried over from Mesa version)
# ------------------------------------------------------------------

def violate_PS(personality, ps_violate, approach):
    a = approach
    x = 10.0 / ps_violate
    match personality:
        case 'regular':
            return ((random.randint(-4, -1) / x) * a, (random.randint(1, 4) / x) * a)
        case 'grumpy':
            return ((random.randint(-7, -2) / x) * a, (random.randint(2, 7) / x) * a)
        case 'sad':
            return ((random.randint(-5, -2) / x) * a, (random.randint(2, 5) / x) * a)
        case 'chill':
            return ((random.randint(-3, -1) / x) * a, (random.randint(1, 3) / x) * a)


def collision_reaction(personality, approach, Triggered_before):
    a = approach
    if Triggered_before:
        a = a * 1.4
    match personality:
        case 'regular':
            if random.random() < 0.3:
                return True, ((random.randint(-4, -1) / 10) * a, (random.randint(1, 4) / 10) * a)
            return False, (0.0, 0.0)
        case 'grumpy':
            if random.random() < 0.7:
                return True, ((random.randint(-7, -2) / 10) * a, (random.randint(2, 7) / 10) * a)
            return False, (0.0, 0.0)
        case 'sad':
            if random.random() < 0.6:
                return True, ((random.randint(-5, -2) / 10) * a, (random.randint(2, 5) / 10) * a)
            return False, (0.0, 0.0)
        case 'chill':
            if random.random() < 0.2:
                return True, ((random.randint(-3, -1) / 10) * a, (random.randint(1, 3) / 10) * a)
            return False, (0.0, 0.0)


def disturb_group(personality, approach, Triggered_before):
    a = approach
    if Triggered_before:
        a = a * 1.4
    match personality:
        case 'regular':
            if random.random() < 0.25:
                return True, ((random.randint(-3, -2) / 10) * a, (random.randint(2, 3) / 10) * a)
            return False, (0.0, 0.0)
        case 'grumpy':
            if random.random() < 0.5:
                return True, ((random.randint(-6, -2) / 10) * a, (random.randint(2, 6) / 10) * a)
            return False, (0.0, 0.0)
        case 'sad':
            if random.random() < 0.5:
                return True, ((random.randint(-4, -2) / 10) * a, (random.randint(2, 4) / 10) * a)
            return False, (0.0, 0.0)
        case 'chill':
            if random.random() < 0.2:
                return True, ((random.randint(-2, -1) / 10) * a, (random.randint(1, 2) / 10) * a)
            return False, (0.0, 0.0)


# HumanAgent
# ------------------------------------------------------------------

class HumanAgent:
    # SFM parameters
    DESIRED_SPEED       = 1.0    # m/s base, overridden by group speed
    GOAL_FORCE_STRENGTH = 3.0
    SOCIAL_FORCE_STRENGTH = 1.0
    SOCIAL_FORCE_RANGE  = 1.5    # metres — how far social repulsion acts
    GROUP_FORCE_RANGE = 4.0
    ROBOT_FORCE_STRENGTH = 1.5
    GROUP_FORCE_STRENGTH = 2.0   # Group members want to be next to each other in a zone
    ROBOT_FORCE_RANGE   = 1.0
    MAX_SPEED           = 2.0
    ARRIVAL_RADIUS      = 1.0    # metres — how close = arrived

    PERSONALITIES = ['regular', 'grumpy', 'sad', 'chill']
    PER_PROBS     = [0.7, 0.05, 0.15, 0.1]
    CHARACTERS    = ['extrovert', 'mid', 'introvert']
    CHA_PROBS     = [0.25, 0.5, 0.25]

    def __init__(self, pos, goal, group_id, group_size, speed, width, height):
        self.prev_pos       = None
        self.pos            = np.array(pos,  dtype=np.float32)
        self.vel            = np.zeros(2,    dtype=np.float32)
        self.goal           = np.array(goal, dtype=np.float32)
        self.group_id       = group_id
        self.group_size     = group_size
        self.speed          = speed          # shared group speed
        self.width          = width
        self.height         = height
        self.emotion_timer  = None
        self.triggered_before = False

        self.personality    = np.random.choice(self.PERSONALITIES, p=self.PER_PROBS)
        self.character      = np.random.choice(self.CHARACTERS,    p=self.CHA_PROBS)
        self.behaviour_type = 'regular'
        self.emotion        = (0.0, 0.0)
        self.arrived        = False
        self.orientation    = None
        self.prev_emotion   = None
        self.prev_vel       = np.zeros(2, dtype=np.float32)
        self.ps_front, self.ps_side = self._get_personal_space_size(self.personality)

    def _get_personal_space_size(self, personality):
        match personality:
            case 'regular':
                ps_front = np.random.uniform(1.0, 1.2)
                ps_side = ps_front * 0.5
                return ps_front, ps_side
            case 'grumpy':
                ps_front = np.random.uniform(1.2, 1.7)
                ps_side = ps_front * 0.6
                return ps_front, ps_side
            case 'sad':
                ps_front = np.random.uniform(0.9, 1.1)
                ps_side = ps_front * 0.4
                return ps_front, ps_side
            case 'chill':
                ps_front = np.random.uniform(0.8, 1.3)
                ps_side = ps_front * 0.5
                return ps_front, ps_side
            

    # Main step — called every simulation tick
    # ------------------------------------------------------------------

    def step(self, dt, other_humans, robot):
        if self.arrived:
            return
        if self.prev_pos is not None:
            self.orientation = np.arctan2(self.pos[1] - self.prev_pos[1], self.pos[0] - self.prev_pos[0])

        self.prev_emotion = self.emotion
        self.prev_pos = self.pos
        self._update_behaviour(other_humans, robot)
        self._apply_sfm(dt, other_humans, robot)
        self._check_arrival()


    # Behaviour / emotion update
    # ------------------------------------------------------------------

    def _update_behaviour(self, other_humans, robot):
        approach = self._get_approach_factor(robot)
        robot_dist = np.linalg.norm(self.pos - robot.pos)
        ps_violate, trigger = self._violate_personal_space(robot, robot_dist)
        if self.emotion_timer is None:
            if robot_dist < 0.1:
                self.behaviour_type = 'collision'
                self.triggered_before, self.emotion = collision_reaction(self.personality, approach, self.triggered_before)
                self.emotion_timer = np.random.randint(50, 100)
            elif self._robot_crosses_group(robot, other_humans):
                self.behaviour_type = 'disturb_group'
                self.triggered_before, self.emotion = disturb_group(self.personality, approach, self.triggered_before)
                self.emotion_timer = np.random.randint(30, 80)
            elif trigger:
                self.behaviour_type = 'violate_ps'
                self.emotion = violate_PS(self.personality, ps_violate, approach)
                self.emotion_timer = np.random.randint(20, 60)
            else:
                self.behaviour_type = 'regular'
                self.emotion = (0.0, 0.0)
        else:
            self.emotion_timer -= 1
            if self.emotion_timer <= 0:
                self.behaviour_type = 'regular'
                self.emotion = (0.0, 0.0)
                self.emotion_timer = None
            else:
                decay_rates = {
                    'regular':   0.02,   # slow recovery
                    'grumpy':    0.05,   # medium recovery
                    'sad':       0.035,
                    'chill':     0.10,   # fast recovery
                }
                decay_v = decay_rates.get(self.personality, 0.05)
                decay_i = decay_v * 2
                current_valence = self.emotion[0]
                current_intensity = self.emotion[1]
                new_valence = current_valence + decay_v * (0.1 - current_valence)
                new_intensity = current_intensity + decay_i * (0.1 - current_intensity)
                self.emotion = (new_valence, new_intensity)


    def _violate_personal_space(self, robot, dist):
        if self.orientation is None:
            return 0.0, False

        dy_world = robot.pos[1] - self.pos[1]
        dx_world = robot.pos[0] - self.pos[0]

        local_x = dx_world * np.cos(self.orientation) + dy_world * np.sin(self.orientation)
        local_y = -dx_world * np.sin(self.orientation) + dy_world * np.cos(self.orientation)

        sigma = self.ps_front if local_x >= 0 else self.ps_side
        
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

    def _apply_sfm(self, dt, other_humans, robot):
        force = np.zeros(2, dtype=np.float32)
        force += self._goal_force()
        force += self._social_force_humans(other_humans)
        force += self._social_force_robot(robot)
        force += self._boundary_force()

        # Emotion modulates social sensitivity — upset humans move more erratically
        valence   = self.emotion[0] if self.emotion else 0.0
        intensity = self.emotion[1] if self.emotion else 0.0
        emotion_scale = 1.0 - valence * intensity  # upset = stronger reactions
        force[:2] *= emotion_scale

        self.vel  = self.vel + force * dt
        speed     = np.linalg.norm(self.vel)
        max_speed = self.MAX_SPEED * (1.0 + valence * 0.3)  # upset humans rush
        if speed > max_speed:
            self.vel = self.vel / speed * max_speed

        self.pos = self.pos + self.vel * dt
        # Hard clamp to environment bounds
        self.pos[0] = np.clip(self.pos[0], 0.1, self.width  - 0.1)
        self.pos[1] = np.clip(self.pos[1], 0.1, self.height - 0.1)

    def _goal_force(self):
        """Attract agent toward its goal at desired speed."""
        diff  = self.goal - self.pos
        dist  = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        desired_vel = (diff / dist) * self.speed
        return (desired_vel - self.vel) * self.GOAL_FORCE_STRENGTH

    def _social_force_humans(self, other_humans):
        """Repulsion from other humans."""
        force = np.zeros(2, dtype=np.float32)
        for other in other_humans:
            if other is self:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if other.group_id == self.group_id and 1.2 > dist:
                magnitude = self.GROUP_FORCE_STRENGTH * np.exp(
                    -dist / self.GROUP_FORCE_RANGE
                )
                force -= (diff / dist) * magnitude
            elif 0 < dist < self.SOCIAL_FORCE_RANGE:
                magnitude = self.SOCIAL_FORCE_STRENGTH * np.exp(
                    -dist / self.SOCIAL_FORCE_RANGE
                )
                force += (diff / dist) * magnitude
        return force

    def _social_force_robot(self, robot):
        """ Medium repulsion from robot, more patience for agent with lower intelegence """
        diff = self.pos - robot.pos
        dist = np.linalg.norm(diff)
        if 0 < dist < self.ROBOT_FORCE_RANGE:
            magnitude = self.ROBOT_FORCE_STRENGTH * np.exp(
                -dist / self.ROBOT_FORCE_RANGE
            )
            return (diff / dist) * magnitude
        return np.zeros(2, dtype=np.float32)

    def _boundary_force(self):
        """ Strong repulsion from walls, strongest of all """
        force     = np.zeros(2, dtype=np.float32)
        margin    = 0.5
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

        # 2. Calculate the vector pointing from Human to Robot
        rel_pos_vector = robot.pos - self.pos
        
        # Normalize it to get a clean unit directional vector
        rel_pos_unit = rel_pos_vector / (np.linalg.norm(rel_pos_vector) + 1e-8)
        
        # 3. Calculate the Dot Product
        # Measures how much of the human's velocity aligns with the direction of the robot
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
        Returns a 8-element list matching the observation format:
        [pos_x, pos_y, speed, valence, intensity, prev_valence, prev_intensity]
        """
        valence        = self.emotion[0]      if self.emotion      else 0.0
        intensity      = self.emotion[1]      if self.emotion      else 0.0
        prev_valence   = self.prev_emotion[0] if self.prev_emotion else 0.0
        prev_intensity = self.prev_emotion[1] if self.prev_emotion else 0.0
        orientation    = self.orientation     if self.orientation  else 0.0
        return [
            float(self.pos[0]),
            float(self.pos[1]),
            float(self.speed),
            float(valence),
            float(intensity),
            float(prev_valence),
            float(prev_intensity),
            float(orientation)
        ]
