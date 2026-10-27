import numpy as np


class Robot:
    MAX_SPEED = 0.8        # m/s
    ARRIVAL_RADIUS = 0.3   # m
    RADIUS = 0.25          # m, physical footprint
    SENSOR_RANGE = 2.5     # m, lidar/camera range
    GOAL_MARGIN = 0.4      # m, goals are kept this far from the side walls

    def __init__(self, width, height, goals=None):
        self.width, self.height = width, height
        self.radius = self.RADIUS
        m = self.GOAL_MARGIN
        """
        self.goals = [np.array(g, dtype=np.float32)
                      for g in ((m, 2.7), (width - m, 1.4), (m, 1.5), (width - m, 2.7))]
        goals = [np.array(g, dtype=np.float32)
                              for g in ((m, height - m), (width - m, m), (m, height - m), (width - m, height - m))]"""
        goals = goals
        self.goals = [np.array(g, dtype=np.float32) for g in goals]
        self.goal_index = 0
        self.reset()

    def reset(self, pos=None):
        """Place the robot (default: middle of the right edge) and take the next goal."""
        if pos is None:
            pos = (self.width - 0.2, self.height / 2.0)
        self.pos = np.array(pos, dtype=np.float32)
        self.prev_pos = self.pos.copy()     # zero velocity after a teleport
        self.vel = np.zeros(2, dtype=np.float32)
        self.time_stamp = 0
        self.assign_goal()

    def assign_goal(self):
        """Cycle through the fixed goal list."""
        self.goal = self.goals[self.goal_index].copy()
        self.goal_index = (self.goal_index + 1) % len(self.goals)

    def step(self, direction, dt):
        """Move along ``direction`` (length <= 1 is a speed fraction). Returns True
        on arrival, in which case the next goal has already been assigned."""
        self.prev_pos = self.pos.copy()
        direction = np.asarray(direction, dtype=np.float32)
        speed = float(np.linalg.norm(direction))
        if speed > 1.0:
            direction = direction / speed
        self.vel = direction * self.MAX_SPEED
        self.pos = self.pos + self.vel * dt
        self.clamp_to_bounds()
        self.time_stamp += 1
        if np.linalg.norm(self.pos - self.goal) < self.ARRIVAL_RADIUS:
            self.assign_goal()
            return True
        return False

    def clamp_to_bounds(self):
        self.pos[0] = np.clip(self.pos[0], self.radius, self.width - self.radius)
        self.pos[1] = np.clip(self.pos[1], self.radius, self.height - self.radius)

    def measured_vel(self, dt):
        """Velocity from odometry (actual displacement), not the command."""
        return (self.pos - self.prev_pos) / dt

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _in_range(self, humans):
        dists = [(float(np.linalg.norm(h.pos - self.pos)), h) for h in humans]
        return sorted((dh for dh in dists if dh[0] <= self.SENSOR_RANGE), key=lambda dh: dh[0])

    def get_observation(self, humans, top_n=1):
        """Feature vectors of the ``top_n`` nearest humans in range, zero-padded."""
        rows = [h.feature_vector() for _, h in self._in_range(humans)[:top_n]]
        rows += [[0.0] * 7] * (top_n - len(rows))
        return np.array(rows, dtype=np.float32)

    def detect(self, humans):
        """Positions of every human in sensor range, relative to the robot
        (world-aligned axes): {human_id: rel_pos}. This is what a lidar tracker
        gives, independent of how many humans the policy observes."""
        return {h.human_id: (h.pos - self.pos).astype(np.float32) for _, h in self._in_range(humans)}
