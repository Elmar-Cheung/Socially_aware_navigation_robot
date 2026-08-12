import numpy as np
import random


class Robot:
    MAX_SPEED    = 1.0   # m/s
    ARRIVAL_RADIUS = 0.1 # metres

    def __init__(self, width, height):
        self.width   = width
        self.height  = height

        # Spawn on a random edge cell (not corner)
        self.pos          = self._random_edge_pos()
        self.vel          = np.zeros(2, dtype=np.float32)
        self.prev_pos     = None
        self.goal         = None
        self.time_stamp   = 0
        self.max_speed = 1.0

        self.assign_goal()

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def assign_goal(self):
        """
        Assigns a new random goal biased toward the opposite side
        of the environment from the robot's current position.
        """
        margin = 0.2
        rx, ry = self.pos

        # Bias goal to opposite half of environment
        min_dist = min(self.width, self.height) // 2

        attempts = 0
        while attempts < 100:
            gx = random.uniform(margin, self.width  - margin)
            gy = random.uniform(margin, self.height - margin)
            dist = np.sqrt((rx - gx)**2 + (ry - gy)**2)
            if dist > min_dist:
                self.goal = np.array([gx, gy], dtype=np.float32)
                return
            attempts += 1

        # Fallback: any free position
        self.goal = np.array([
            random.uniform(margin, self.width  - margin),
            random.uniform(margin, self.height - margin)
        ], dtype=np.float32)


    # Movement
    # ------------------------------------------------------------------

    def step(self, action, dt):
        self.prev_pos = self.pos.copy()

        speed = np.linalg.norm(action)
        if speed > 1e-6:
            direction = action / speed
        else:
            direction = np.zeros(2, dtype=np.float32)

        self.vel = direction * min(speed, 1.0) * self.MAX_SPEED
        new_pos = self.pos + self.vel * dt

        # Clamp strictly to environment bounds
        new_pos[0] = np.clip(new_pos[0], 0.0, self.width)
        new_pos[1] = np.clip(new_pos[1], 0.0, self.height)
        self.pos = new_pos

        self.time_stamp += 1

        if self.goal is not None:
            if np.linalg.norm(self.pos - self.goal) < self.ARRIVAL_RADIUS:
                self.assign_goal()
                return True

        return False


    # Observation
    # ------------------------------------------------------------------

    def get_observation(self, humans, radius=2.5, top_n=2):
        nearby = [
            h for h in humans
            if np.linalg.norm(h.pos - self.pos) <= radius
        ]
        nearby.sort(key=lambda h: np.linalg.norm(h.pos - self.pos))
        nearby = nearby[:top_n]

        data = []
        for h in nearby:
            data.append(h.feature_vector())

        # Pad to top_n rows
        while len(data) < top_n:
            data.append([0.0] * 7)

        return np.array(data, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_edge_pos(self):
        """Spawns robot on a random edge position, avoiding corners."""
        margin   = 1.0
        corner_margin = 2.0  # stay away from corners

        edges = {
            'top':    (
                random.uniform(corner_margin, self.width - corner_margin),
                self.height - margin
            ),
            'bottom': (
                random.uniform(corner_margin, self.width - corner_margin),
                margin
            ),
            'left':   (
                margin,
                random.uniform(corner_margin, self.height - corner_margin)
            ),
            'right':  (
                self.width - margin,
                random.uniform(corner_margin, self.height - corner_margin)
            )
        }
        x, y = random.choice(list(edges.values()))
        return np.array([x, y], dtype=np.float32)

    @property
    def goal_pos(self):
        """Convenience alias matching old Mesa Robot API."""
        return tuple(self.goal) if self.goal is not None else None
