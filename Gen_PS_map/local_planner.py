"""Fast-marching local planner on a robot-centred costmap.

Costs come from the predicted personal-space ellipses of observed humans (the
policy's action), obstacles and walls. The planner follows the travel-time
gradient, blends in escape/goal forces, and falls back to a local grid search
whenever the chosen step would violate a safety margin.

All grid coordinates are relative to the robot (world-aligned axes).
"""
from collections import deque

import numpy as np
import skfmm


def _unit(v):
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else None


def _segment_circle_entry(p0, p1, center, r):
    """Fraction t in [0, 1] where segment p0->p1 first enters the circle, or None."""
    p0, p1, center = (np.asarray(a, dtype=np.float64) for a in (p0, p1, center))
    d, f = p1 - p0, p0 - center
    a, c = float(d @ d), float(f @ f) - r * r
    if c < 0:
        return 0.0                                   # p0 already inside
    if a < 1e-12:
        return None
    b = 2.0 * float(f @ d)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
        if 0.0 <= t <= 1.0:
            return float(t)
    return None


class LocalPlanner:
    MAP_SIZE = 5.0            # m, side of the local costmap
    RESOLUTION = 0.025        # m per cell
    SAFETY_MARGIN = 0.2       # m kept beyond body contact with humans and obstacles
    WALL_SAFE = 0.1           # m, robot centre must stay this far inside the walls
    ESCAPE_WALL_MARGIN = 0.4  # m, wall escape force acts within this distance
    MIN_PS = 0.05             # m, floor on predicted PS sizes (avoids division by zero)
    PREDICT_TICKS = 3         # human position is predicted this many steps ahead
    GOAL_FORCE = 3.0          # pull towards the goal when no human is within 2 m
    STUCK_WINDOW = 1.0        # s, window for the "net progress" check
    STUCK_DIST = 0.10         # m of net progress within the window
    SOFT_EDGE_COST = 0.25       # cost at the edge of a predicted PS ellipse
    SOFT_EDGE_WIDTH = 0.15      # width of the soft edge
    def __init__(self, width, height, obstacles, dt, robot_radius, robot_speed, human_radius):
        self.width, self.height, self.obstacles, self.dt = width, height, obstacles, dt
        self.robot_radius, self.robot_speed, self.human_radius = robot_radius, robot_speed, human_radius
        n = int(self.MAP_SIZE / self.RESOLUTION)
        self.n, self.center = n, n // 2
        offsets = (np.arange(n) - self.center) * self.RESOLUTION
        self.w_x, self.w_y = np.meshgrid(offsets, offsets, indexing='xy')
        self._pos_history = deque(maxlen=max(2, int(round(self.STUCK_WINDOW / dt))))
        self.escape_ticks = 0
        self.reset()

    def reset(self):
        self.escape_ticks = 0
        self.prev_movement = None
        self.stuck_ticks = 0
        self._pos_history.clear()
        self.safety_triggered = False
        self.costmap = None         # last costmap, for rendering
        self.resolved_goal = None   # local goal after obstacle resolution (relative)
        self.goal_idx = None        # (x, y) grid index of the local goal
        self.debug = {}
        self.last_unsafe = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_position(self, pos):
        """Track net progress to detect being stuck (widens the fallback search)."""
        self._pos_history.append(np.array(pos, dtype=np.float32))
        if len(self._pos_history) == self._pos_history.maxlen:
            moved = np.linalg.norm(self._pos_history[-1] - self._pos_history[0])
            self.stuck_ticks = self.stuck_ticks + 1 if moved < self.STUCK_DIST else 0

    def plan(self, robot_pos, goal, humans, ps_sizes, nearby):
        """Unit direction for the next step.

        humans:   observed rows [rel_x, rel_y, rel_orientation, speed, ...]
        ps_sizes: matching predicted PS rows [front, left, right, back] (policy action)
        nearby:   relative positions of every detected human (clearance escape)
        """
        self.safety_triggered = False
        self._pos = np.asarray(robot_pos, dtype=np.float32)
        self._nearby = list(nearby)
        half = self.MAP_SIZE / 2
        rel_goal = np.clip(np.asarray(goal, dtype=np.float32) - self._pos, -half, half)
        gx, gy = self.resolved_goal = self._resolve_goal(rel_goal)

        x_next, y_next, human_cost = self._human_costmap(humans, ps_sizes)
        nearest_human = float(np.min(np.hypot(humans[:, 0], humans[:, 1]))) if len(humans) else np.inf
        self._x_next, self._y_next = x_next, y_next

        self._x_now = humans[:, 0] if len(humans) else np.empty(0)
        self._y_now = humans[:, 1] if len(humans) else np.empty(0)

        blocked = self._blocked_mask()
        goal_ix = int(np.clip(int(gx / self.RESOLUTION) + self.center, 0, self.n - 1))
        goal_iy = int(np.clip(int(gy / self.RESOLUTION) + self.center, 0, self.n - 1))
        if blocked[goal_iy, goal_ix]:                     # move the goal to the nearest free cell
            free_y, free_x = np.nonzero(~blocked)
            if free_y.size:
                k = np.argmin((free_y - goal_iy) ** 2 + (free_x - goal_ix) ** 2)
                goal_iy, goal_ix = int(free_y[k]), int(free_x[k])
        self.goal_idx = (goal_ix, goal_iy)

        cost = human_cost + self._obstacle_costmap() + self._wall_costmap(margin=self.ESCAPE_WALL_MARGIN)
        cost[np.hypot(self.w_x - gx, self.w_y - gy) < self.robot_radius + 0.15] = 0.0
        self.costmap = cost

        # 1. Already inside a safety margin: flee first.
        if not self._is_safe(0.0, 0.0):
            escape = self._clearance_escape()
            if escape is not None:
                step = escape * self.robot_speed * self.dt
                if self._clearance(*step) > self._clearance(0.0, 0.0):
                    self.safety_triggered = True
                    self.debug = dict(mode='early_escape', movement=escape)
                    self.prev_movement = escape
                    return escape
        if self.escape_ticks > 0 and self.prev_movement is not None \
        and self._is_safe(*(self.prev_movement * self.robot_speed * self.dt)):
            self.escape_ticks -= 1
            self.debug = dict(mode='escape_hold', movement=self.prev_movement)
            return self.prev_movement

        # 2. Fast marching travel time to the goal.
        phi = np.ones((self.n, self.n), dtype=np.float32)
        phi[goal_iy, goal_ix] = -1.0
        speed = np.clip(1.0 / (1.0 + cost), 0.01, 1.0)
        try:
            dist = skfmm.distance(np.ma.MaskedArray(phi, mask=blocked))
            travel_time = np.ma.filled(dist / (speed + 1e-6), fill_value=np.nan)
        except Exception as e:                            # e.g. goal fully enclosed
            print("FMM failed:", repr(e))
            return self._smooth(self._fallback_step(cost, self.prev_movement))

        forces = self._wall_force(self._pos) + self._obstacle_force(self._pos)
        goal_force = self.goal_force() if nearest_human > 2.0 else np.zeros(2, np.float32)
        forces = forces + goal_force
        c = self.center

        # 3a. Robot cell unreachable in the FMM field: grid search, blended with forces.
        if blocked[c, c] or np.isnan(travel_time[c, c]):
            step = self._fallback_step(cost, self.prev_movement)
            movement = _unit(step + forces)
            movement = self._smooth(movement if movement is not None else step)
            self.debug = dict(mode='h_avoid', movement=movement, forces=forces)
            return movement

        # 3b. Normal case: follow the travel-time gradient around the robot.
        grad_y, grad_x = np.gradient(travel_time)
        win = np.s_[max(c - 2, 0):c + 3, max(c - 2, 0):c + 3]
        with np.errstate(invalid='ignore'):
            d = -np.array([np.nanmean(grad_x[win]), np.nanmean(grad_y[win])])
        d = np.nan_to_num(d)
        direction = _unit(d) if np.hypot(*d) > 1e-4 else _unit(np.array([gx, gy]))
        if direction is None:
            direction = np.zeros(2, dtype=np.float32)
        movement = _unit(direction + forces)
        if movement is None:
            movement = _unit(self._wall_force(self._pos))
        if movement is None:
            movement = direction
        self.debug = dict(mode='normal', direction=direction, forces=forces, goal_force=goal_force,
                          nearest_human=nearest_human, movement=movement)

        if self._is_safe(*(movement * self.robot_speed * self.dt)):
            self.prev_movement = movement
            return movement
        return self._smooth(self._fallback_step(cost, movement))

    # ------------------------------------------------------------------
    # Costmaps
    # ------------------------------------------------------------------

    def _human_costmap(self, humans, ps_sizes):
        """Cost from each observed human's predicted PS ellipse, at the current
        and at the predicted position (the latter weighted 0.7)."""
        cost = np.zeros((self.n, self.n), dtype=np.float32)
        if len(humans) == 0:
            return np.empty(0), np.empty(0), cost
        ps = np.maximum(ps_sizes, self.MIN_PS)
        x, y = humans[:, 0], humans[:, 1]
        heading = np.arctan2(-y, -x) - humans[:, 2]         # undo the robot-relative orientation
        cos_a, sin_a = np.cos(heading), np.sin(heading)
        step = self.dt * humans[:, 3] * self.PREDICT_TICKS
        x_next, y_next = x + step * cos_a, y + step * sin_a

        for i in range(len(humans)):
            now = self._ellipse_cost(x[i], y[i], cos_a[i], sin_a[i], ps[i])
            ahead = self._ellipse_cost(x_next[i], y_next[i], cos_a[i], sin_a[i], ps[i])
            cost += np.maximum(now, 0.7 * ahead)
        return x_next, y_next, cost

    def _ellipse_cost(self, hx, hy, cos_a, sin_a, ps):
        """(1 - r) / r inside an asymmetric ellipse (r = normalised distance), 0 outside."""
        front, left, right, back = ps

        dx, dy = self.w_x - hx, self.w_y - hy
        u = dx * cos_a + dy * sin_a                       # along the human's heading
        v = -dx * sin_a + dy * cos_a                      # to the human's left
        r = np.hypot(u / np.where(u >= 0, front, back), v / np.where(v >= 0, left, right))
        inside = np.clip((1.0 - r) / (r + 1e-6), 0.0, None)
        outside = self.SOFT_EDGE_COST * np.exp(-(r - 1.0) / self.SOFT_EDGE_WIDTH) * (r >= 1.0)
        return inside + outside

    def _obstacle_costmap(self, margin=0.1):
        cost = np.zeros((self.n, self.n), dtype=np.float32)
        for obs in self.obstacles:
            loc = obs.pos - self._pos
            r = np.hypot(self.w_x - loc[0], self.w_y - loc[1]) / (obs.radius + margin)
            cost += np.clip((1.0 - r) / (r + 1e-6), 0.0, None)
        return cost

    def _wall_costmap(self, margin=0.2):
        wx, wy = self.w_x + self._pos[0], self.w_y + self._pos[1]
        d = np.minimum(np.minimum(wx, self.width - wx), np.minimum(wy, self.height - wy)) / margin
        return np.clip((1.0 - d) / np.clip(d, 1e-3, None), 0.0, None)

    def _blocked_mask(self):
        """Cells the robot centre cannot occupy: inflated obstacles and walls."""
        blocked = np.zeros((self.n, self.n), dtype=bool)
        for obs in self.obstacles:
            loc = obs.pos - self._pos
            blocked |= np.hypot(self.w_x - loc[0], self.w_y - loc[1]) < obs.radius + self.robot_radius
        m = self.robot_radius - 0.01
        wx, wy = self.w_x + self._pos[0], self.w_y + self._pos[1]
        return blocked | (wx < m) | (wx > self.width - m) | (wy < m) | (wy > self.height - m)

    # ------------------------------------------------------------------
    # Local goal
    # ------------------------------------------------------------------

    def _resolve_goal(self, goal, max_iters=3):
        """If an obstacle cuts the line to the goal, slide the local goal along
        the obstacle's (inflated) boundary, staying within the costmap window."""
        goal = np.array(goal, dtype=np.float32)
        window = self.MAP_SIZE / 2
        margin = 2 * self.RESOLUTION
        for _ in range(max_iters):
            hits = []
            for obs in self.obstacles:
                loc = obs.pos - self._pos
                r = obs.radius + self.robot_radius + margin
                t = _segment_circle_entry((0.0, 0.0), goal, loc, r)
                if t is not None:
                    hits.append((t, loc, r))
            if not hits:
                break
            _, center, r = min(hits, key=lambda h: h[0])
            point_at = lambda a: center + r * np.array([np.cos(a), np.sin(a)], dtype=np.float32)
            robot_angle = np.arctan2(-center[1], -center[0])
            goal_angle = np.arctan2(goal[1] - center[1], goal[0] - center[0])
            delta = np.arctan2(np.sin(goal_angle - robot_angle), np.cos(goal_angle - robot_angle))
            if np.linalg.norm(point_at(robot_angle)) > window:
                break
            if np.linalg.norm(point_at(goal_angle)) <= window:
                goal = point_at(goal_angle)
                continue
            lo, hi = 0.0, delta          # binary search for where the arc leaves the window
            for _ in range(20):
                mid = (lo + hi) / 2.0
                lo, hi = (mid, hi) if np.linalg.norm(point_at(robot_angle + mid)) <= window else (lo, mid)
            goal = point_at(robot_angle + lo)
        lower = np.array([self.WALL_SAFE, self.WALL_SAFE], dtype=np.float32) - self._pos
        upper = np.array([self.width - self.WALL_SAFE, self.height - self.WALL_SAFE], dtype=np.float32) - self._pos
        return np.clip(goal, lower, upper)

    # ------------------------------------------------------------------
    # Safety and forces
    # ------------------------------------------------------------------

    def _is_safe(self, rx, ry):
        self.last_unsafe = 0
        """Is a step to (rx, ry) (relative) clear of predicted humans, obstacles and walls?"""
        if len(self._x_next) and np.any(np.hypot(self._x_next - rx, self._y_next - ry)
                                    < self.robot_radius + self.human_radius + self.SAFETY_MARGIN):
            self.last_unsafe = 'human'
            return False
        for obs in self.obstacles:
            loc = obs.pos - self._pos
            if np.hypot(loc[0] - rx, loc[1] - ry) < self.robot_radius + obs.radius + self.SAFETY_MARGIN:
                return False
        hx = np.concatenate([self._x_now, self._x_next]); hy = np.concatenate([self._y_now, self._y_next])
        if len(hx) and np.any(np.hypot(hx - rx, hy - ry) < self.robot_radius + self.human_radius + self.SAFETY_MARGIN):
            return False
        x, y = self._pos[0] + rx, self._pos[1] + ry
        ok = self.WALL_SAFE < x < self.width - self.WALL_SAFE and self.WALL_SAFE < y < self.height - self.WALL_SAFE
        if not ok:
            self.last_unsafe = 'wall'
        return self.WALL_SAFE < x < self.width - self.WALL_SAFE and self.WALL_SAFE < y < self.height - self.WALL_SAFE

    def _clearance_escape(self):
        """Unit vector away from every obstacle/detected human inside its safety margin."""
        force = np.zeros(2, dtype=np.float32)
        away = [(self._pos - o.pos, o.radius) for o in self.obstacles]
        away += [(-np.asarray(rel, dtype=np.float32), self.human_radius) for rel in self._nearby]
        for diff, radius in away:
            dist = np.linalg.norm(diff)
            if dist < self.robot_radius + radius + self.SAFETY_MARGIN:
                force += diff / (dist + 1e-8)
        return _unit(force)

    def _clearance(self, rx, ry):
        """Smallest gap (m) to any human (current or predicted), obstacle or wall after a step to (rx, ry)."""
        gap = np.inf
        hx = np.concatenate([self._x_now, self._x_next]); hy = np.concatenate([self._y_now, self._y_next])
        if len(hx):
            gap = float(np.min(np.hypot(hx - rx, hy - ry))) - (self.robot_radius + self.human_radius)
        for obs in self.obstacles:
            loc = obs.pos - self._pos
            gap = min(gap, float(np.hypot(loc[0] - rx, loc[1] - ry)) - (self.robot_radius + obs.radius))
        x, y = self._pos[0] + rx, self._pos[1] + ry
        return min(gap, x - self.WALL_SAFE, self.width - self.WALL_SAFE - x,
                    y - self.WALL_SAFE, self.height - self.WALL_SAFE - y)

    def goal_force(self):
        """Constant-magnitude pull towards the local goal cell."""
        if self.goal_idx is None:
            return np.zeros(2, dtype=np.float32)
        d = (np.array(self.goal_idx, dtype=np.float32) - self.center) * self.RESOLUTION
        u = _unit(d)
        return u * self.GOAL_FORCE if u is not None else np.zeros(2, dtype=np.float32)

    def _wall_force(self, pos, strength=2.0):
        m = self.ESCAPE_WALL_MARGIN
        low = np.maximum(m - pos, 0.0)
        high = np.maximum(pos - (np.array([self.width, self.height]) - m), 0.0)
        return (strength * (low - high) / m).astype(np.float32)

    def _obstacle_force(self, pos, strength=2.0):
        force = np.zeros(2, dtype=np.float32)
        for obs in self.obstacles:
            diff = pos - obs.pos
            dist = np.linalg.norm(diff)
            safe = self.robot_radius + obs.radius + self.SAFETY_MARGIN
            if dist < safe:
                force += diff / (dist + 1e-8) * strength * (safe - dist) / safe
        return force

    def escape_forces(self, pos):
        """Wall + obstacle + goal force at an arbitrary position (for rendering)."""
        pos = np.asarray(pos, dtype=np.float32)
        return self._wall_force(pos) + self._obstacle_force(pos) + self.goal_force()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _search_radius(self, base=0.15, max_r=0.6, grow_after=10, step_every=5):
        """Widen the fallback search the longer the robot is stuck."""
        extra = max(0, self.stuck_ticks - grow_after) // step_every
        return min(max_r, base + extra * 0.075)

    def _fallback_step(self, cost, preferred=None):
        """Lowest-cost safe cell near the robot, preferring cells within 90° of
        ``preferred``. Sets ``safety_triggered`` if no safe cell exists."""
        r = max(1, int(self._search_radius() / self.RESOLUTION))
        c = self.center
        win = np.s_[c - r:c + r + 1, c - r:c + r + 1]
        wx, wy = self.w_x[win].ravel(), self.w_y[win].ravel()
        order = np.argsort(cost[win], axis=None)
        cands = [(wx[i], wy[i]) for i in order if abs(wx[i]) > 1e-9 or abs(wy[i]) > 1e-9]
        pref = _unit(np.asarray(preferred, dtype=np.float32)) if preferred is not None else None

        if pref is not None:
            for rx, ry in cands:
                if rx * pref[0] + ry * pref[1] >= 0.0 and self._is_safe(rx, ry):
                    return _unit(np.array([rx, ry]))
        for rx, ry in cands:
            if self._is_safe(rx, ry):
                return _unit(np.array([rx, ry]))

        self.safety_triggered = True
        if cands:
            best = max(cands, key=lambda c: self._clearance(*c))
            return _unit(np.array(best))
        for escape in (self._clearance_escape(), _unit(self._wall_force(self._pos))):
            if escape is not None:
                return escape
        return np.zeros(2, dtype=np.float32)

    def _smooth(self, movement, alpha=0.6):
        """Blend with the previous direction; stops the grid search flip-flopping
        between two cells on alternate ticks."""
        if self.prev_movement is not None:
            blended = _unit(alpha * movement + (1.0 - alpha) * self.prev_movement)
            movement = blended if blended is not None else movement
        self.prev_movement = np.asarray(movement, dtype=np.float32)
        return self.prev_movement
