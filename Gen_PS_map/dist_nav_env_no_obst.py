"""[No obstacles] Distance-based baseline: the policy observes the nearest human's position,
heading and speed only (no emotion) and predicts a symmetric personal space
[front, side, back]. The reward is progress, the safety penalty, and a fixed
personal-space intrusion penalty; detected reactions are logged, not rewarded."""
import numpy as np

from human_agent import HumanAgent
from base_env_no_obst import BaseNavEnv, HumanOnlyReplayBuffer  # noqa: F401 (re-export)


class DistNavEnvNoObst(BaseNavEnv):
    DT = 0.1
    OBS_WIDTH = 6          # [rel_x, rel_y, rel_orientation, speed, dist, id]
    ID_COL = 5
    ACTION_WIDTH = 3       # [front, side, back]

    # Fixed reference personal space used by the reward (m), in the human's frame.
    COMFORT_FRONT = 1.8
    COMFORT_SIDE = 1.0
    COMFORT_BACK = 0.7

    INTRUSION_WEIGHT = 2.0
    BUFFER = 1.3            # penalty starts at 1.3x the comfort ellipse (graded approach)
    PROGRESS_WEIGHT = 1.0   # potential-based: guides learning, doesn't change optimal policy
    CONTACT_PENALTY = 1.2
    CONTACT_MARGIN = 0.1    # m beyond body contact

    def _get_obs(self):
        agents = np.zeros((self.TOP_N, self.OBS_WIDTH), dtype=np.float32)
        for i, rel_x, rel_y, rel_ori, speed, dist, _, _, hid in self._observed_humans():
            agents[i] = [rel_x, rel_y, rel_ori, speed, dist, hid]
        return self._make_obs(agents)

    def _ps_sizes(self, action):
        """[front, side, back] -> [front, left, right, back]."""
        return action[:, [0, 1, 1, 2]]

    def _get_reward(self, goal, reactions):
        parts = dict(move=0.0, collision=0.0, soc_pen=0.0, shaping=0.0)

        d_old = float(np.linalg.norm(goal - self.robot.prev_pos))
        d_new = float(np.linalg.norm(goal - self.robot.pos))
        arrived = d_new < self.robot.ARRIVAL_RADIUS
        phi_old, phi_new = -self.PROGRESS_WEIGHT * d_old, (0.0 if arrived else -self.PROGRESS_WEIGHT * d_new)
        parts['shaping'] = self.GAMMA * phi_new - phi_old

        robot_vel = self.robot.measured_vel(self.dt)
        for rel_x, rel_y, rel_ori, _, dist, hid in self._obs['agents']:
            if hid == 0.0:
                continue
            r = float(np.sqrt(self._ps_depth_sq(rel_x, rel_y, rel_ori)))
            if r < self.BUFFER:
                depth = (self.BUFFER - r) / self.BUFFER
                to_human = np.array([rel_x, rel_y]) / (dist + 1e-6)
                closing = float(robot_vel @ to_human) / self.robot.MAX_SPEED
                blame = float(np.clip(0.6 + 0.4 * closing, 0.2, 1.0))
                parts['soc_pen'] -= self.INTRUSION_WEIGHT * blame * depth ** 2
            if dist < self.robot.radius + HumanAgent.RADIUS + self.CONTACT_MARGIN:
                parts['collision'] -= self.CONTACT_PENALTY
        return sum(parts.values()), parts

    def _ps_depth_sq(self, rel_x, rel_y, rel_ori):
        """Squared normalised distance of the robot inside the human's fixed PS
        ellipse (< 1 = inside). ``rel_x, rel_y`` = human - robot."""
        heading = np.arctan2(-rel_y, -rel_x) - rel_ori      # undo the robot-relative orientation
        dx, dy = -rel_x, -rel_y                              # human -> robot
        c, s = np.cos(heading), np.sin(heading)
        along, lateral = dx * c + dy * s, -dx * s + dy * c
        sigma = self.COMFORT_FRONT if along >= 0 else self.COMFORT_BACK
        return (along / sigma) ** 2 + (lateral / self.COMFORT_SIDE) ** 2


# Old class name, so existing training scripts keep working:
# ``from dist_nav_env_no_obst import SocialNavEnv``
SocialNavEnv = DistNavEnvNoObst

from render import draw  # noqa: E402,F401  
