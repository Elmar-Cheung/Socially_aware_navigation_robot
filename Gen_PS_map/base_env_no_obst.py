
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer

from human_agent import HumanAgent
from local_planner import LocalPlanner
from perception import ReactionDetector
from robot import Robot


class HumanOnlyReplayBuffer(DictReplayBuffer):
    """Stores only transitions with a human in view (the policy only acts then)."""
    def add(self, obs, next_obs, action, reward, done, infos):
        if any(info.get("human_nearby", False) for info in infos):
            super().add(obs, next_obs, action, reward, done, infos)


class Obstacle:
    def __init__(self, x, y, radius):
        self.pos = np.array([x, y], dtype=np.float32)
        self.radius = radius


class GroupSpawner:
    """Spawns all humans as one group at ``spawn_pos`` and cycles the group
    through ``goals``, re-assigning once every member has arrived."""
    GOALS = [(2.5, 4.6), (4.6, 2.5), (2.5, 0.6),(0.6, 2.5)]
    MEMBER_SPACING = 0.6   # m, members are spread along y around the spawn point

    def __init__(self, width, height, num_humans, rng, spawn_pos=(0.3, 2.0)):
        self.width, self.height, self.rng = width, height, rng
        self.goal_index = 0
        self.arrivals = 0
        speed = float(rng.uniform(0.5, 1.5))      # shared group speed
        goal = self._next_goal()
        offsets = (np.arange(num_humans) - (num_humans - 1) / 2) * self.MEMBER_SPACING
        self.humans = [
            HumanAgent(pos=(spawn_pos[0], spawn_pos[1] + dy), goal=goal.copy(), group_id=0,
                       human_id=i + 1, group_size=num_humans, speed=speed,
                       width=width, height=height, rng=rng)
            for i, dy in enumerate(offsets)
        ]

    def _next_goal(self):
        goal = np.array(self.GOALS[self.goal_index], dtype=np.float32)
        self.goal_index = (self.goal_index + 1) % len(self.GOALS)
        return goal

    def step(self, dt, robot, obstacles):
        for human in self.humans:
            human.step(dt, self.humans, robot, obstacles)
        if self.humans and all(h.arrived for h in self.humans):
            self.arrivals += 1
            goal = self._next_goal()
            for h in self.humans:
                h.goal, h.arrived = goal.copy(), False


class BaseNavEnv(gym.Env):
    """The policy predicts the personal space of the nearest observed human; a
    fast-marching local planner drives the robot around the predicted spaces.
    An episode ends when the robot reaches its goal (the world carries on) or
    after ``MAX_STEPS`` (the world is reset)."""

    # World: same map as base_env.py, but the obstacles are moved far outside it
    # (> 20 m away, well beyond the planner window and the humans' obstacle range).
    OBSTACLES = [(3.0, -30.0, 0.1), (3.0, 34.0, 0.1), (-20.0, -20.0, 0.01)]   # (x, y, radius)
    HUMAN_SPAWN = (0.3, 2.0)

    DT = 0.1
    MAX_STEPS = 500
    TOP_N = 1              # humans in the policy observation
    GAMMA = 0.99           # for potential-based shaping, match the agent's gamma
    SAFETY_PENALTY = 1.5
    PROGRESS_WEIGHT = 1.0
    TIME_PENALTY = 0.005
    # Layout, set by each variant
    OBS_WIDTH = None       # columns per observed human
    ID_COL = None          # column holding the human id (0 = no human)
    ACTION_WIDTH = None

    def __init__(self, width=5, height=5, num_humans=1):
        self.width, self.height, self.num_humans = width, height, num_humans
        self.dt = self.DT
        self.obstacles = [Obstacle(*o) for o in self.OBSTACLES]
        m = 0.4
        self.robot_goals = (m, height - m), (width - m, m), (m, m), (width - m, height - m)
        self.robot = Robot(width, height, self.robot_goals)
        self.planner = LocalPlanner(width, height, self.obstacles, self.dt, Robot.RADIUS,
                                    Robot.MAX_SPEED, HumanAgent.RADIUS)
        # Detected reactions are always logged (info / stats), so both variants
        # can be compared on them; whether they enter the reward is up to the variant.
        self.detector = ReactionDetector(self.dt)

        self.observation_space = gym.spaces.Dict({
            'agents': gym.spaces.Box(-50, 50, shape=(self.TOP_N, self.OBS_WIDTH), dtype=np.float32),
            'obstacles': gym.spaces.Box(-50, 50, shape=(len(self.obstacles), 3), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(0.0, 2.0, shape=(self.TOP_N, self.ACTION_WIDTH), dtype=np.float32)

        self.spawner = None
        self._world_needs_reset = True
        self.action = None
        self.prev_action = None
        self.delta = None

    @property
    def humans(self):
        return self.spawner.humans

    @property
    def obstacle(self):          # old attribute name
        return self.obstacles

    # ------------------------------------------------------------------
    # Variant interface
    # ------------------------------------------------------------------

    def _get_obs(self):
        raise NotImplementedError

    def _ps_sizes(self, action):
        raise NotImplementedError

    def _get_reward(self, goal, reactions):
        raise NotImplementedError

    def _potential(self):
        return 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.robot.goal_index = 0     # reproducible: restart the goal cycle too
        if self._world_needs_reset or seed is not None:
            self._reset_world()
        self.current_step = 0
        self.stats = dict(soc_pen=0.0, move=0.0, collision=0.0, shaping=0.0, human_reaction=0.0,
                          safety_triggers=0, reactions=0, human_interaction=0,
                          min_human_dist=np.inf)
        self._obs = self._get_obs()
        self._prev_phi = self._potential()            # shaping potential of s0
        return self._obs, self._get_info({}, {})

    def _reset_world(self):
        self.spawner = GroupSpawner(self.width, self.height, self.num_humans, self.np_random)
        self.robot.reset()
        self.planner.reset()
        self.detector.reset()
        self._prev_raw_orientation, self._unwrap_offset = None, 0.0
        self._world_needs_reset = False

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.TOP_N, self.ACTION_WIDTH)
        self.action = action
        agents = self._obs['agents']
        seen = agents[:, self.ID_COL] != 0.0

        self.delta = np.abs(action - self.prev_action) if self.prev_action is not None else np.zeros_like(action)
        self.prev_action = action.copy()
        goal = self.robot.goal.copy()             # progress is measured to this goal
        direction = self.planner.plan(self.robot.pos, goal, agents[seen], self._ps_sizes(action[seen]),
                                      self.robot.detect(self.humans).values())
        arrived = self.robot.step(direction, self.dt)
        self._resolve_obstacle_penetration()
        self.planner.record_position(self.robot.pos)
        self.spawner.step(self.dt, self.robot, self.obstacles)

        # Perception, exactly once per step
        self._obs = self._get_obs()
        reactions = self.detector.update(self.robot.detect(self.humans), self.robot.measured_vel(self.dt))
        reward, parts = self._get_reward(goal, reactions)
        self._update_stats(parts, reactions)

        self.current_step += 1
        truncated = self.current_step >= self.MAX_STEPS
        self._world_needs_reset = truncated
        return self._obs, reward, arrived, truncated, self._get_info(parts, reactions, done=arrived or truncated)

    # ------------------------------------------------------------------
    # Helpers for variants
    # ------------------------------------------------------------------

    def _observed_humans(self):
        """Raw perception of the TOP_N nearest humans in range, as
        (row, rel_x, rel_y, rel_orientation, speed, dist, valence, intensity, id).
        rel_orientation is the human's heading relative to the direction
        towards the robot (0 = facing it)."""
        for i, (x, y, valence, intensity, heading, speed, hid) in enumerate(
                self.robot.get_observation(self.humans, self.TOP_N)):
            if hid == 0.0:
                continue
            rel_x, rel_y = x - self.robot.pos[0], y - self.robot.pos[1]
            rel_ori = np.arctan2(-rel_y, -rel_x) - heading
            rel_ori = np.arctan2(np.sin(rel_ori), np.cos(rel_ori))
            yield i, rel_x, rel_y, rel_ori, speed, np.hypot(rel_x, rel_y), valence, intensity, hid

    def _make_obs(self, agents):
        """Observation dict: the agents matrix plus obstacles relative to the robot."""
        obstacles = np.array([[*(o.pos - self.robot.pos), o.radius] for o in self.obstacles], dtype=np.float32)
        return {'agents': agents, 'obstacles': obstacles}

    def _task_reward(self, goal):
        """Reward parts shared by every variant: progress (while a human is in
        view) and the safety-fallback penalty."""
        parts = dict(move=-self.TIME_PENALTY, collision=0.0, soc_pen=0.0,
                 shaping=0.0, human_reaction=0.0, emo_shaping=0.0)
        d_old = float(np.linalg.norm(goal - self.robot.prev_pos))
        d_new = float(np.linalg.norm(goal - self.robot.pos))
        phi_new = 0.0 if d_new < self.robot.ARRIVAL_RADIUS else -self.PROGRESS_WEIGHT * d_new
        parts['shaping'] = self.GAMMA * phi_new + self.PROGRESS_WEIGHT * d_old
        if self.planner.safety_triggered:
            parts['collision'] = -self.SAFETY_PENALTY
        return parts

    @property
    def human_detected(self):
        return bool(np.any(self._obs['agents'][:, self.ID_COL] != 0.0))

    def predicted_ps(self, human_id):
        """The policy's predicted PS [front, left, right, back] for an observed human, or None."""
        if self.action is None:
            return None
        ids = list(self._obs['agents'][:, self.ID_COL])
        return self._ps_sizes(self.action)[ids.index(human_id)] if human_id in ids else None

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _update_stats(self, parts, reactions):
        s = self.stats
        for k in ('soc_pen', 'move', 'collision', 'shaping'):
            s[k] += parts[k]
        s['safety_triggers'] += int(self.planner.safety_triggered)
        s['reactions'] += len(reactions)
        if self.human_detected:
            s['human_interaction'] += 1
            s['min_human_dist'] = min(s['min_human_dist'], float(self._obs['agents'][0, 4]))

    def _get_info(self, parts, reactions, done=False):
        agents = self._obs['agents']
        n = max(self.current_step, 1)
        raw_ori = float(agents[0, 2])
        return {
            'robot_pos': tuple(self.robot.pos),
            'goal_pos': tuple(self.robot.goal),
            'dist_to_goal': float(np.linalg.norm(self.robot.pos - self.robot.goal)),
            'time_used': self.robot.time_stamp,
            'tot_soc_pen': parts.get('soc_pen', 0.0),
            'base_shaping_reward': parts.get('shaping', 0.0),
            'move_reward': parts.get('move', 0.0),
            'collision_reward': parts.get('collision', 0.0),
            'human_nearby': self.human_detected,
            'orientation': raw_ori,
            'orientation_unwrapped': self._unwrap(raw_ori),
            'speed': float(agents[0, 3]),
            'human_arrived': self.spawner.arrivals,
            'episode_step': self.current_step if done else 0,
            'avg_pen': self.stats['soc_pen'] / n if done else 0.0,
            'avg_trigger': self.stats['safety_triggers'] / n if done else 0.0,
            'cascade_mp': self.stats['move'],
            'cascade_sp': self.stats['soc_pen'],
            'cascade_cp': self.stats['collision'],
            'human_reaction_reward': parts.get('human_reaction', 0.0),
            # Perceived reactions vs. ground truth, for validating the detector
            'reactions': reactions,
            'gt_responses': {h.human_id: (h.response, h.startle_reaction) for h in self.humans},
            'last_unsafe': 1 if self.planner.last_unsafe == 'human' else 2 if self.planner.last_unsafe == 'wall' else 0,
            'action_front': float(self.action[0, 0]) if self.action is not None else 0.0,
            'action_left': float(self.action[0, 1]) if self.action is not None else 0.0,
            'action_right': float(self.action[0, 2]) if self.action is not None else 0.0,   
            'action_back': float(self.action[0, 3]) if self.action is not None else 0.0,
            'action_delta_front': float(self.delta[0, 0]) if self.delta is not None else 0.0,
            'action_delta_left': float(self.delta[0, 1]) if self.delta is not None else 0.0,
            'action_delta_right': float(self.delta[0, 2]) if self.delta is not None else 0.0,
            'action_delta_back': float(self.delta[0, 3]) if self.delta is not None else 0.0,
        }

    def _unwrap(self, raw):
        """Continuous version of the relative orientation, for plotting."""
        if self._prev_raw_orientation is not None:
            delta = raw - self._prev_raw_orientation
            if delta > np.pi:
                self._unwrap_offset -= 2 * np.pi
            elif delta < -np.pi:
                self._unwrap_offset += 2 * np.pi
        self._prev_raw_orientation = raw
        return raw + self._unwrap_offset

    def _resolve_obstacle_penetration(self, margin=0.2):
        """Push the robot out of any obstacle it ended up too close to."""
        for obs in self.obstacles:
            diff = self.robot.pos - obs.pos
            dist = float(np.linalg.norm(diff))
            min_dist = obs.radius + self.robot.radius + margin
            if dist < min_dist:
                direction = diff / dist if dist > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
                self.robot.pos = (obs.pos + direction * min_dist).astype(np.float32)
                into = float(self.robot.vel @ -direction)
                if into > 0:
                    self.robot.vel = self.robot.vel + direction * into
        self.robot.clamp_to_bounds()