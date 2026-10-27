import gymnasium as gym
from networkx import edges
import numpy as np
from human_agent import HumanAgent, collision_reaction, disturb_group, violate_PS
from robot import Robot
 
import numpy as np
import random
from collections import deque
 
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
 
import skfmm
from stable_baselines3.common.callbacks import BaseCallback
 
from stable_baselines3.common.buffers import DictReplayBuffer
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
 
import pandas as pd
 
class HumanOnlyReplayBuffer(DictReplayBuffer):
    def add(self, obs, next_obs, action, reward, done, infos):
        if not any(info.get("human_nearby", False) for info in infos):
            return
        super().add(obs, next_obs, action, reward, done, infos)
 
class GroupSpawner:
    """
    Spawns groups
    """
 
    def __init__(self, width, height, num_humans):
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.goal_positions = [[3.5, 3.5], [0.6, 3.5], [3.5, 0.6], [0.6, 0.6]]
        self.goal_index = 0
        self.humans = []                # all active HumanAgent instances
        self.group_goals = {}           # group_id -> np.array([gx, gy])
        self.group_members = {}         # group_id -> [HumanAgent, ...]
        self.group_arrived = {}         # group_id -> bool
        self.spawn_queue = deque()      # queued group sizes to spawn
        self.remaining = num_humans     # humans yet to be spawned
        self.group_id_counter = 0
        self.human_id_counter = 1
 
        self._fill_spawn_queue()
 
    def try_spawn(self):
 
        while self.spawn_queue:
            group_size = self.spawn_queue[0]
            positions = self._find_free_edge_positions(group_size)
            #positions = np.array([2.0, 2.0], dtype=np.float32)
            if positions is None:
                break
            self.spawn_queue.popleft()
            self._spawn_group(positions, group_size)
 
    def check_group_arrived(self, group_id):
        members = self.group_members.get(group_id, [])
        if not members:
            return
        if all(m.arrived for m in members):
            # Assign a new goal, reusing the current position as the "spawn" reference
            new_goal = self._assign_group_goal(members[0].pos, len(members))
            self.group_goals[group_id] = new_goal
 
            for member in members:
                member.goal = new_goal.copy()
                member.arrived = False
 
    def get_nearby_humans(self, pos, radius=4.0, top_n=10):
        """
        Find the closest humans around robot
        """
        in_radius = [
            h for h in self.humans
            if np.linalg.norm(h.pos - pos) <= radius
        ]
        in_radius.sort(key=lambda h: np.linalg.norm(h.pos - pos))
        return in_radius[:top_n]
 
    def _fill_spawn_queue(self):
        while self.remaining > 0:
            group_size = 1
            self.spawn_queue.append(group_size)
            self.remaining -= group_size
 
 
    def _find_free_edge_positions(self, group_size, min_separation=1.5):
        """
        Finds position for groups to be spawned
        """
        margin = 0.1  # distance from actual wall boundary
 
        edges = {
            'top':    [(float(x), self.height - margin)
                        for x in np.linspace(margin, self.width - margin,
                                               int(self.width * 2))],
            'bottom': [(float(x), margin)
                        for x in np.linspace(margin, self.width - margin,
                                               int(self.width * 2))],
            'left':   [(margin, float(y))
                        for y in np.linspace(margin, self.height - margin,
                                            int(self.height * 2))],
            'right':  [(self.width - margin, float(y))
                        for y in np.linspace(margin, self.height - margin,
                                            int(self.height * 2))]
        }
 
        # Exclude corners
        corners = {
            (margin, margin),
            (margin, self.height - margin),
            (self.width - margin, margin),
            (self.width - margin, self.height - margin)
        }
 
        edge_list = list(edges.items())
        random.shuffle(edge_list)  # randomise which edge is tried first
 
        for edge_name, candidates in edge_list:
            candidates = [c for c in candidates if c not in corners]
            #random.shuffle(candidates)
 
            for i in range(len(candidates) - group_size + 1):
                window = candidates[i: i + group_size]
 
                # Check all window positions are free
                if all(self._is_position_free(np.array(pos), min_separation)
                        for pos in window):
                    return [np.array(pos, dtype=np.float32) for pos in window]
 
        return None  # no free position found
 
    def _is_position_free(self, pos, min_separation):
        """
        Returns True if no existing human is within min_separation.
        """
        for human in self.humans:
            if np.linalg.norm(human.pos - pos) < min_separation:
                return False
        return True
 
    def _spawn_group(self, positions, group_size):
        """
        Creates HumanAgent and registers the group.
        """
        from human_agent import HumanAgent  # local import to avoid circular
 
        group_id = self.group_id_counter
        self.group_id_counter += 1
 
        speed = random.uniform(0.5, 1.5)  # shared group speed
        gx, gy = self.goal_positions[self.goal_index]
        self.goal_index = (self.goal_index + 1) % len(self.goal_positions)
        goal = np.array([gx, gy], dtype=np.float32)
        #goal = self._assign_group_goal(positions[0], group_size)
        members = []

        human_id = self.human_id_counter
        if human_id == 1:
            human = HumanAgent(
                pos=np.array([0.3, 2.0], dtype=np.float32),
                goal=goal.copy(),
                group_id=group_id,
                human_id=human_id,
                group_size=group_size,
                speed=speed,
                width=self.width,
                height=self.height
            )
        else:
            human = HumanAgent(
                pos=np.array([2.0, 1.5], dtype=np.float32),
                goal=np.array([2.0, 1.5], dtype=np.float32),
                group_id=group_id,
                human_id=human_id,
                group_size=group_size,
                speed=speed,
                width=self.width,
                height=self.height
            )
        self.human_id_counter += 1
        self.humans.append(human)
        members.append(human)
    
        self.group_goals[group_id] = goal
        self.group_members[group_id] = members
        self.group_arrived[group_id] = False


    def _assign_group_goal(self, spawn_pos, group_size):
        """
        Assigns a goal on the opposite side of spawn location.
        """
        x, y = float(spawn_pos[0]), float(spawn_pos[1])
        margin = 0.4
 
        # Determine which edge the group spawned on
        left   = x < self.width * 0.1
        right  = x > self.width * 0.9
        bottom = y < self.height * 0.1
        top    = y > self.height * 0.9
 
        if left:
            gx = self.width - margin
            gy = random.uniform(margin, self.height - group_size - margin)
        elif right:
            gx = margin
            gy = random.uniform(margin, self.height - group_size - margin)
        elif bottom:
            gx = random.uniform(margin, self.width - group_size - margin)
            gy = self.height - margin
        else:
            gx = random.uniform(margin, self.width - group_size - margin)
            gy = margin
        gx, gy = self.goal_positions[self.goal_index]
        self.goal_index = (self.goal_index + 1) % len(self.goal_positions)
        return np.array([gx, gy], dtype=np.float32)
 
class SocialNavEnv(gym.Env):
    def __init__(self, width=4, height=4, num_humans=2):
        self.obstacle = []
        self.width = width
        self.height = height
        self.dt = 0.05  # simulation timestep
        self.tot_soc_pen = 0.0
        self.move_reward = 0.0
        self.base_shaping_reward = 0.0
        self.collision_reward = 0.0
        self.trajectory = None
        self.rel_goal_pos = None
        self.num_humans = num_humans
        self.action = None
        self.truncate = False
        self.couter = 0

        self.wall_margin = 0.4
 
        self.prev_valence = np.zeros((2,2),dtype=np.float32)
        self.prev_phi = 0.0
 
        self.check_frequ = 5000
        self.obs_buffer = deque(maxlen=5000)
        self.safety_trigger = False
        self._human_detected = False
        self._prev_movement = None  # for h_avoid_mode smoothing, see _get_movement
        self._stuck_ticks = 0       # consecutive ticks with no NET progress, see step()
        self._pos_history = []
        self._pos_history_len = 20  # ~1s of sim time at dt=0.05 -- window for "net progress"
        self.observation_space = gym.spaces.Dict({
            'agents': gym.spaces.Box(
                low=-50, high=50,
                shape=(2, 8), dtype=np.float32
            )
        })
 
        self._prev_raw_orientation = None
        self._unwrap_offset = 0.0
        self.h_arrived = 0
 
        self.move_dist = 0.0
 
        self.current_step = 0
        self.max_steps = 500
        self.cascade_pen = 0
        self.trigger_count = 0
        self.min_human_dist = 1000.0
        self.cascade_val = 0.0
        self.cascade_int = 0.0
        self.human_interaction = 0
 
        self.cascade_mp = 0.0
        self.cascade_sp = 0.0
        self.cascade_cp = 0.0
 
        self._debug_repulsive =None
        self._debug_repulsive_extent = None
        self.resolve_goal = None

        self.goal_x_idx, self.goal_y_idx = None, None

        self.robot = Robot(self.width, self.height)
        self.spawner = GroupSpawner(self.width, self.height, self.num_humans)
        for _ in range(10):
            self.spawner.try_spawn()
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        self.previous_reward = 0.0
        # Small action space - velocity commands
        self.action_space = gym.spaces.Box(
            low=0, high=2.0,
            shape=(2, 3), dtype=np.float32
        )
    
    def _get_obs(self):
        obs = self.robot.get_observation(self.spawner.humans)
        self._human_detected = np.any(obs[:, -1] != 0)
        agent_matrix = np.zeros((2, 8), dtype=np.float32)
        for i, human in enumerate(obs[:2]):
            if human[6] != 0.0:
                rel_x       = human[0] - self.robot.pos[0]
                rel_y       = human[1] - self.robot.pos[1]
                true_valence     = human[2]
                true_intensity   = human[3]
                world_frame_orientation = human[4]
 
                # Convert the orientation from world frame to robot relative orientation
                target_orientation = np.arctan2(-rel_y, -rel_x)
                orientation = target_orientation - world_frame_orientation
                orientation = np.arctan2(np.sin(orientation), np.cos(orientation))
                """
                # Add noise and uncertainty to observed valence and intensity
                # Typical acuracy of SOTA emotion recognition algorithm is about 60 - 75%
                abs_deg = np.degrees(abs(orientation))
                if abs_deg <= 30.0:
                    accuracy_chance = 0.65
                    jitter_range = 0.15
                elif abs_deg <= 75.0:
                    factor = (abs_deg - 60.0) / 45.0
                    accuracy_chance = 0.65 - (factor * 0.45)
                    jitter_range = 0.15 + (factor * 0.1)
                else:
                    accuracy_chance = 0.0
                    jitter_range = 0.0
 
                if abs(true_valence) != 0.0 and abs_deg <= 75.0:
                    if random.random() < accuracy_chance:
                        # Accurate detection
                        # There might be a 10 - 15% jitter
                        v_jitter = random.uniform(-jitter_range, jitter_range)
                        i_jitter = random.uniform(-jitter_range, jitter_range)
 
                        valence = true_valence + v_jitter
                        intensity = true_intensity + i_jitter
                    else:
                        # made the wrong detection
                        if abs(true_valence) <= 0.7:
                            valence = random.uniform(-0.2, 0.2)
                            intensity = random.uniform(0.0, 0.3)
                        else:
                            v_jitter = random.uniform(-jitter_range, jitter_range)
                            i_jitter = random.uniform(-jitter_range, jitter_range)
                            
                            valence = -true_valence + v_jitter
                            intensity = true_intensity + i_jitter    
 
                    valence = np.clip(valence, -1.0, 1.0)
                    intensity = np.clip(intensity, 0.0, 1.0)  
                            
                else:
                    valence = 0.0
                    intensity = 0.0
                """
                speed       = human[5]
                human_id    = human[6]
                dist        = np.sqrt(rel_x**2 + rel_y**2)
                self.human_interaction += 1
                if dist < self.min_human_dist:
                    self.min_human_dist = dist
                agent_matrix[int(human_id-1)] = [rel_x, rel_y, orientation, speed, dist, true_valence, true_intensity, human_id]

        gx, gy = self.robot.goal if self.robot.goal is not None else self.robot.pos
 
        rel_goal_x = gx - self.robot.pos[0]
        rel_goal_y = gy - self.robot.pos[1]
        self.rel_goal_pos = (rel_goal_x, rel_goal_y)
 
        return {'agents': agent_matrix}
 
    def _get_info(self, num_step, avg_pen, avg_safety_trigger):
        gx, gy = self.robot.goal if self.robot.goal is not None else self.robot.pos
        goal = self.robot.goal if self.robot.goal is not None else self.robot.pos
        human_matrix = self._get_obs()["agents"]
        raw = float(human_matrix[0, 2]) if len(human_matrix) > 0 else 0.0
 
        return {
            'robot_pos':     tuple(self.robot.pos),
            'goal_pos':      (gx, gy),
            'dist_to_goal':  float(np.linalg.norm(self.robot.pos - goal)),
            'time_used':     self.robot.time_stamp,
            'tot_soc_pen':   self.tot_soc_pen,
            'base_shaping_reward': self.base_shaping_reward,
            'move_reward':   self.move_reward,
            'collision_reward':   self.collision_reward,
            'human_nearby':  self._human_detected,
            'orientation':   raw,
            'orientation_unwrapped':  self._live_unwrap(raw),
            'speed':         human_matrix[0, 3],
            'human_arrived': self.h_arrived,
            'episode_step' : num_step,
            'avg_pen':       avg_pen,
            'avg_trigger':   avg_safety_trigger,
            'cascade_cp':    self.cascade_cp,
            'cascade_mp':    self.cascade_mp,
            'cascade_sp':    self.cascade_sp,
        }
 
    def _live_unwrap(self, raw):
        if self._prev_raw_orientation is None:
            self._prev_raw_orientation = raw
            return raw
        delta = raw - self._prev_raw_orientation
        if delta > np.pi:
            self._unwrap_offset -= 2 * np.pi
        elif delta < -np.pi:
            self._unwrap_offset += 2 * np.pi
        self._prev_raw_orientation = raw
        return raw + self._unwrap_offset
 
 
    def _get_reward(self, obs):
        reward = 0.0
        self.tot_soc_pen = 0.0
        self.base_shaping_reward = 0.0
        self.collision_reward = 0.0
        self.move_reward = 0.0
        human_matrix = obs["agents"]
 
        gx, gy = self.robot.goal
        prx, pry = self.robot.prev_pos
        crx, cry = self.robot.pos
 
        prev_dist = np.sqrt((gx - prx) ** 2 + (gy - pry) ** 2)
        cur_dist = np.sqrt((gx - crx) ** 2 + (gy - cry) ** 2)
 
        if self._human_detected:
            reward += (prev_dist - cur_dist) * 2.0
            self.move_reward += (prev_dist - cur_dist) * 2.0
            self.cascade_mp += self.move_reward
 
        if self.safety_trigger:
            reward -= 0.5
            self.collision_reward -= 0.5
            self.cascade_cp += self.collision_reward
            self.safety_trigger = False
            self.trigger_count += 1
 
        curr_phi = 0.0
 
        for i in range(len(human_matrix)):
            _, _, _, _, _, valence, intensity, id = human_matrix[i]
            curr_phi -= abs(valence) * intensity
            reward_value = np.clip(valence * intensity * 10.0, -2.5, 2.5)
            reward += reward_value
            self.tot_soc_pen += reward_value
 
            self.cascade_int += intensity
            self.cascade_val += valence
            self.prev_valence[int(id) - 1] = [id, valence]
            
        gamma = 0.99
        shaping = gamma * curr_phi - self.prev_phi
        reward += shaping
        self.prev_phi = curr_phi
        self.base_shaping_reward += shaping
        self.cascade_sp += self.tot_soc_pen
        self.cascade_pen += self.tot_soc_pen
 
        return reward
 
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        margin = 0.2
        if self.truncate:
            for h in self.spawner.humans:
                h.pos = np.array([0.3, 2.0], dtype=np.float32)
            self.robot.pos = np.array([6.8, 2.0], dtype=np.float32)
            self.robot.assign_goal()
        self.truncate = False
        self.robot.time_stamp = 0
        self.human_interaction = 0
        self.cascade_pen = 0
        self.trigger_count = 0
        self.min_human_dist = 1000.0
        self.cascade_int = 0.0
        self.cascade_val = 0.0
        self.current_step = 0
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        self._prev_movement = None
        self._stuck_ticks = 0
        self._pos_history = []

        return self._get_obs(), self._get_info(num_step=0, avg_pen=0, avg_safety_trigger=0)
 
    def step(self, action):
        terminate = False
        truncate = False
        num_step = 0
        avg_safety_trigger = 0
        avg_pen = 0
 
        self.obs_buffer.append(np.concatenate((self._get_obs()["agents"].flatten(), np.array(self.rel_goal_pos, dtype=np.float32))))
        self.couter += 1

        """
        if len(self.obs_buffer) == self.obs_buffer.maxlen and (self.couter % self.check_frequ == 0):
            buffer = np.array(self.obs_buffer)
 
            other_obs = buffer[:, 0:5]
            v_and_i   = buffer[:, 5:7]
 
            other_obs_mean = other_obs - np.mean(other_obs, axis=0)
            v_and_i_mean   = v_and_i - np.mean(v_and_i, axis=0)
 
            o_std = np.std(other_obs, axis=0) + 1e-6
            v_std = np.std(v_and_i, axis=0) + 1e-6
 
            z_o = other_obs_mean / o_std
            z_v = v_and_i_mean / v_std
 
            cross_corr_matrix = np.dot(z_o.T, z_v) / len(self.obs_buffer)
            print("Linear Correlation:", cross_corr_matrix)
            features = ['rel_x', 'rel_y', 'orientation', 'speed', 'dist']
            X = buffer[:, 0:5]
            y_valence = buffer[:, 5]
            y_intensity = buffer[:, 6]
 
            # If you can identify episode boundaries, split whole episodes into train/test
            n = len(X)
            split_idx = int(n * 0.8)
 
            X_train, X_test = X[:split_idx], X[split_idx:]
            yv_train, yv_test = y_valence[:split_idx], y_valence[split_idx:]
            yi_train, yi_test = y_intensity[:split_idx], y_intensity[split_idx:]
            
            model_valence = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, yv_train)
            print("Valence R^2:", r2_score(yv_test, model_valence.predict(X_test)))
            X_train, X_test, yi_train, yi_test = train_test_split(X, y_intensity, test_size=0.2, random_state=0)
            model_intensity = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, yi_train)
            print("Intensity R^2:", r2_score(yi_test, model_intensity.predict(X_test)))
            importances = pd.Series(model_valence.feature_importances_, index=features)
            print(importances.sort_values(ascending=False))
        """

        self.action = action
        self.spawner.try_spawn()
 
        obs = self._get_obs()
        movement = self._get_movement(obs, action, self.rel_goal_pos, robot_radius=self.robot.radius)
        prev_pos = self.robot.pos.copy()
        arrived = self.robot.step(movement, self.dt)

        current_pos = self.robot.pos.copy()

        self._pos_history.append(current_pos.copy())
        if len(self._pos_history) > self._pos_history_len:
            self._pos_history.pop(0)
        if len(self._pos_history) == self._pos_history_len:
            net_progress = np.linalg.norm(current_pos - self._pos_history[0])
            if net_progress < 0.10:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
        self.move_dist = np.linalg.norm(current_pos - prev_pos)
 
        for human in self.spawner.humans:
            human.step(self.dt, self.spawner.humans, self.robot, self.obstacle)
            if human.arrived:
                self.spawner.check_group_arrived(human.group_id)
        
        obs    = self._get_obs()
        reward = self._get_reward(obs)
 
        if arrived:
            terminate = True
            num_step = self.current_step
            avg_pen = self.cascade_pen / num_step
            avg_safety_trigger = self.trigger_count / num_step
 
        if self.current_step >= self.max_steps:
            num_step = self.current_step
            self.current_step = 0
            truncate = True
            self.truncate = truncate
            print("Robot Pos: ", self.robot.pos)
            print("Human Statel: ", obs["agents"])
            for human in self.spawner.humans:
                h_ps = [human.ps_front, human.ps_left, human.ps_right, human.ps_back]
                print("Human PS: ", h_ps)
        self.current_step += 1
 
        return obs, reward, terminate, truncate, self._get_info(num_step, avg_pen, avg_safety_trigger)


    def _get_movement(self, obs, action, rel_goal, map_size=5.0, resolution=0.025, robot_radius=0.25):
        human_matrix = obs["agents"]
        obstacles = self.obstacle
 
        mask = human_matrix[:, 7] != 0.0
        valid_human = human_matrix[mask]
        actual_action = action[mask]
 
        grid_cells = int(map_size / resolution)
        center = grid_cells // 2
 
        grid_y_idx, grid_x_idx = np.mgrid[0:grid_cells, 0:grid_cells]
        w_x, w_y = np.meshgrid(
            (np.arange(grid_cells) - center) * resolution,
            (np.arange(grid_cells) - center) * resolution,
            indexing='xy'
        )
        
        repulsive = np.zeros((grid_cells, grid_cells), dtype=np.float32)
 
        gx = np.clip(rel_goal[0], -map_size/2, map_size/2)
        gy = np.clip(rel_goal[1], -map_size/2, map_size/2)

        self.resolve_goal = (gx, gy)

        gx_idx = int(gx / resolution) + center
        gy_idx = int(gy / resolution) + center
 
        goal_x_idx = np.clip(gx_idx, 0, grid_cells - 1)
        goal_y_idx = np.clip(gy_idx, 0, grid_cells - 1)
 
        world_x = w_x + self.robot.pos[0]
        world_y = w_y + self.robot.pos[1]
 
        x_next, y_next = [], []
 
        phi = np.ones((grid_cells, grid_cells), dtype=np.float32)
        phi[goal_y_idx, goal_x_idx] = -1.0

        nearest_human_dist = 10.0

        if len(valid_human) != 0:
            nearest_human_dist = float(np.min(np.hypot(valid_human[:, 0], valid_human[:, 1])))
            f_size = actual_action[:, 0]
            l_size = actual_action[:, 1]
            b_size = actual_action[:, 2]
            """
            min_ps = 1e-6
            l_size = max(s, min_ps)
            f_size = max(f, min_ps)
            b_size = max(b, min_ps)
            """
 
            x_local = valid_human[:, 0]
            y_local = valid_human[:, 1]
            
            rel_ori     = valid_human[:, 2]
            target_orientation = np.arctan2(-y_local, -x_local)
            ori = target_orientation - rel_ori
            ori = np.arctan2(np.sin(ori), np.cos(ori))
 
            vel     = valid_human[:, 3]
            d_dist    = self.dt * vel * 5.0
            x_next  = x_local + d_dist * np.cos(ori)
            y_next  = y_local + d_dist * np.sin(ori)
 
            cos_a, sin_a = np.cos(ori), np.sin(ori)
 
            for i in range(len(valid_human)):
                dx_g = w_x - x_local[i]
                dy_g = w_y - y_local[i]
 
                dx_n = w_x - x_next[i]
                dy_n = w_y - y_next[i]
 
                u = dx_g * cos_a[i] + dy_g * sin_a[i]
                v = -dx_g * sin_a[i] + dy_g * cos_a[i]
 
                u_n = dx_n * cos_a[i] + dy_n * sin_a[i]
                v_n = -dx_n * sin_a[i] + dy_n * cos_a[i]
                
                semi_long = np.where(u >= 0, f_size[i], b_size[i])
                ellipse_dist = np.sqrt((u / semi_long) ** 2 + (v / l_size[i]) ** 2)
 
                semi_long_n = np.where(u_n >= 0, f_size[i], b_size[i])
                ellipse_dist_n = np.sqrt((u_n / semi_long_n) ** 2 + (v_n / l_size[i]) ** 2)
 
                repulsive_matrix = np.clip((1.0 - ellipse_dist) / (ellipse_dist + 1e-6), 0.0, None)
                repulsive_matrix_n = np.clip(0.7 * (1.0 - ellipse_dist_n) / (ellipse_dist_n + 1e-6), 0.0, None)
                
                fin_matrix = np.maximum(repulsive_matrix, repulsive_matrix_n)
                repulsive += fin_matrix
 
        wall_repulsive = self._get_wall_costmap(world_x, world_y)
        repulsive += wall_repulsive

        goal_dist = np.hypot(w_x - gx, w_y - gy)
        goal_clear_radius = robot_radius + 0.15  # matches _goal_blocked's default margin
        repulsive[goal_dist < goal_clear_radius] = 0.0
 
        self._debug_repulsive = repulsive.copy()
        self._debug_repulsive_extent = [
            self.robot.pos[0] - map_size / 2, self.robot.pos[0] + map_size / 2,
            self.robot.pos[1] - map_size / 2, self.robot.pos[1] + map_size / 2,
        ]
 
        h_avoid_mode = False
 
        speed = np.clip(1.0 / (1.0 + repulsive), 0.01, 1.0)
        try:
            distance_field = skfmm.distance(phi)
            travel_time = distance_field / (speed + 1e-6)

            travel_time = np.ma.filled(travel_time, fill_value=np.nan)
 
        except Exception as e:
            print("fail:", repr(e))
            # Fallback if FMM fails (e.g., goal is completely isolated)
            dx, dy = rel_goal
            if not np.all(np.isinf(repulsive)):
                min_idx = np.unravel_index(np.argmin(repulsive), repulsive.shape)
                best_dy = (min_idx[0] - center) * resolution
                best_dx = (min_idx[1] - center) * resolution
            else:
                best_dx, best_dy = 0.0, 0.0
            norm = np.hypot(best_dx, best_dy) + 1e-8
            return self._apply_safety_check(
                np.array([best_dx / norm, best_dy / norm], dtype=np.float32),
                x_next, y_next, robot_speed=self.robot.max_speed,  # match your Robot's actual speed constant
                obstacles=self.obstacle,
            )
        
        movement = np.array([0, 0], dtype=np.float32)

        if not self._is_safe(0.0, 0.0, x_next, y_next, obstacles=self.obstacle):
            escape_dir = self._clearance_escape_direction()  # unit(pos - obs.pos), summed
            if escape_dir is not None:
                candidate_rx = escape_dir[0] * self.robot.max_speed * self.dt
                candidate_ry = escape_dir[1] * self.robot.max_speed * self.dt
                if self._is_safe(candidate_rx, candidate_ry, x_next, y_next, obstacles=self.obstacle):
                    self.safety_trigger = True
                    self._set_dbg_movement('early_escape', movement=escape_dir)
                    return escape_dir

        goal_force = self._goal_attraction_force(goal_x_idx, goal_y_idx) if nearest_human_dist > 2.0 else np.zeros(2, dtype=np.float32)

        if h_avoid_mode:
            movement = self._fallback_step(w_x, w_y, repulsive, x_next, y_next,
                                    self.obstacle, center, resolution,
                                    search_radius_m=self._escalated_search_radius(),
                                    preferred_direction=self._prev_movement)
            # blend even the fallback choice with the escape force before returning
            wall_force = self._wall_escape_force()
            combined = movement + wall_force + goal_force

            combined_norm = np.linalg.norm(combined)
            if combined_norm < 1e-6:
                combined = wall_force
                combined_norm = np.linalg.norm(combined)
            if combined_norm > 1e-6:
                movement = combined / combined_norm
            movement = self._smooth_movement(movement)
            self._set_dbg_movement('h_avoid_mode', movement=movement, wall_force=wall_force,
                                    goal_force=goal_force,
                                    combined=combined, stuck_ticks=self._stuck_ticks,
                                    search_radius=self._escalated_search_radius())
            return movement

        # normal path
        grad_y, grad_x = np.gradient(travel_time)

        half = 2
        y0, y1 = max(center - half, 0), min(center + half + 1, grad_y.shape[0])
        x0, x1 = max(center - half, 0), min(center + half + 1, grad_x.shape[1])

        with np.errstate(invalid='ignore'):
            dy = -float(np.nanmean(grad_y[y0:y1, x0:x1]))
            dx = -float(np.nanmean(grad_x[y0:y1, x0:x1]))
        if np.isnan(dx) or np.isnan(dy):
            dx, dy = 0.0, 0.0
        norm_grad = np.hypot(dx, dy) + 1e-8

        if norm_grad > 1e-4:
            best_dx, best_dy = dx, dy
        else:
            best_dx, best_dy = gx, gy

        norm = np.hypot(best_dx, best_dy) + 1e-8
        direction = np.array([best_dx / norm, best_dy / norm], dtype=np.float32)  # from FMM gradient
        #print("Direction: ", direction)
        wall_force = self._wall_escape_force()  # not yet normalized, can be zero
        combined = direction + wall_force + goal_force

        self._dbg_movement = dict(
            mode='normal', robot_pos=self.robot.pos.copy(), goal=self.resolve_goal,
            goal_x_idx=goal_x_idx, goal_y_idx=goal_y_idx, center=center,
            nearest_human_dist=nearest_human_dist,
            norm_grad=norm_grad, dx=dx, dy=dy, gx=gx, gy=gy,
            direction=direction.copy(), wall_force=wall_force.copy(),
            combined=combined.copy(), stuck_ticks=self._stuck_ticks,
            search_radius=self._escalated_search_radius(),
        )
        combined_norm = np.linalg.norm(combined)
        if combined_norm < 1e-6:
            combined = wall_force
            combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            movement = combined / combined_norm
        else:
            movement = direction  # or handle degenerate case

        candidate_rx = movement[0] * self.robot.max_speed * self.dt
        candidate_ry = movement[1] * self.robot.max_speed * self.dt
        if self._is_safe(candidate_rx, candidate_ry, x_next, y_next, obstacles=self.obstacle):
            self._prev_movement = movement.astype(np.float32)
            return movement
        else:
            fb = self._fallback_step(w_x, w_y, repulsive, x_next, y_next,
                                        self.obstacle, center, resolution,
                                        search_radius_m=self._escalated_search_radius(),
                                        preferred_direction=movement)
            # Same 2-cycle risk as h_avoid_mode's fallback use -- see
            # `_smooth_movement`'s docstring -- since this is the same
            # `_fallback_step` re-picking from scratch every tick.
            return self._smooth_movement(fb)

    def _clearance_escape_direction(self, margin=0.2, robot_radius=0.25, human_radius=0.25):
        force = np.zeros(2, dtype=np.float32)
        any_violated = False
        for obs in self.obstacle:
            diff = self.robot.pos - obs.pos
            dist = np.linalg.norm(diff)
            safe = robot_radius + obs.radius + margin
            if dist < safe:
                any_violated = True
                force += diff / (dist + 1e-8)  # weight could scale by (safe - dist)
        for human in self.spawner.humans:
            diff = self.robot.pos - human.pos
            dist = np.linalg.norm(diff)
            safe = robot_radius + human_radius + margin
            if dist < safe:
                any_violated = True
                force += diff / (dist + 1e-8)
        if not any_violated:
            return None
        norm = np.linalg.norm(force)
        return force / norm if norm > 1e-6 else None

    def _goal_attraction_force(self, goal_x_idx, goal_y_idx, map_size=5.0, resolution=0.025, strength=3.0):
        self.goal_x_idx, self.goal_y_idx = goal_x_idx, goal_y_idx

        grid_cells = int(map_size / resolution)
        center = grid_cells // 2

        dx = (goal_x_idx - center) * resolution
        dy = (goal_y_idx - center) * resolution
        diff = np.array([dx, dy], dtype=np.float32)
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (diff / dist) * strength

    def _smooth_movement(self, movement, alpha=0.6):
        if self._prev_movement is None:
            self._prev_movement = movement.astype(np.float32)
            return self._prev_movement
        blended = alpha * movement + (1.0 - alpha) * self._prev_movement
        norm = np.linalg.norm(blended)
        if norm > 1e-6:
            blended = blended / norm
        else:
            blended = movement
        self._prev_movement = blended.astype(np.float32)
        return self._prev_movement

    def _wall_escape_force(self, strength=2.0):
        x, y = self.robot.pos
        force = np.zeros(2, dtype=np.float32)
        if x < self.wall_margin:
            force[0] += strength * (self.wall_margin - x) / self.wall_margin
        if x > self.width - self.wall_margin:
            force[0] -= strength * (self.wall_margin - (self.width - x)) / self.wall_margin
        if y < self.wall_margin:
            force[1] += strength * (self.wall_margin - y) / self.wall_margin
        if y > self.height - self.wall_margin:
            force[1] -= strength * (self.wall_margin - (self.height - y)) / self.wall_margin
        return force

    def _segment_circle_entry(self, p0, p1, center, r):

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
 
        d = p1 - p0
        f = p0 - center
        a = float(np.dot(d, d))
        if a < 1e-12:
            return 0.0 if float(np.dot(f, f)) < r * r else None
 
        b = 2.0 * float(np.dot(f, d))
        c = float(np.dot(f, f)) - r * r
        if c < 0:
            return 0.0  # p0 already inside the circle
 
        disc = b * b - 4 * a * c
        if disc < 0:
            return None  # segment never reaches the circle
 
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        for t in (t1, t2):
            if 0.0 <= t <= 1.0:
                return float(t)
        return None

    def _get_wall_costmap(self, world_x, world_y, wall_margin=0.2):
        dist_left   = world_x
        dist_right  = self.width  - world_x
        dist_bottom = world_y
        dist_top    = self.height - world_y

        dist_to_wall = np.minimum(np.minimum(dist_left, dist_right),
                                np.minimum(dist_bottom, dist_top))

        norm_dist = dist_to_wall / wall_margin
        # Same smooth-taper fix as above/as the human costmap, instead of a
        # hard cutoff at norm_dist == 1.0.
        return np.clip((1.0 - norm_dist) / (np.clip(norm_dist, 1e-3, None)), 0.0, None)
            
    def _is_safe(self, rx, ry, x_next, y_next, obstacles=None,
             robot_radius=0.25, human_radius=0.25, margin=0.2):
        safe_dist_human = robot_radius + human_radius + margin
        if len(x_next) > 0:
            dists = np.hypot(x_next - rx, y_next - ry)
            if not np.all(dists >= safe_dist_human):
                return False
        
        pos_x = self.robot.pos[0] + rx
        pos_y = self.robot.pos[1] + ry
        wall_margin = 0.1
        if pos_x <= wall_margin or pos_x >= self.width - wall_margin \
                or pos_y <= wall_margin or pos_y >= self.height - wall_margin:
            return False
        
        return True

    def _set_dbg_movement(self, mode, **kwargs):
        """Lightweight debug snapshot for the early-escape and h_avoid_mode
        return paths, which don't have the full set of locals the normal
        path's richer snapshot does. See `dump_movement_debug`."""
        d = {'mode': mode, 'robot_pos': self.robot.pos.copy()}
        d.update(kwargs)
        self._dbg_movement = d

    def dump_movement_debug(self):
        d = getattr(self, '_dbg_movement', None)
        if d is None:
            return "no _get_movement call recorded yet this episode"
        mode = d.get('mode', '?')
        lines = [f"mode={mode}  robot_pos={d['robot_pos']}  stuck_ticks={d.get('stuck_ticks', '?')}  search_radius={d.get('search_radius', '?')}"]
        if mode == 'early_escape':
            lines.append(f"movement={d.get('movement')}  (current position itself was unsafe -- fleeing whatever obstacle/human is violating the margin)")
        elif mode == 'h_avoid_mode':
            lines.append(f"movement={d.get('movement')}  wall_force={d.get('wall_force')}  obstacle_force={d.get('obstacle_force')}  goal_force={d.get('goal_force')}")
            lines.append(f"combined(pre-normalize)={d.get('combined')}  (h_avoid_mode: FMM's own cell was blocked/nan, using _fallback_step's grid search instead of the gradient)")
        else:
            lines += [f"resolved_local_goal(world)={d['goal']}",
                      f"nearest_human_dist={d['nearest_human_dist']:.2f} (goal_force is zeroed whenever this is <= 2.0)",
                      f"goal grid idx=({d['goal_x_idx']},{d['goal_y_idx']}) center={d['center']} obstacle_mask_at_goal_cell={d['obstacle_mask_at_goal']}",
                      f"norm_grad={d['norm_grad']:.4f}  gradient(dx,dy)={d['dx']:.3f},{d['dy']:.3f}  raw_goal_offset(gx,gy)={d['gx']:.3f},{d['gy']:.3f}",
                      f"direction={d['direction']}  wall_force={d['wall_force']}  obstacle_force={d['obstacle_force']}  goal_force={d['goal_force']}",
                      f"combined(pre-normalize)={d['combined']}"]
        return "\n".join(lines)

    def _escalated_search_radius(self, base_m=0.15, max_m=0.6, grow_after=10, step_every=5):
        if self._stuck_ticks <= grow_after:
            return base_m
        extra_steps = (self._stuck_ticks - grow_after) // step_every
        return float(min(max_m, base_m + extra_steps * 0.075))

    def _fallback_step(self, w_x, w_y, repulsive, x_next, y_next, obstacles,
                        center, resolution, search_radius_m=0.15, preferred_direction=None):
        r_cells = max(1, int(search_radius_m / resolution))
        y0, y1 = center - r_cells, center + r_cells + 1
        x0, x1 = center - r_cells, center + r_cells + 1
        window_cost = repulsive[y0:y1, x0:x1]
        order = np.argsort(window_cost, axis=None)

        pref = None
        if preferred_direction is not None:
            pn = np.linalg.norm(preferred_direction)
            if pn > 1e-6:
                pref = preferred_direction / pn

        def _iter_candidates(require_aligned):
            for idx in order:
                yy, xx = np.unravel_index(idx, window_cost.shape)
                rx, ry = w_x[y0+yy, x0+xx], w_y[y0+yy, x0+xx]
                if abs(rx) < 1e-9 and abs(ry) < 1e-9:
                    continue
                if require_aligned and pref is not None:
                    norm_c = np.hypot(rx, ry) + 1e-8
                    if (rx * pref[0] + ry * pref[1]) / norm_c < 0.0:
                        continue
                yield rx, ry

        best_candidate = None
        if pref is not None:
            for rx, ry in _iter_candidates(require_aligned=True):
                if self._is_safe(rx, ry, x_next, y_next, obstacles=obstacles):
                    norm = np.hypot(rx, ry) + 1e-8
                    return np.array([rx/norm, ry/norm], dtype=np.float32)

        for rx, ry in _iter_candidates(require_aligned=False):
            if self._is_safe(rx, ry, x_next, y_next, obstacles=obstacles):
                norm = np.hypot(rx, ry) + 1e-8
                return np.array([rx/norm, ry/norm], dtype=np.float32)
            if best_candidate is None:
                best_candidate = (rx, ry)

        self.safety_trigger = True
        if best_candidate is None:
            escape_dir = self._clearance_escape_direction()
            if escape_dir is not None:
                return escape_dir
            wall_force = self._wall_escape_force()
            wall_norm = np.linalg.norm(wall_force)
            if wall_norm > 1e-6:
                return (wall_force / wall_norm).astype(np.float32)
            return np.zeros(2, dtype=np.float32)
        rx, ry = best_candidate
        norm = np.hypot(rx, ry) + 1e-8
        return np.array([rx/norm, ry/norm], dtype=np.float32)

# ----- Matplotlib Visualiseation -----
 
def get_asymmetric_proxemic_polygon(pos_x, pos_y, orientation, sigma_front, sigma_back, sigma_left, sigma_right, num_points=100):
    if orientation is None:
        orientation = 0.0
        
    gamma = np.linspace(0, 2 * np.pi, num_points)
    sigma_long = np.where(np.cos(gamma) >= 0, sigma_front, sigma_back)
    sigma_lat = np.where(np.sin(gamma) >= 0, sigma_left, sigma_right)

    denom = (np.cos(gamma)**2 / (sigma_long**2)) + (np.sin(gamma)**2 / (sigma_lat**2))
    r = 1.0 / np.sqrt(denom)

    local_x = r * np.cos(gamma)
    local_y = r * np.sin(gamma)
    
    world_x = pos_x + (local_x * np.cos(orientation) - local_y * np.sin(orientation))
    world_y = pos_y + (local_x * np.sin(orientation) + local_y * np.cos(orientation))
    
    return np.column_stack((world_x, world_y))
 
BEHAVIOUR_COLORS = {
    'collision':    'red',
    'disturb_group':'limegreen',
    'violate_ps':   'purple',
    'startled':      'orange',
    'regular':      'lightgray',
}
 
PERSONALITY_MARKERS = {
    'regular': 'o',
    'grumpy':  's',
    'sad':     '^',
    'chill':   'D',
}
 
def get_personality(stability, plasticity):
    if stability >= 0 and plasticity >= 0:
        return 'regular'
    elif stability >= 0 and plasticity < 0:
        return 'chill'
    elif stability < 0 and plasticity >= 0:
        return 'grumpy'
    else:
        return 'sad'

def draw_repulsion_field(env, ax, grid_n=15, extent=2.5):
    xs = np.linspace(-extent, extent, grid_n)
    ys = np.linspace(-extent, extent, grid_n)
    saved_pos = env.robot.pos.copy()

    U, V, X, Y = [], [], [], []
    for dx in xs:
        for dy in ys:
            env.robot.pos = saved_pos + np.array([dx, dy], dtype=np.float32)
            wf = env._wall_escape_force()
            if env.goal_x_idx is not None:
                gf = env._goal_attraction_force(env.goal_x_idx, env.goal_y_idx)   # add this
            else:
                gf = 0.0
            total = wf + gf
            X.append(env.robot.pos[0])
            Y.append(env.robot.pos[1])
            U.append(total[0])
            V.append(total[1])
    env.robot.pos = saved_pos  # restore — don't leave the env mutated

    ax.quiver(X, Y, U, V, color='cyan', alpha=0.7, scale=20, zorder=5)

def draw(env, ax):
    action = env.action
    # print(f"drawing: {len(env.spawner.humans)} humans, robot at {env.robot.pos}")
    # print("Action: ", action)
    # print("Observation: ", env._get_obs())
    ax.clear()
    ax.set_xlim(0.0, env.width)
    ax.set_ylim(0.0, env.height)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, color = 'white', linewidth = 0.2, alpha = 0.3)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
 
    if getattr(env, 'feasible_mask', None) is not None:
        masked = np.ma.masked_where(~env.feasible_mask, env.feasible_mask.astype(float))
        ax.imshow(
            masked,
            extent=[env.feasible_grid_x.min(), env.feasible_grid_x.max(),
                    env.feasible_grid_y.min(), env.feasible_grid_y.max()],
            origin='lower', cmap='Greens', alpha=0.4, zorder=0,
        )
 
    if getattr(env, '_debug_repulsive', None) is not None:
        ax.imshow(
            env._debug_repulsive,
            extent=env._debug_repulsive_extent,
            origin='lower',
            cmap='inferno',
            alpha=0.5,
            zorder=1,
        )
 
    arrow_len = 0.5
    human_matrix = env._get_obs()["agents"]
    matrix_ids = [row[-1] for row in human_matrix] 
 
    for agent in env.spawner.humans:
        h_id = agent.human_id
        a_front, a_left, a_right, a_back = 0, 0, 0, 0
        
        # Directly find which row index matches this human's ID
        if h_id in matrix_ids:
            i = matrix_ids.index(h_id)
            a_front, a_side, a_back = action[i]
            a_left = a_right = a_side
 
        x, y = agent.pos
        orientation = agent.orientation if agent.orientation is not None else 0.0
        ps_front, ps_back, ps_left, ps_right = agent.ps_front, agent.ps_back, agent.ps_left, agent.ps_right
        color = BEHAVIOUR_COLORS.get(agent.behaviour_type, 'lightgray')
        personality = get_personality(agent.stability, agent.plasticity)
        marker = PERSONALITY_MARKERS.get(personality, 'o')
 
        poly_points = get_asymmetric_proxemic_polygon(x, y, orientation, ps_front, ps_back, ps_left, ps_right)
 
        dx = arrow_len * np.cos(orientation)
        dy = arrow_len * np.sin(orientation)
 
        # Plot the arrow pointing from the agent's center (x, y)
        ax.quiver(
            x, y, dx, dy, 
            angles='xy', 
            scale_units='xy', 
            scale=1, 
            color='black',       # Arrow color
            width=0.007,         # Thickness of the shaft
            headwidth=4,         # Width of the arrowhead
            headlength=5         # Length of the arrowhead
        )
 
        proxemic_zone = mpatches.Polygon(
            poly_points, closed=True,
            facecolor='red', edgecolor='red',
            linestyle='--', alpha=0.5
        )
        ax.add_patch(proxemic_zone)

        # Physical body footprint (human radius)
        human_body = mpatches.Circle(
            (x, y), getattr(agent, 'radius', 0.25),
            facecolor=color, edgecolor='white',
            linewidth=0.5, alpha=0.9, zorder=2
        )
        ax.add_patch(human_body)

        if a_front != 0 or a_left != 0 or a_back != 0 or a_right != 0:
            poly_points_action = get_asymmetric_proxemic_polygon(x, y, orientation, a_front, a_back, a_left, a_right)
            proxemic_zone_a = mpatches.Polygon(
                poly_points_action, closed=True, 
                facecolor='blue', edgecolor='blue', 
                linestyle='--', alpha=0.5
            )
            ax.add_patch(proxemic_zone_a)
            
        ax.plot(x, y, marker=marker, color=color, 
                markersize=10, markeredgecolor='white',
                markeredgewidth=0.5, zorder=2)
    
        ax.text(x, y, str(agent.group_id % 10), ha='center', va='center',
                fontsize=5, color='black', zorder=3)
        
    obstacle = env.obstacle
    for o in obstacle:
        pos, rad = o.pos, o.radius
        circle = mpatches.Circle(pos, rad, color='gray', alpha=0.6)
        ax.add_patch(circle)
 
 
    rx, ry = env.robot.pos
    robot_body = mpatches.Circle(
        (rx, ry), getattr(env.robot, 'radius', 0.25),
        facecolor='orange', edgecolor='white',
        linewidth=0.8, alpha=0.85, zorder=3
    )
    ax.add_patch(robot_body)
    ax.plot(rx, ry, marker='*', color='orange',
            markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=4)

    draw_repulsion_field(env, ax)

    if env.robot.goal is not None:
        goal = env.robot.goal
        ax.plot(goal[0], goal[1], 'x',
                color='yellow', markersize=10, markeredgewidth=2, zorder=1)

    if env.resolve_goal is not None:
        goal = env.resolve_goal
        goal = goal + np.array(env.robot.pos)
        ax.plot(goal[0], goal[1], 'x',
                color='cyan', markersize=10, markeredgewidth=2, zorder=1)
        
    legend_elements = [
        mpatches.Patch(color='red', label='Collision'),
        mpatches.Patch(color='limegreen', label='Disturb group'),
        mpatches.Patch(color='lightgray', label='Regular'),
        mpatches.Patch(color='orange', label='Robot')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.6)
 
    for spine in ax.spines.values():
        spine.set_edgecolor('white')