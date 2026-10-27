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

    def _fill_spawn_queue(self):
        while self.remaining > 0:
            is_group = random.random() < 0.2
            """
            if is_group and self.remaining > 1:
                group_size = random.randint(2, min(5, self.remaining))
            else:
                group_size = 1
            """
            group_size = self.remaining
            self.spawn_queue.append(group_size)
            self.remaining -= group_size


    def _find_free_edge_positions(self, group_size, min_separation=1.5):
        """
        Finds position for groups to be spawned
        """
        margin = 0.2  # distance from actual wall boundary

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
            random.shuffle(candidates)

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
        goal = self._assign_group_goal(positions[0], group_size)
        members = []

        for pos in positions:
            human_id = self.human_id_counter
            human = HumanAgent(
                pos=pos.copy(),
                goal=goal.copy(),
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

        return np.array([gx, gy], dtype=np.float32)

class Obstacle:
    def __init__(self, x, y, radius):
        self.pos = np.array([x, y], dtype=np.float32)
        self.radius = radius


class SocialNavEnv(gym.Env):
    def __init__(self, width=5, height=5, num_humans=1):
        self.obstacle = []

        self.width = width
        self.height = height
        self.dt = 0.1  # simulation timestep
        self.tot_soc_pen = 0.0
        self.move_reward = 0.0
        self.social_reward = 0.0
        self.collision_reward = 0.0
        self.trajectory = None
        self.rel_goal_pos = None
        self.num_humans = num_humans
        self.action = None
        self.couter = 0
        self.check_frequ = 5000
        self.obs_buffer = deque(maxlen=5000)
        self.safety_trigger = False
        self._human_detected = False

        self.observation_space = gym.spaces.Dict({
            'agents': gym.spaces.Box(
                low=-50, high=50,
                shape=(2, 8), dtype=np.float32
            ),
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

        self.feasible_mask = None
        self.feasible_grid_x = None   # shift local grid to world coords
        self.feasible_grid_y = None

        self.robot = Robot(self.width, self.height)
        self.spawner = GroupSpawner(self.width, self.height, self.num_humans)
        for _ in range(10):
            self.spawner.try_spawn()

        self.action_space = gym.spaces.Box(
            low=0, high=3.0,
            shape=(2, 3), dtype=np.float32
        )
    
    def _get_obs(self):
        obs = self.robot.get_observation(self.spawner.humans)
        self._human_detected = len(obs) > 0
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

                speed       = human[5]
                human_id    = human[6]
                dist        = np.sqrt(rel_x**2 + rel_y**2)
                self.human_interaction += 1
                if dist < self.min_human_dist:
                    self.min_human_dist = dist
                agent_matrix[i] = [rel_x, rel_y, orientation, speed, dist, valence, intensity, human_id]
            else:
                agent_matrix[i] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
            'social_reward': self.social_reward,
            'move_reward':   self.move_reward,
            'human_nearby':  self._human_detected,
            'orientation':   raw,
            'orientation_unwrapped':  self._live_unwrap(raw),
            'speed':         human_matrix[0, 3],
            'human_arrived': self.h_arrived,
            'episode_step' : num_step,
            'avg_pen':       avg_pen,
            'avg_trigger':   avg_safety_trigger,
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
        self.social_reward = 0.0
        self.move_reward = 0.0
        human_matrix = obs["agents"]

        if self.safety_trigger:
            reward -= 1.0
            self.move_reward -= 1.0
            self.safety_trigger = False
            self.trigger_count += 1

        for i in range(len(human_matrix)):
            _, _, _, _, _, valence, intensity, _ = human_matrix[i]
            reward += valence * intensity * 5.0
            self.tot_soc_pen += valence * intensity * 5.0
            if valence < 0:
                self.cascade_int += intensity
                self.cascade_val += valence

        self.cascade_pen += self.tot_soc_pen

        return reward

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.time_stamp = 0
        self.human_interaction = 0
        self.cascade_pen = 0
        self.trigger_count = 0
        self.min_human_dist = 1000.0
        self.cascade_int = 0.0
        self.cascade_val = 0.0
        self.current_step = 0
        return self._get_obs(), self._get_info(num_step=0, avg_pen=0, avg_safety_trigger=0)

    def step(self, action):
        terminate = False
        truncate = False
        num_step = 0
        avg_safety_trigger = 0
        avg_pen = 0

        self.obs_buffer.append(np.concatenate((self._get_obs()["agents"].flatten(), np.array(self.rel_goal_pos, dtype=np.float32))))
        self.couter += 1

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
            # print("Actoin: ", action)

        self.action = action
        self.spawner.try_spawn()

        obs = self._get_obs()
        movement = self._get_movement(obs, action, self.rel_goal_pos)
        prev_pos = self.robot.pos.copy()
        arrived = self.robot.step(movement, self.dt)
        current_pos = self.robot.pos.copy()
        self.move_dist = np.linalg.norm(current_pos - prev_pos)

        for human in self.spawner.humans:
            human.step(self.dt, self.spawner.humans, self.robot)
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
        self.current_step += 1

        return obs, reward, terminate, truncate, self._get_info(num_step, avg_pen, avg_safety_trigger)


    def _get_movement(self, obs, action, rel_goal, map_size=5.0, resolution=0.025, robot_radius=0.3):
        human_matrix = obs["agents"]
        # print("abs_diff:  ", self.abs_diff)
        mask = human_matrix[:, 7] != 0.0
        valid_human = human_matrix[mask]
        actual_action = action[mask]

        if len(valid_human) == 0:
            dx, dy = rel_goal
            norm = np.hypot(dx, dy) + 1e-8
            return np.array([dx / norm, dy / norm], dtype=np.float32)
        
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
        ori     = valid_human[:, 2]
        vel     = valid_human[:, 3]
        d_dist    = self.dt * vel * 5.0
        x_next  = x_local + d_dist * np.cos(ori)
        y_next  = y_local + d_dist * np.sin(ori)

        grid_cells = int(map_size / resolution)
        center = grid_cells // 2

        grid_y_idx, grid_x_idx = np.mgrid[0:grid_cells, 0:grid_cells]
        w_x, w_y = np.meshgrid(
            (np.arange(grid_cells) - center) * resolution,
            (np.arange(grid_cells) - center) * resolution,
            indexing='xy'
        )

        repulsive = np.zeros((grid_cells, grid_cells), dtype=np.float32)
        cos_a, sin_a = np.cos(ori), np.sin(ori)
        
        gx = np.clip(rel_goal[0], -map_size/2, map_size/2)
        gy = np.clip(rel_goal[1], -map_size/2, map_size/2)

        gx_idx = int(gx / resolution) + center
        gy_idx = int(gy / resolution) + center

        goal_x_idx = np.clip(gx_idx, 0, grid_cells - 1)
        goal_y_idx = np.clip(gy_idx, 0, grid_cells - 1)
                
        phi = np.ones((grid_cells, grid_cells), dtype=np.float32)
        phi[goal_y_idx, goal_x_idx] = -1.0 # Indicate the goal to FMM algorithm

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
            raw_curve = np.exp(1.0 - ellipse_dist) - 1.0
            max_scale = np.exp(1.0) - 1.0
            intensity = raw_curve / max_scale

            raw_curve_n = np.exp(1.0 - ellipse_dist_n) - 1.0
            max_scale = np.exp(1.0) - 1.0
            intensity_n = raw_curve_n / max_scale

            repulsive_matrix = np.where(ellipse_dist < 1.0, 1.0 / (ellipse_dist + 1e-6), 0.0)
            repulsive_matrix_n = np.where(ellipse_dist_n < 1.0, 0.7 * (1.0 / (ellipse_dist_n + 1e-6)), 0.0)
            #repulsive_matrix = np.where(ellipse_dist < 1.0, intensity, 0.0)
            #repulsive_matrix_n = np.where(ellipse_dist_n < 1.0, 0.7 * intensity_n, 0.0)
            
            fin_matrix = np.maximum(repulsive_matrix, repulsive_matrix_n)
            repulsive += fin_matrix

        h_avoid_mode = False

        speed = np.clip(1.0 / (1.0 + repulsive), 0.01, 1.0)
        
        try:
            distance_field = skfmm.distance(phi)
            travel_time = distance_field / (speed + 1e-6)
            """
            if self.robot.prev_pos is not None:
                delta = self.robot.pos - self.robot.prev_pos
                robot_ori = np.arctan2(delta[1], delta[0])
                robot_vel = np.hypot(delta[0], delta[1]) / self.dt
                cos_t, sin_t = np.cos(-robot_ori), np.sin(-robot_ori)
                forward = w_x * cos_t - w_y * sin_t
                lateral = w_x * sin_t + w_y * cos_t
    
                f_reach, b_reach, side_reach = self._get_movable_space(robot_vel)
                f_reach     = max(f_reach, 2 * resolution)
                b_reach     = max(b_reach, 2 * resolution)
                side_reach  = max(side_reach, 2 * resolution)
                semi_long = np.where(forward >= 0, f_reach, b_reach)
                feasible_mask = (forward / semi_long)**2 + (lateral / side_reach)**2 <= 1.0
                travel_time[~feasible_mask] = np.inf
            """
            # Check if all NaN, if yes enter human avoidance
            if np.any(np.isnan(travel_time)):
                h_avoid_mode = True

        except Exception:
            print("fail")
            # Fallback if FMM fails (e.g., goal is completely isolated)
            dx, dy = rel_goal
            norm = np.hypot(dx, dy) + 1e-8
            return self._apply_safety_check(
                np.array([dx / norm, dy / norm], dtype=np.float32),
                x_next, y_next, robot_speed=self.robot.max_speed  # match your Robot's actual speed constant
            )
        
        if h_avoid_mode:
            min_idx = np.unravel_index(np.argmin(repulsive), repulsive.shape)
            best_dy = (min_idx[0] - center) * resolution
            best_dx = (min_idx[1] - center) * resolution
            norm = np.hypot(best_dx, best_dy) + 1e-8
            movement = self._apply_safety_check(
                np.array([best_dx / norm, best_dy / norm], dtype=np.float32),
                x_next, y_next, robot_speed=self.robot.max_speed  # match your Robot's actual speed constant
            )
        
        else:
            # 5. Trace the path gradient backwards from center to goal
            grad_y, grad_x = np.gradient(travel_time)
            
            dy = -grad_y[center, center]
            dx = -grad_x[center, center]
            norm_grad = np.hypot(dx, dy) + 1e-8
            if norm_grad < 1e-4:
                best_dx, best_dy = gx, gy
            else:
                max_dist = 0.1
                best_dx = (dx / norm_grad) * max_dist
                best_dy = (dy / norm_grad) * max_dist

            norm = np.hypot(best_dx, best_dy) + 1e-8
            if norm < 0.01:
                return np.array([0.0, 0.0], dtype=np.float32)
            
            movement = self._apply_safety_check(
                np.array([best_dx / norm, best_dy / norm], dtype=np.float32),
                x_next, y_next, robot_speed=self.robot.max_speed  # match your Robot's actual speed constant
            )

        if np.allclose(movement, 0.0, atol=1e-6):
            dists_now = np.hypot(x_local, y_local)  # humans relative to robot, already local frame
            nearest = np.argmin(dists_now)
            away_dx = -x_local[nearest]
            away_dy = -y_local[nearest]
            norm = np.hypot(away_dx, away_dy) + 1e-8
            escape_dir = np.array([away_dx / norm, away_dy / norm], dtype=np.float32)
            # move a small fixed step away, uncapped by the joint safety check
            step = self.robot.max_speed * self.dt
            return escape_dir * step
        else:
            return(movement)

    def _get_movable_space(self, vel):
        a_max, w_max_0, dt, vel_max = 0.6, 1.5, self.dt, 1.0
        f_reach = vel * dt + 0.5 * a_max * dt**2          # front: cruise + accel headroom
        b_reach = max(0.5 * a_max * dt**2, 0.02)           # back: brake/creep room only, much smaller
        ratio = np.clip(vel / vel_max, 0.0, 1.0)
        side_reach = max(0.05, w_max_0 * dt * (1 - ratio) * f_reach)  # narrows as speed → vel_max
        return f_reach, b_reach, side_reach

    def build_kinematic_mask(grid_x, grid_y, theta_r, f_min, f_max, l_min, l_max):
        # grid_x, grid_y: meshgrid of cell centers in world/grid frame, relative to robot position
        cos_t, sin_t = np.cos(-theta_r), np.sin(-theta_r)   # inverse rotation
        forward = grid_x * cos_t - grid_y * sin_t
        lateral = grid_x * sin_t + grid_y * cos_t

        mask = (forward >= f_min) & (forward <= f_max) & \
            (lateral >= l_min) & (lateral <= l_max)
        return mask
    
    def _apply_safety_check(self, direction, x_next, y_next, robot_speed,
                            robot_radius=0.3, human_radius=0.3, margin=0.05):

        if len(x_next) == 0:
            return direction  # no humans to worry about

        safe_dist = robot_radius + human_radius + margin

        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            rx = direction[0] * robot_speed * scale * self.dt
            ry = direction[1] * robot_speed * scale * self.dt

            # distance from robot's predicted position to every human's predicted position
            dists = np.hypot(x_next - rx, y_next - ry)

            if np.all(dists >= safe_dist):
                if scale < 1.0:
                    self.safety_trigger = True
                return direction * scale   # scale baked into the returned vector

        # Shouldn't reach here since scale=0.0 (robot doesn't move) is always safe
        # relative to the robot's own motion, but guard anyway:
        return np.array([0.0, 0.0], dtype=np.float32)


# ----- Matplotlib Visualiseation -----

def get_asymmetric_proxemic_polygon(pos_x, pos_y, orientation, sigma_front, sigma_side, sigma_back, num_points=100):
    if orientation is None:
        orientation = 0.0
        
    # 1. Sample angles around a circle (relative to agent's heading)
    gamma = np.linspace(0, 2 * np.pi, num_points)
    
    # 2. Assign dynamic longitudinal sigma based on front vs. back
    # cos(gamma) > 0 means the point is in front; cos(gamma) <= 0 means it is behind
    sigma_long = np.where(np.cos(gamma) >= 0, sigma_front, sigma_back)
    
    # 3. Calculate radius for each angle using the asymmetric denominator
    denom = (np.cos(gamma)**2 / (sigma_long**2)) + (np.sin(gamma)**2 / (sigma_side**2))
    r = 1.0 / np.sqrt(denom)
    
    # 4. Transform to local coordinates
    local_x = r * np.cos(gamma)
    local_y = r * np.sin(gamma)
    
    # 5. Rotate and translate to world coordinates
    world_x = pos_x + (local_x * np.cos(orientation) - local_y * np.sin(orientation))
    world_y = pos_y + (local_x * np.sin(orientation) + local_y * np.cos(orientation))
    
    # 6. Stack into an array of (x, y) coordinates
    return np.column_stack((world_x, world_y))

BEHAVIOUR_COLORS = {
    'collision':    'red',
    'disturb_group':'limegreen',
    'violate_ps':   'purple',
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

def draw(env, ax):
    action = env.action
    # print(f"drawing: {len(env.spawner.humans)} humans, robot at {env.robot.pos}")
    # print("Action: ", action)
    # print("Observation: ", env._get_obs())
    ax.clear()
    ax.set_xlim(-0.5, env.width + 0.5)
    ax.set_ylim(-0.5, env.height + 0.5)
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
    arrow_len = 0.5
    human_matrix = env._get_obs()["agents"]
    matrix_ids = [row[-1] for row in human_matrix] 

    for agent in env.spawner.humans:
        h_id = agent.human_id
        a_front, a_side, a_back = 0, 0, 0
        
        # Directly find which row index matches this human's ID
        if h_id in matrix_ids:
            i = matrix_ids.index(h_id)
            a_front, a_side, a_back = action[i]

        x, y = agent.pos
        orientation = agent.orientation if agent.orientation is not None else 0.0
        ps_front, ps_side, ps_back = agent.ps_front, agent.ps_side, agent.ps_back
        color = BEHAVIOUR_COLORS.get(agent.behaviour_type, 'lightgray')
        personality = get_personality(agent.stability, agent.plasticity)
        marker = PERSONALITY_MARKERS.get(personality, 'o')

        poly_points = get_asymmetric_proxemic_polygon(x, y, orientation, ps_front, ps_side, ps_back)

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
            linestyle='--', alpha=0.15
        )
        ax.add_patch(proxemic_zone)

        if a_front != 0 or a_side != 0 or a_back != 0:
            poly_points_action = get_asymmetric_proxemic_polygon(x, y, orientation, a_front, a_side, a_back)
            proxemic_zone_a = mpatches.Polygon(
                poly_points_action, closed=True, 
                facecolor='blue', edgecolor='blue', 
                linestyle='--', alpha=0.15
            )
            ax.add_patch(proxemic_zone_a)
            
        ax.plot(x, y, marker=marker, color=color, 
                markersize=10, markeredgecolor='white',
                markeredgewidth=0.5, zorder=2)
    
        ax.text(x, y, str(agent.group_id % 10), ha='center', va='center',
                fontsize=5, color='black', zorder=3)
        
    obstacle = [
        Obstacle(8,  8,  0.75),
        Obstacle(15, 10, 0.75),
        Obstacle(5,  20, 0.5),
        Obstacle(20, 5,  0.5),
        Obstacle(20, 20, 1.0)
    ]
    for o in obstacle:
        pos, rad = o.pos, o.radius
        circle = mpatches.Circle(pos, rad, color='gray', alpha=0.6)
        ax.add_patch(circle)


    rx, ry = env.robot.pos
    ax.plot(rx, ry, marker='*', color='orange',
            markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=4)
    if env.robot.goal is not None:
        goal = env.robot.goal
        ax.plot(goal[0], goal[1], 'x',
                color='yellow', markersize=10, markeredgewidth=2, zorder=1)
    
    legend_elements = [
        mpatches.Patch(color='red', label='Collision'),
        mpatches.Patch(color='limegreen', label='Disturb group'),
        mpatches.Patch(color='lightgray', label='Regular'),
        mpatches.Patch(color='orange', label='Robot')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.6)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')