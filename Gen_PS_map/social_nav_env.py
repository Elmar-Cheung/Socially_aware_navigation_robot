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

        self._fill_spawn_queue()

    def try_spawn(self):
        # Force respawn if population drops too low"
        """
        if len(self.humans) < 15 and not self.spawn_queue:
            self.remaining += 10
            self._fill_spawn_queue()
        
        if len(self.humans) > 35:
            self.remaining -= 10
            self._fill_spawn_queue()
        """

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
            human = HumanAgent(
                pos=pos.copy(),
                goal=goal.copy(),
                group_id=group_id,
                group_size=group_size,
                speed=speed,
                width=self.width,
                height=self.height
            )
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
        margin = 0.2

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
        self.check_frequ = 1000
        self.obs_buffer = deque(maxlen=1000)
        self.safety_trigger = False
        self._human_detected = False
        self.observation_space = gym.spaces.Dict({
            'agents': gym.spaces.Box(
                low=-50, high=50,
                shape=(1, 7), dtype=np.float32
            )
        })

        self._prev_raw_orientation = None
        self._unwrap_offset = 0.0
        self.h_arrived = 0

        self.robot = Robot(self.width, self.height)
        self.spawner = GroupSpawner(self.width, self.height, self.num_humans)
        for _ in range(10):
            self.spawner.try_spawn()
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        self.previous_reward = 0.0
        # Small action space - velocity commands
        self.action_space = gym.spaces.Box(
            low=0, high=3.0,
            shape=(1, 3), dtype=np.float32
        )
    
    def _get_obs(self):
        obs = self.robot.get_observation(self.spawner.humans)
        self._human_detected = len(obs) > 0
        agent_matrix = np.zeros((1, 7), dtype=np.float32)
        for i, human in enumerate(obs[:1]):
            rel_x       = human[0] - self.robot.pos[0]
            rel_y       = human[1] - self.robot.pos[1]
            valence     = human[2]
            intensity   = human[3]
            orientation = human[4]
            speed       = human[5]
            dist        = np.sqrt(rel_x**2 + rel_y**2)
            agent_matrix[i] = [rel_x, rel_y, orientation, speed, dist, valence, intensity]

        gx, gy = self.robot.goal if self.robot.goal is not None else self.robot.pos

        rel_goal_x = gx - self.robot.pos[0]
        rel_goal_y = gy - self.robot.pos[1]
        self.rel_goal_pos = (rel_goal_x, rel_goal_y)

        return {'agents': agent_matrix}

    def _get_info(self):
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
            'collision_reward': self.collision_reward,
            'human_nearby':  self._human_detected,
            'orientation':            raw,
            'orientation_unwrapped':  self._live_unwrap(raw),
            'speed':         human_matrix[0, 3],
            'human_arrived': self.h_arrived,
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
        human_matrix = obs["agents"]

        if self.safety_trigger:
            reward -= 5.0
            self.tot_soc_pen -= 5.0
            self.safety_trigger = False
            self.social_reward -= 5.0

        for i in range(len(human_matrix)):
            _, _, _, _, _, valence, intensity = human_matrix[i]
            reward += valence * intensity * 5.0
            self.tot_soc_pen += valence * intensity * 5.0


        return reward

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.time_stamp = 0
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        print(self.action)
        return self._get_obs(), self._get_info()

    def step(self, action):
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
            
            print(cross_corr_matrix)

        self.action = action
        self.spawner.try_spawn()

        obs = self._get_obs()
        movement = self._get_movement(obs, action, self.rel_goal_pos)

        arrived = self.robot.step(movement, self.dt)

        for human in self.spawner.humans:
            human.step(self.dt, self.spawner.humans, self.robot)
            if human.arrived:
                self.h_arrived = 1
                self.spawner.check_group_arrived(human.group_id)
            else:
                self.h_arrived = 0
        
        obs    = self._get_obs()
        reward = self._get_reward(obs)

        terminate = arrived
        
        return obs, reward, terminate, False, self._get_info()

    def _get_movement(self, obs, action, rel_goal, map_size=5.0, resolution=0.025, robot_radius=0.3):
        human_matrix = obs["agents"]

        mask = human_matrix[:, 2] != 0.0
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
        d_dist    = self.dt * vel
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
        phi[goal_y_idx, goal_x_idx] = -1.0 # Prevents robot staying stationary
        """
        if len(valid_human) >= 2:
            # gather more human data(ish)
            r_matrix = np.zeros((5, grid_cells, grid_cells), dtype=np.float32)
            for i in range(5):
                random_m = np.random.random(-0.5, 0.5, size=(5, 1))
                front_size = f_size - random_m
                random_m = np.random.random(-0.5, 0.5, size=(5, 1))
                lateral_size = l_size - random_m
                random_m = np.random.random(-0.5, 0.5, size=(5, 1))
                back_size = b_size - random_m
                for i in range(len(valid_human)):
                    dx_g = w_x - x_local[i]
                    dy_g = w_y - y_local[i]

                    dx_n = w_x - x_next[i]
                    dy_n = w_y - y_next[i]

                    u = dx_g * cos_a[i] + dy_g * sin_a[i]
                    v = -dx_g * sin_a[i] + dy_g * cos_a[i]

                    u_n = dx_n * cos_a[i] + dy_n * sin_a[i]
                    v_n = -dx_n * sin_a[i] + dy_n * cos_a[i]
                    
                    semi_long = np.where(u >= 0, front_size[i], back_size[i])
                    ellipse_dist = np.sqrt((u / semi_long) ** 2 + (v / lateral_size[i]) ** 2)

                    semi_long_n = np.where(u_n >= 0, front_size[i], back_size[i])
                    ellipse_dist_n = np.sqrt((u_n / semi_long_n) ** 2 + (v_n / lateral_size[i]) ** 2)

                    repulsive_matrix = np.where(ellipse_dist < 1.0, 1.0 / (ellipse_dist + 1e-6), 0.0)
                    repulsive_matrix_n = np.where(ellipse_dist_n < 1.0, 0.7 * (1.0 / (ellipse_dist_n + 1e-6)), 0.0)
                    
                    fin_matrix = np.maximum(repulsive_matrix, repulsive_matrix_n)
                    r_matrix[i] = np.sum(fin_matrix, axis=0)

            for i in range(5):
                h_avoid_mode = False
                # 4. Compute arrival/travel time from the goal to all cells
                try:
                    distance_field = skfmm.distance(phi)
                    travel_time = distance_field / (speed + 1e-6)
                    # Check if all NaN, if yes enter human avoidance
                    if np.any(np.isnan(travel_time)):
                        h_avoid_mode = True

                except Exception:
                    print("fail")
                    # Fallback if FMM fails (e.g., goal is completely isolated)
                    dx, dy = rel_goal
                    norm = np.hypot(dx, dy) + 1e-8
                    return np.array([dx / norm, dy / norm], dtype=np.float32)
                
                if h_avoid_mode:
                    min_idx = np.unravel_index(np.argmin(repulsive), repulsive.shape)
                    best_dy = (min_idx[0] - center) * resolution
                    best_dx = (min_idx[1] - center) * resolution
                    norm = np.hypot(best_dx, best_dy) + 1e-8
                    return np.array([best_dx / norm, best_dy / norm], dtype=np.float32)
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
                    
                    return np.array([best_dx / norm, best_dy / norm], dtype=np.float32)
        """
                
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

            repulsive_matrix = np.where(ellipse_dist < 1.0, 1.0 / (ellipse_dist + 1e-6), 0.0)
            repulsive_matrix_n = np.where(ellipse_dist_n < 1.0, 0.7 * (1.0 / (ellipse_dist_n + 1e-6)), 0.0)
            
            fin_matrix = np.maximum(repulsive_matrix, repulsive_matrix_n)
            repulsive += fin_matrix

        h_avoid_mode = False

        speed = np.clip(1.0 / (1.0 + repulsive), 0.01, 1.0)
        # 4. Compute arrival/travel time from the goal to all cells
        try:
            distance_field = skfmm.distance(phi)
            travel_time = distance_field / (speed + 1e-6)
            # Check if all NaN, if yes enter human avoidance
            if np.any(np.isnan(travel_time)):
                h_avoid_mode = True

        except Exception:
            print("fail")
            # Fallback if FMM fails (e.g., goal is completely isolated)
            dx, dy = rel_goal
            norm = np.hypot(dx, dy) + 1e-8
            return self._apply_safety_check(
                np.array([best_dx / norm, best_dy / norm], dtype=np.float32),
                x_next, y_next, robot_speed=self.robot.max_speed  # match your Robot's actual speed constant
            )
        if h_avoid_mode:
            min_idx = np.unravel_index(np.argmin(repulsive), repulsive.shape)
            best_dy = (min_idx[0] - center) * resolution
            best_dx = (min_idx[1] - center) * resolution
            norm = np.hypot(best_dx, best_dy) + 1e-8
            return self._apply_safety_check(
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
            
            return self._apply_safety_check(
                np.array([best_dx / norm, best_dy / norm], dtype=np.float32),
                x_next, y_next, robot_speed=self.robot.max_speed  # match your Robot's actual speed constant
            )
        
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

    
    def _future_prediction(self, action, x_next, y_next):
        # Prediction future movement and reward
        reward = 0.0
        self.tot_soc_pen = 0.0
        # grid to real robot pos
        # find which has greatest impact on valence and intensity and ues the corresponding observation

        for i in range(len(x_next)):
            dist = np.sqrt((x_next - self.robot.pos[0]) ** 2 + (y_next - self.robot.pos[1]) ** 2)
            # correlation:
            valence = self.correlation[0][4] * dist
            if valence < 0.0:
                intensity = self.correlation[0][4] * dist
                reward += valence * intensity * 5.0
                self.tot_soc_pen += valence * intensity * 5.0


# ----- Matplotlib Visualiseation -----

def get_proxemic_polygon(pos_x, pos_y, orientation, sigma_front, sigma_side, num_points=100):
    if orientation is None:
        orientation = 0.0
    # 1. Sample angles around a circle (relative to agent's heading)
    gamma = np.linspace(0, 2 * np.pi, num_points)
    
    denom = (np.cos(gamma)**2 / (sigma_front**2)) + (np.sin(gamma)**2 / (sigma_side**2))
    r = 1.0 / np.sqrt(denom)
    
    local_x = r * np.cos(gamma)
    local_y = r * np.sin(gamma)
    
    world_x = pos_x + (local_x * np.cos(orientation) - local_y * np.sin(orientation))
    world_y = pos_y + (local_x * np.sin(orientation) + local_y * np.cos(orientation))
    
    # 5. Stack into an array of (x, y) coordinates for the polygon
    polygon_points = np.column_stack((world_x, world_y))
    return polygon_points

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

def draw(env, ax):
    print(f"drawing: {len(env.spawner.humans)} humans, robot at {env.robot.pos}")
    ax.clear()
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, color = 'white', linewidth = 0.2, alpha = 0.3)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))

    arrow_len = 0.5
    for agent in env.spawner.humans:
        x, y = agent.pos
        orientation = agent.orientation if agent.orientation is not None else 0.0
        ps_front, ps_side = agent.ps_front, agent.ps_side
        color = BEHAVIOUR_COLORS.get(agent.behaviour_type, 'lightgray')
        marker = PERSONALITY_MARKERS.get(agent.personality, 'o')
        poly_points = get_proxemic_polygon(x, y, orientation, ps_front, ps_side)

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