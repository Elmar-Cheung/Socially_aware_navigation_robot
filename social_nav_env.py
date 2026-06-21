import gymnasium as gym
from networkx import edges
import numpy as np
from human_agent import HumanAgent, collision_reaction, disturb_group, violate_PS
from robot import Robot

import numpy as np
import random
from collections import deque
import scipy.ndimage as ndimage
import heapq

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

import affective_buffer as ab

global prioritise_replay
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
        # Force respawn if population drops too low
        if len(self.humans) < 15 and not self.spawn_queue:
            self.remaining += 10
            self._fill_spawn_queue()
        
        if len(self.humans) > 35:
            self.remaining -= 10
            self._fill_spawn_queue()

        while self.spawn_queue:
            group_size = self.spawn_queue[0]
            positions = self._find_free_edge_positions(group_size)
            if positions is None:
                break
            self.spawn_queue.popleft()
            self._spawn_group(positions, group_size)

    def check_group_arrived(self, group_id):
        """
        Check if all m,embers arrive, if yes, fill the spawbn queue
        """
        members = self.group_members.get(group_id, [])
        if not members:
            return
        if all(m.arrived for m in members):
            self.group_arrived[group_id] = True
            for member in members:
                if member in self.humans:
                    self.humans.remove(member)
            del self.group_members[group_id]
            del self.group_goals[group_id]

            # Spawn replacement humans
            num_new = random.randint(len(members) - 3, len(members) + 3)
            self.remaining += num_new
            self._fill_spawn_queue()

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
            if is_group and self.remaining > 1:
                group_size = random.randint(2, min(5, self.remaining))
            else:
                group_size = 1
            self.spawn_queue.append(group_size)
            self.remaining -= group_size


    def _find_free_edge_positions(self, group_size, min_separation=1.5):
        """
        Finds position for groups to be spawned
        """
        margin = 0.5  # distance from actual wall boundary

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
        margin = 1.0

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
    def __init__(self, width=20, height=20, num_humans=40):
        self.obstacle = [
            Obstacle(8,  8,  0.75),
            Obstacle(15, 10, 0.75),
            Obstacle(5,  20, 0.5),
            Obstacle(20, 5,  0.5),
            Obstacle(20, 20, 1.0)
        ]

        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.dt = 0.1  # simulation timestep
        self.safe_conduct = False
        self.tot_soc_pen = 0.0
        self.move_reward = 0.0
        self.social_reward = 0.0
        self.collision_reward = 0.0
        self.clipped_action = [0.0, 0.0]
        self.prev_obs = None
        self.field = None
        self.trajectory = None
        self.prev_field = None
        self.prev_trajectory = None
        self.human_states = np.zeros((10, 7), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            'agents': gym.spaces.Box(
                low=-50, high=50,
                shape=(10, 5), dtype=np.float32
            ),
            'field': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7, 7), dtype=np.float32
            )
        })

        self.robot = Robot(self.width, self.height)
        self.spawner = GroupSpawner(self.width, self.height, self.num_humans)
        for _ in range(10):
            self.spawner.try_spawn()
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        self.previous_reward = 0.0
        # Small action space - velocity commands
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),   # vx, vy
            high=np.array([1.0,  1.0]),
            dtype=np.float32
        )

    def _cast_lidar(self, num_rays=100, max_range=7.0):
        robot_pos = self.robot.pos
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        # 1. Vectorize ray directions shape: (num_rays, 2)
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        # Initialize all distances to max range
        hit_distances = np.full(num_rays, max_range, dtype=np.float32)
        
        # 2. Extract all obstacle positions and radiuses into arrays
        if len(self.obstacle) > 0:
            obs_positions = np.array([obs.pos for obs in self.obstacle], dtype=np.float32) # (N, 2)
            obs_radii = np.array([obs.radius for obs in self.obstacle], dtype=np.float32)     # (N,)
            
            # Vectorized Ray-Circle Intersection for ALL rays and ALL obstacles at once
            # oc shape: (1, N, 2) | ray_dirs shape: (num_rays, 1, 2)
            oc = robot_pos - obs_positions[np.newaxis, :, :] 
            dirs = ray_dirs[:, np.newaxis, :]
            
            a = 1.0 # np.dot(ray_dir, ray_dir) is always 1.0 since cos^2 + sin^2 = 1
            b = 2.0 * np.sum(oc * dirs, axis=2) # shape: (num_rays, N)
            c = np.sum(oc * oc, axis=2) - obs_radii**2 # shape: (num_rays, N)
            
            discriminant = b**2 - 4.0 * a * c # shape: (num_rays, N)
            
            # Mask where intersection happens
            valid_mask = discriminant >= 0
            sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
            
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0
            
            # Check valid distances (> 0.01)
            t1_valid = np.where((t1 > 0.01) & valid_mask, t1, max_range)
            t2_valid = np.where((t2 > 0.01) & valid_mask, t2, max_range)
            min_t = np.minimum(t1_valid, t2_valid)
            
            # Get the absolute closest hit for each individual ray across all obstacles
            closest_hit = np.min(min_t, axis=1)
            hit_distances = np.minimum(hit_distances, closest_hit)

        # Return shape (num_rays, 2) matching your original format
        lidar = np.stack([hit_distances, angles], axis=1)
        return lidar
    """
    def _cast_lidar(self, num_rays=100, max_range=7.0):

        Cast rays around robot.

        robot_pos = self.robot.pos
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        lidar = np.zeros((num_rays, 2), dtype=np.float32)

        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            hit_dist = max_range

            for obs in self.obstacle:
                d = self._ray_dist(robot_pos, direction, obs.pos, obs.radius)
                if d is not None and d < hit_dist:
                    hit_dist = d

            lidar[i] = [hit_dist, angle]

        return lidar

    def _ray_dist(self, ray_origin, ray_dir, circle_center, radius):
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None  # no intersection

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        if t1 > 0.01:
            return t1
        if t2 > 0.01:
            return t2
        return None
    """

    def _get_costmap(self, lidar, map_size=1.0, resolution=0.05, robot_radius=0.3):
        L_dist = lidar[:, 0]
        L_angle = lidar[:, 1]

        grid_cells = int(map_size / resolution)
        costmap = np.zeros((grid_cells, grid_cells), dtype=np.uint8)
        center = grid_cells // 2

        obstacle_map = L_dist < 7.0
        d_obs = L_dist[obstacle_map]
        a_obs = L_angle[obstacle_map]

        if not np.any(obstacle_map):
            return costmap
        
        x_local = d_obs * np.cos(a_obs)
        y_local = d_obs * np.sin(a_obs)

        x_grid = (x_local / resolution).astype(np.int32) + center
        y_grid = (y_local / resolution).astype(np.int32) + center

        valid_bound = (
            (x_grid >= 0)
            & (x_grid < grid_cells)
            & (y_grid >= 0)
            & (y_grid < grid_cells)
        )
        x_grid = x_grid[valid_bound]
        y_grid = y_grid[valid_bound]

        costmap[y_grid, x_grid] = 255

        pixel_dist = ndimage.distance_transform_edt(costmap != 255)
        pixel_radius = robot_radius / resolution

        inflation_mask = (pixel_dist > 0) & (pixel_dist <= pixel_radius)
        gradual_cost = 128 * (1.0 - (pixel_dist[inflation_mask] / pixel_radius))

        costmap[inflation_mask] = gradual_cost.astype(np.uint8)
        costmap[y_grid, x_grid] = 255

        return costmap
        
    
    def dijkstre_field(self, costmap, rel_goal_pos, penalty_mult=0.5, resolution = 0.05):
        new_cost = self.crop_field(costmap, 7)
        height, width = new_cost.shape
        start = (width // 2, height // 2)
        center = width // 2

        x, y = rel_goal_pos[0], rel_goal_pos[1]
        half_extent_m = center * resolution        # physical half-size of the 7x7 window
        scale = max(abs(x), abs(y), 1e-6)
        t = min(1.0, half_extent_m / scale)

        sub_x, sub_y = x * t, y * t
        x_g = int(round(sub_x / resolution)) + center
        y_g = int(round(sub_y / resolution)) + center

        grid_cell_max = width - 1
        x_g = max(0, min(x_g, grid_cell_max))
        y_g = max(0, min(y_g, grid_cell_max))
        goal = (x_g, y_g)

        if not (0 <= goal[0] < width and 0 <= goal[1] < height):
            return None, None
        
        if new_cost[goal[1], goal[0]] == 255:
            found_alternative = False
            for dx, dy in [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]:
                nx, ny = goal[0] + dx, goal[1] + dy
                if 0 <= nx < width and 0 <= ny < height and new_cost[ny, nx] < 255:
                    goal = (nx, ny)
                    found_alternative = True
                    break
            if not found_alternative:
                return None, None

        directions = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]

        priority_queue = [(0.0, goal)]
        g_score = {goal: 0.0}
        trajectory = {}
        
        while priority_queue:
            g, current = heapq.heappop(priority_queue)

            if g > g_score.get(current, np.inf):
                continue
            x, y = current
            current_cost = new_cost[y, x]

            for dx, dy, step_cost in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if new_cost[ny, nx] == 255:
                    continue

                tentative_g_score = g + step_cost + (current_cost * penalty_mult)
                neighbour = (nx, ny)

                if tentative_g_score < g_score.get(neighbour, np.inf):
                    g_score[neighbour] = tentative_g_score
                    trajectory[neighbour] = current
                    heapq.heappush(priority_queue, (tentative_g_score, neighbour))

        field = np.full((height, width), np.inf, dtype=np.float32)
        for (cx, cy), val in g_score.items():
            field[cy, cx] = val

        return field, trajectory

    def crop_field(self, field, patch_size, cap=100):
        center = field.shape[0] // 2
        half = patch_size // 2
        patch = field[center-half: center+half+1, center-half: center+half+1].copy()
        patch = np.clip(patch, -cap, cap)
        patch -= patch[half, half]

        return patch.astype(np.float32)
    
    def _get_obs(self):
        self.prev_field = self.field
        self.prev_trajectory = self.trajectory
        obs = self.robot.get_observation(self.spawner.humans)
        agent_matrix = np.zeros((10, 5), dtype=np.float32)
        for i, human in enumerate(obs[:10]):
            rel_x = human[0] - self.robot.pos[0]
            rel_y = human[1] - self.robot.pos[1]
            self.human_states[i] = [rel_x, rel_y, human[2], human[3], human[4], human[7],
                            np.sqrt(rel_x**2 + rel_y**2)]
            agent_matrix[i] = [rel_x, rel_y, human[2], human[7],
                            np.sqrt(rel_x**2 + rel_y**2)]
        gx, gy = self.robot.goal if self.robot.goal is not None else self.robot.pos
        dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        inv_dist = 1.0 / (dist + 0.1)
        rel_goal_x = gx - self.robot.pos[0]
        rel_goal_y = gy - self.robot.pos[1]

        lidar = self._cast_lidar()
        rel_goal = (rel_goal_x, rel_goal_y)
        costmap = self._get_costmap(lidar)
        self.field, self.trajectory = self.dijkstre_field(costmap, rel_goal)
        if self.field is None:
            field_obs = np.full((7, 7), 100.0, dtype=np.float32)
        else:
            field_obs = np.nan_to_num(self.field, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)
        self.field = field_obs
        return {'agents': agent_matrix, 'field': field_obs}

    def _get_info(self):
        gx, gy = self.robot.goal if self.robot.goal is not None else self.robot.pos
        goal = self.robot.goal if self.robot.goal is not None else self.robot.pos
        return {
            'robot_pos':     tuple(self.robot.pos),
            'goal_pos':      (gx, gy),
            'dist_to_goal':  float(np.linalg.norm(self.robot.pos - goal)),
            'time_used':     self.robot.time_stamp,
            'tot_soc_pen':   self.tot_soc_pen,
            'move_reward':   self.move_reward,
            'social_reward': self.social_reward,
            'collision_reward': self.collision_reward
        }

    def _get_reward(self, arrived, obs, action):
        self.move_reward = 0.0
        self.social_reward = 0.0
        self.tot_soc_pen = 0.0
        self.collision_reward = 0.0

        self.safe_conduct = True
        in_social_zone = False
        total_social_penalty = 0.0
        reward = 0.0

        dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        if arrived:
            reward += 10.0
            return reward
            
        # Distance improvement reward
        resolution = 0.05
        if self.prev_field is not None:
            center = self.prev_field.shape[0] // 2
            dx = action[0] * self.dt
            dy = action[1] * self.dt
            move_x = int(round(dx / resolution)) + center
            move_y = int(round(dy / resolution)) + center
            move_x = np.clip(move_x, 0, self.prev_field.shape[1] - 1)
            move_y = np.clip(move_y, 0, self.prev_field.shape[0] - 1)

            cost = self.prev_field[move_y, move_x]
            if not np.isfinite(cost):
                cost = 50.0

        # reward -= cost / 2
        # self.move_reward -= cost / 2

        if self.prev_obs is not None:
            f = self.prev_obs['field']
            field = self.crop_field(f, 5)
            rank_matrix = np.argsort(np.argsort((-field).ravel())).reshape(field.shape)

            move_x = int(round(dx / resolution)) + 2
            move_y = int(round(dy / resolution)) + 2    
            if 0 <= move_y < 5 and 0 <= move_x < 5:
                reward += (rank_matrix[move_y][move_x] - 12) / 2
                self.move_reward += (rank_matrix[move_y][move_x] - 12) / 2
            else:
                print("Out of bound")

        # Step penalty
        reward -= 0.001
        
        margin = 1.0
        rx, ry = self.robot.pos
        if (rx < margin or rx > self.width - margin or
            ry < margin or ry > self.height - margin):
            reward -= 0.1  # penalty for being near boundary

        # Social penalty
        for i in range(len(self.human_states)):
            _, _, _, valence, intensity, _, _ = self.human_states[i]
            if valence < 0.0:
                self.safe_conduct = False
                total_social_penalty += abs(valence) * intensity * 10.0

        if self.safe_conduct:
            for i in range(len(self.human_states)):
                _, _, speed, _, _, _, agent_dist = self.human_states[i]
                if speed == 0.0 and agent_dist == 0.0:
                    continue
                if agent_dist < 4:
                    in_social_zone = True
                    break

        if in_social_zone and self.safe_conduct and cost < 0:
            reward += cost * 2.0
            self.social_reward = cost * 2.0

            # Social reward for safe approach

        for o in  self.obstacle:
            dist = np.linalg.norm(self.robot.pos - o.pos)
            if dist <= (0.3 + o.radius):
                # Collide with obstacle
                reward -= 5.0
                self.collision_reward -= 5.0

        reward -= total_social_penalty
        self.tot_soc_pen = total_social_penalty
        return reward

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.time_stamp = 0
        self.robot.assign_goal()
        check = True
        while check:
            check = False
            for o in self.obstacle:
                dist = np.linalg.norm(self.robot.goal - o.pos)
                if dist < o.radius:
                    self.robot.assign_goal()
                    check = True
                    break

        self.previous_reward = 0.0
        
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.spawner.try_spawn()
        arrived = self.robot.step(action, self.dt)
        
        self.robot.pos[0] = np.clip(self.robot.pos[0], 0.0, self.width)
        self.robot.pos[1] = np.clip(self.robot.pos[1], 0.0, self.height)

        self.prev_obs = self._get_obs()
        for human in self.spawner.humans:
            human.step(self.dt, self.spawner.humans, self.robot)
            if human.arrived:
                self.spawner.check_group_arrived(human.group_id)
        
        obs    = self._get_obs()
        reward = self._get_reward(arrived, obs, action)

        ab.collect(
            obs = obs,
            action = action,
            humans = self.spawner.humans,
            timestep = self.robot.time_stamp
        )

        terminate = arrived
        
        return obs, reward, terminate, False, self._get_info()
        
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