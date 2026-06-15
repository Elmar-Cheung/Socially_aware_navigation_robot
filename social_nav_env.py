import gymnasium as gym
from networkx import edges
import numpy as np
from human_agent import HumanAgent, collision_reaction, disturb_group, violate_PS
from robot import Robot

import numpy as np
import random
from collections import deque

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_spawn(self):
        # Force respawn if population drops too low
        if len(self.humans) < 15 and not self.spawn_queue:
            self.remaining += 10
            self._fill_spawn_queue()
        
        if len(self.humans) > 45:
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
    def __init__(self, x, y, radius=0.5):
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
        self.clipped_action = [0.0, 0.0]

        self.observation_space = gym.spaces.Dict({
            'agents': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(10, 7), dtype=np.float32
            ),
            'robot_state': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,), dtype=np.float32
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
    
    def _cast_lidar(self, num_rays=200, max_range=4.0):
        """
        Cast rays around robot.
        hit_type 0 is freespace
        hit_type 1 is obstacle/human
        """
        robot_pos = self.robot.pos
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        lidar = np.zeros((num_rays, 3), dtype=np.float32)

        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            hit_dist = max_range
            hit_type = 0

            # Check obstacles
            for obs in self.obstacles:
                d = self._ray_circle_intersect(robot_pos, direction, obs.pos, obs.radius)
                if d is not None and d < hit_dist:
                    hit_dist = d
                    hit_type = 1

            # Check humans
            for human in self.spawner.humans:
                d = self._ray_circle_intersect(robot_pos, direction, human.pos, radius=0.3)
                if d is not None and d < hit_dist:
                    hit_dist = d
                    hit_type = 1

            lidar[i] = [hit_dist, hit_type, angle]

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
    

    def _get_obs(self):
        obs = self.robot.get_observation(self.spawner.humans)
        agent_matrix = np.zeros((10, 7), dtype=np.float32)
        for i, human in enumerate(obs[:10]):
            rel_x = human[0] - self.robot.pos[0]
            rel_y = human[1] - self.robot.pos[1]
            agent_matrix[i] = [rel_x, rel_y, human[2], human[3], human[4], human[7],
                            np.sqrt(rel_x**2 + rel_y**2)]
        
        gx, gy = self.robot.goal if self.robot.goal is not None else self.robot.pos
        dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        inv_dist = 1.0 / (dist + 0.1)
        rel_goal_x = gx - self.robot.pos[0]
        rel_goal_y = gy - self.robot.pos[1]
        robot_state = np.array([
            rel_goal_x, rel_goal_y, inv_dist
        ], dtype=np.float32)
        
        return {'agents': agent_matrix, 'robot_state': robot_state}

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
            'social_reward': self.social_reward
        }

    def _get_reward(self, arrived, obs):
        self.move_reward = 0.0
        self.social_reward = 0.0
        self.tot_soc_pen = 0.0

        agent_matrix = obs['agents']
        self.safe_conduct = True
        in_social_zone = False
        total_social_penalty = 0.0
        reward = 0.0

        dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        if arrived:
            reward += 10.0
            return reward
        
        # Distance improvement reward
        if hasattr(self, '_prev_dist'):
            move = self._prev_dist - dist
            if dist < 0.5:
                scale = 6.0
            elif dist < 1.5:
                scale = 4.0
            elif dist < 2.7:
                scale = 2.5
            else:
                scale = 1.5

            if abs(move) < 0.01:
                reward -= 0.05
                self.move_reward -= 0.05
            else:
                reward += move * scale   # works for both + and - move
                self.move_reward += move * scale
        else:
            move = 0.0

        reward -= 0.01 *(dist / self._prev_dist)
        self.move_reward -= 0.01 *(dist / self._prev_dist)

        self._prev_dist = dist

        # Step penalty
        reward -= 0.001
        
        margin = 1.0
        rx, ry = self.robot.pos
        if (rx < margin or rx > self.width - margin or
            ry < margin or ry > self.height - margin):
            reward -= 0.1  # penalty for being near boundary

        # Social penalty
        for i in range(len(agent_matrix)):
            _, _, _, valence, intensity, _, _ = agent_matrix[i]
            if valence < 0.0:
                self.safe_conduct = False
                total_social_penalty += abs(valence) * intensity * 10.0

        if self.safe_conduct:
            for i in range(len(agent_matrix)):
                _, _, speed, _, _, _, agent_dist = agent_matrix[i]
                if speed == 0.0 and agent_dist == 0.0:
                    continue
                if agent_dist < 4:
                    in_social_zone = True
                    break

        if in_social_zone and self.safe_conduct and move > 0:
            reward += move * 2.0
            self.social_reward = move * 2.0

            # Social reward for safe approach

        reward -= total_social_penalty
        self.tot_soc_pen = total_social_penalty
        return reward

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.time_stamp = 0
        self.robot.assign_goal()
        self.previous_reward = 0.0
        
        self._prev_dist = np.linalg.norm(self.robot.pos - self.robot.goal)
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.spawner.try_spawn()
        arrived = self.robot.step(action, self.dt)
        
        self.robot.pos[0] = np.clip(self.robot.pos[0], 0.0, self.width)
        self.robot.pos[1] = np.clip(self.robot.pos[1], 0.0, self.height)

        for human in self.spawner.humans:
            human.step(self.dt, self.spawner.humans, self.robot)
            if human.arrived:
                self.spawner.check_group_arrived(human.group_id)
        
        obs    = self._get_obs()
        reward = self._get_reward(arrived, obs)

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