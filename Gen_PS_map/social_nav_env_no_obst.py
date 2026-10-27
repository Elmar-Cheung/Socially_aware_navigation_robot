from random import seed

import numpy as np
import gymnasium as gym
#from base_env_no_obst import BaseNavEnv, HumanOnlyReplayBuffer
from base_env_no_obst_test import BaseNavEnv, HumanOnlyReplayBuffer
from perception import perceive_emotion


class SocialNavEnvNoObst(BaseNavEnv):
    DT = 0.1
    OBS_WIDTH = 8          # [rel_x, rel_y, rel_orientation, speed, dist, valence, intensity, id]
    ID_COL = 7
    ACTION_WIDTH = 4       # [front, left, right, back]

    # Penalty per robot-caused reaction detected from perception.
    REACTION_PENALTY = {'stop': 0.4, 'sidestep': 0.2, 'jolt': 1.2, 'freeze': 1.0}
    EMOTION_WEIGHT = 3.0

    ONSET_WEIGHT = 5.0    # cost of making things worse (sums to ~peak per event)
    LEVEL_WEIGHT = 0.4    # small ongoing cost while the human is upset
    DISRUPTION_WEIGHT = 0.05

    RESIDUAL = False
    PRIOR = np.array([1.5, 0.8, 0.8, 1.0], dtype=np.float32)   # front, left, right, back
    RES_SCALE = 0.6

    def __init__(self, width=None, height=None, num_humans=1, emotion_noise=False, residual=False):
        self.emotion_noise = emotion_noise
        self.RESIDUAL = residual
        self.prev_valence = self.prev_intensity = 0.0
        super().__init__(width, height, num_humans)
        if self.RESIDUAL:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.TOP_N, self.ACTION_WIDTH),
                                               dtype=np.float32)

    def _get_obs(self):
        agents = np.zeros((self.TOP_N, self.OBS_WIDTH), dtype=np.float32)
        for i, rel_x, rel_y, rel_ori, speed, dist, valence, intensity, hid in self._observed_humans():
            if self.emotion_noise:
                valence, intensity = perceive_emotion(valence, intensity, rel_ori, self.np_random)
            agents[i] = [rel_x, rel_y, rel_ori, speed, dist, valence, intensity, hid]
        return self._make_obs(agents)

    def _ps_sizes(self, action):
        if self.RESIDUAL:
            return np.clip(self.PRIOR + self.RES_SCALE * action, 0.05, 2.0)
        return action

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_neg = self._neg_arousal()
        self._prev_ids = self._obs['agents'][:, self.ID_COL].copy()
        return obs, info

    def _neg_arousal(self):
        a = self._obs['agents']
        return np.maximum(-a[:, 5], 0.0) * a[:, 6]          # >= 0 per observed row

    def _get_reward(self, goal, reactions):
        parts = self._task_reward(goal)                       # time, safety, and progress shaping in parts['shaping']

        # Emotion: onset (rise in negative arousal) + small level term
        neg = self._neg_arousal()
        ids = self._obs['agents'][:, self.ID_COL]
        prev = np.where(ids == self._prev_ids, self._prev_neg, 0.0)
        rise = np.maximum(neg - prev, 0.0)
        emo = -(self.ONSET_WEIGHT * rise.sum() + self.LEVEL_WEIGHT * neg.sum())
        parts['soc_pen'] = float(np.clip(emo, -3.0, 0.0))

        # Reactions: once, blame-weighted, in their own term
        robot_vel = self.robot.measured_vel(self.dt)
        detections = self.robot.detect(self.humans)
        for hid, event in reactions.items():
            rel = detections.get(hid)
            closing = 0.0 if rel is None else float(robot_vel @ (rel / (np.linalg.norm(rel) + 1e-6)))
            blame = float(np.clip(0.2 + closing / self.robot.MAX_SPEED, 0.2, 1.0))
            parts['human_reaction'] -= self.REACTION_PENALTY[event] * blame

        # Emotion potential in its own key, so the progress shaping survives
        phi = self._potential()
        parts['emo_shaping'] = self.GAMMA * phi - self._prev_phi
        self._prev_phi = phi

        self._prev_neg, self._prev_ids = neg, ids.copy()
        return sum(parts.values()), parts
    """
    def _get_reward(self, goal, reactions):
        
        parts = self._task_reward(goal)
        valence, intensity = self._obs['agents'][:, 5], self._obs['agents'][:, 6]
        valence_diff = valence - self.prev_valence
        parts['soc_pen'] = 0.0
        if valence_diff < 0:
            parts['soc_pen'] = float(np.sum(np.clip(min(valence, 0) * intensity * self.EMOTION_WEIGHT, -1.0, 1.0)))
        parts['soc_pen'] -= sum(self.REACTION_PENALTY[e] for e in reactions.values())
        parts['soc_pen'] = np.clip(parts['soc_pen'], -1.0, 1.0)
        disrupt = sum(min(d, 1.0) for hid, d in self.detector.disruption.items() if hid not in reactions)
        parts['soc_pen'] -= disrupt * self.DISRUPTION_WEIGHT
        phi = self._potential()
        parts['shaping'] = self.GAMMA * phi - self._prev_phi
        self._prev_phi = phi
        if valence <= 0:
            self.prev_valence, self.prev_intensity = valence, intensity
        return sum(parts.values()), parts
    """
    def _potential(self):
        """Shaping potential: minus the total observed emotional arousal."""
        a = self._obs['agents']
        return -float(np.sum(np.abs(a[:, 5]) * a[:, 6]))


# Old class name, so existing scripts keep working:
# ``from social_nav_env_no_obst import SocialNavEnv``
SocialNavEnv = SocialNavEnvNoObst

from render import draw  # noqa: E402,F401