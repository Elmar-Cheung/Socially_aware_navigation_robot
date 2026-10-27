"""Emotion-based navigation: the policy observes the nearest human's position,
heading, speed and (camera-estimated) emotion, and predicts its asymmetric
personal space [front, left, right, back]. The reward penalises negative
emotion and robot-caused reactions (stop / sidestep / startle)."""
import numpy as np
 
from base_env import BaseNavEnv, HumanOnlyReplayBuffer  # noqa: F401 (re-export)
from perception import perceive_emotion
 
 
class SocialNavEnv(BaseNavEnv):
    DT = 0.1
    OBS_WIDTH = 8          # [rel_x, rel_y, rel_orientation, speed, dist, valence, intensity, id]
    ID_COL = 7
    ACTION_WIDTH = 4       # [front, left, right, back]
 
    # Penalty per robot-caused reaction detected from perception.
    REACTION_PENALTY = {'stop': 0.4, 'sidestep': 0.3, 'jolt': 1.0, 'freeze': 0.8}
    EMOTION_WEIGHT = 1.5

    ONSET_WEIGHT = 5.0    # cost of making things worse (sums to ~peak per event)
    LEVEL_WEIGHT = 0.4    # small ongoing cost while the human is upset
    DISRUPTION_WEIGHT = 0.05
 
    def __init__(self, width=7, height=4, num_humans=1, emotion_noise=False):
        self.emotion_noise = emotion_noise
        self.prev_valence = self.prev_intensity = 0.0
        super().__init__(width, height, num_humans)
 
    def _get_obs(self):
        agents = np.zeros((self.TOP_N, self.OBS_WIDTH), dtype=np.float32)
        for i, rel_x, rel_y, rel_ori, speed, dist, valence, intensity, hid in self._observed_humans():
            if self.emotion_noise:
                valence, intensity = perceive_emotion(valence, intensity, rel_ori, self.np_random)
            agents[i] = [rel_x, rel_y, rel_ori, speed, dist, valence, intensity, hid]
        return {'agents': agents, 'obstacles': self._obstacle_obs()}
 
    def _ps_sizes(self, action):
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
 
    def _potential(self):
        """Shaping potential: minus the total observed emotional arousal."""
        a = self._obs['agents']
        return -float(np.sum(np.abs(a[:, 5]) * a[:, 6]))
 
 
from render import draw  # noqa: E402,F401  (``from social_nav_env import draw``)