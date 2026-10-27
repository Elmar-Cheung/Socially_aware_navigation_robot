
from collections import deque
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer


class EventWindowReplayBuffer(DictReplayBuffer):

    EVENT_REWARD = -0.3      # reward below this counts as a social mistake (perceivable)
    WINDOW = 30              # steps before the event that get copied (~3 s)
    EVENT_FRACTION = 0.25    # share of each batch drawn from the event buffer
    EVENT_SIZE = 20_000

    def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1, **kw):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, **kw)
        self.event = DictReplayBuffer(self.EVENT_SIZE, observation_space, action_space,
                                      device=device, n_envs=n_envs, **kw)
        self.recent = deque(maxlen=self.WINDOW)
        self.n_events = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        if not any(info.get("human_nearby", False) for info in infos):
            self.recent.clear()                       # encounter over: window no longer contiguous
            return
        copy = lambda o: {k: np.array(v, copy=True) for k, v in o.items()}
        item = (copy(obs), copy(next_obs), np.array(action, copy=True),
                np.array(reward, copy=True), np.array(done, copy=True), infos)
        super().add(*item)
        self.recent.append(item)
        if float(np.min(reward)) < self.EVENT_REWARD:
            for t in self.recent:                     # the approach that led to the mistake
                self.event.add(*t)
            self.recent.clear()
            self.n_events += 1
        if np.any(done):
            self.recent.clear()

    def sample(self, batch_size, env=None):
        n_ev = int(batch_size * self.EVENT_FRACTION) if self.event.size() >= 256 else 0
        main = super().sample(batch_size - n_ev, env=env)
        if n_ev == 0:
            return main
        ev = self.event.sample(n_ev, env=env)
        merge = lambda a, b: ({k: th.cat([a[k], b[k]]) for k in a} if isinstance(a, dict)
                              else None if a is None else th.cat([a, b]))
        return type(main)(*[merge(a, b) for a, b in zip(main, ev)])