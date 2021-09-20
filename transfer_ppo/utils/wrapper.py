import numpy as np
from collections import deque
import gym

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = env.observation_space
        self.observation_space.shape = (k*env.observation_space.shape[0],)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.asarray(self._get_ob()).astype(np.float32)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.asarray(self._get_ob()).astype(np.float32), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
    
### Keep Track of the running average and standard deviation of reward
### Return normalized reward
class RewardNormalize(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.n   = 0
    def step(self, a):
        next_o, r, d, info = self.env.step(a)
        self.r = r ### During roll-out, get the unnormalized reward from self.r
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m = r
            self.old_s = 0
        else:
            self.new_m = self.old_m + (r - self.old_m) / self.n
            self.new_s = self.old_s + (r - self.old_m) * (r - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s
        std = np.sqrt(self.new_s / (self.n - 1)) if self.n > 1 else 1.
        if self.n == 1 or std == 0.:
            r_normalized = r
        else:
            r_normalized = (r-self.new_m)/(std+1e-8) 
        return next_o, r_normalized, d, info