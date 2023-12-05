import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space ={
			"RelativePosition": Box(low=np.array([-np.inf,-np.inf]), high=np.array([np.inf,np.inf])),
			"health": Box(low=np.array([0]), high=np.array([self.max_health]))
		}

    def observation(self, obs):
        return {
			"RelativePosition": obs["target"] - obs["agent"],
			"health": obs["health"]
		}
