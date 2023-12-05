import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class NormalizeObservation(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.observation_spce ={
			"RelativePosition": Box(low = np.array([-1]), high = np.array([1])),
			"health": Box(low=np.array([0]), high=np.array([1]))
		}

	def observation(self, obs):

		return {
			"RelativePosition" : obs["RelativePosition"] / np.array([self.env.size*2,self.env.size*2]),
			"health" : obs["health"]  / np.array([1,self.env.max_health])
		} 

