import gymnasium as gym
import gym_examples
from gym_examples.wrappers import RelativePosition
from gym_examples.wrappers import NormalizeObservation
from gym_examples.wrappers import ClipReward
from gym_examples.wrappers import ReacherRewardWrapper

def grid_world_creator(config):

	obs_filter = config.pop("obs_filter",None)
	reward_filter = config.pop("reward_filter",None)

	env = gym.make('gym_examples/GridWorld-v0',render_mode = "human")
	if obs_filter is not None:
		if obs_filter == "my_normalized_relative_position":
			env = NormalizeObservation(RelativePosition(env))
		'''elif obs_filter == "my_normalized_relative_position":
			env = NormalizeObservation(RelativePosition(env))'''
	if reward_filter is not None:
		if reward_filter == "clip_reward":
			env = ClipReward(env,0,10000)	
		elif reward_filter == "reacher_weighted_reward":
			env = ReacherRewardWrapper(env) 		
	return env
