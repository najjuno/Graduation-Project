import gymnasium as gym
import gym_examples
from grid_world_creator import grid_world_creator
import ray
from ray import tune
from ray.tune.registry import register_env
import os
from ray import rllib

register_env("GridWorld_env",grid_world_creator)
full_path = os.path.abspath("experiment_result")

if __name__ == "__main__":

	ray.init()

	env = gym.make('gym_examples/GridWorld-v0',render_mode = "human")
	rllib.utils.check_env(env)


	tune.run("PPO",
			config={"env" : "GridWorld_env",
					'''"env_config":{
						"obs_filter": "my_normalized_relative_position",
						"reward_filter": "clip_reward"
					},'''
					"evaluation_interval": 20,
					"evaluation_num_episodes": 100,
					},
			checkpoint_freq=10000,
			local_dir = full_path
			)
