import gymnasium as gym
import gym_examples
from gym_examples.wrappers import RelativePosition
from gym_examples.wrappers import NormalizeObservation

env = gym.make('gym_examples/GridWorld-v0',render_mode = "human")
wrapped_env = NormalizeObservation(RelativePosition(env))
#print(wrapped_env.reset())     # E.g.  [-3  3], {}

obs = wrapped_env.reset()
while True:
	action = env.action_space.sample()
	obs, reward, done, truncated, info = wrapped_env.step(action)
	print(obs)
	if done:
		break
