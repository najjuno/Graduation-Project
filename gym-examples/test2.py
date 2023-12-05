import gymnasium as gym
import gym_examples

env = gym.make('gym_examples/GridWorld-v0',render_mode = "human")

print("Observation Space:", env.observation_space)
