import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

	def __init__(self, render_mode=None, size=8):
		
		self.size = size  # The size of the square grid
		self.window_size = 512  # The size of the PyGame window
		self.fire_impact = 10
		self.exit_impact = 5

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # Health is the character's remaining health.

		self.max_health = 5000
		obs_low = np.array([0,0,0,0,0])
		obs_high = np.array([size-1,size-1,size-1,size-1,self.max_health])
		#self.observation_space = spaces.Box(low= obs_low, high = obs_high)
		self.observation_space = spaces.Dict(
			{
				"agent": spaces.Box(low=np.array([0,0]), high=np.array([size-1,size - 1])),
				"target": spaces.Box(low=np.array([0,0]), high=np.array([size-1,size - 1])),
				"health": spaces.Box(low=np.array([0]), high=np.array([self.max_health]))
			}
		)

		# Define action space: bounds, space type, shape
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
		self.action_space = spaces.Discrete(4)

		"""
		The following dictionary maps abstract actions from `self.action_space` to 
		the direction we will walk in if that action is taken.
		I.e. 0 corresponds to "right", 1 to "up" etc.
		"""
		
		self._action_to_direction = {
			0: np.array([1, 0]),
			1: np.array([0, 1]),
			2: np.array([-1, 0]),
			3: np.array([0, -1]),
		}
		
		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		"""
		If human-rendering is used, `self.window` will be a reference
		to the window that we draw to. `self.clock` will be a clock that is used
		to ensure that the environment is rendered at the correct framerate in
		human-mode. They will remain `None` until human-mode is used for the
		first time.
		"""
		self.window = None
		self.clock = None

	def _get_obs(self):
		#return np.concatenate(np.array(self._agent_location), np.array(self._target_location))
		return {"agent": np.array(self._agent_location).astype(np.float32), "target": np.array(self._target_location).astype(np.float32), "health":np.array([self._agent_health]).astype(np.float32)}

	def _get_info(self):
		return {
			"distance": np.linalg.norm(
				self._agent_location - self._target_location, ord=1
			)
		}

	def reset(self, seed=None, options=None):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)

		self.steps = 0

		# Choose the agent's location uniformly at random
		self._agent_location = self.np_random.integers(0, self.size, size=2)
		
		# Choose the 2 fire's location at random, they do not overlap with the agent and escape door
		# We will sample the target's location randomly until it does not coincide with the agent's location
		side = np.random.choice(['top', 'bottom', 'left', 'right'])
		self._target_location = self._random_side_position(side)
		while np.array_equal(self._target_location, self._agent_location):
			self._target_location = self._random_side_position(side)

		# Choose the 2 fire's location at random, they do not overlap with the agent and escape door
		self.fire_locations = [self._random_position(), self._random_position()]
		while any(np.array_equal(self._agent_location, fire_pos) for fire_pos in self.fire_locations) or any(np.array_equal(self._target_location, fire_pos) for fire_pos in self.fire_locations):
			self.fire_locations = [self._random_position(), self._random_position()]
		
		self._agent_health = 5000

		observation = self._get_obs()
		info = self._get_info()
		
		if self.render_mode == "human":
			self._render_frame()

		return observation, info

	def _random_position(self):
		return self.np_random.integers(0, self.size, size=2, dtype=int)

	def _random_side_position(self, side):
		if side == 'top':
			return np.array((0, self.np_random.integers(low=0, high=self.size - 1)))
		elif side == 'bottom':
			return np.array((self.size - 1, self.np_random.integers(low=0, high=self.size - 1)))
		elif side == 'left':
			return np.array((self.np_random.integers(low=0, high=self.size - 1), 0))
		elif side == 'right':
			return np.array((self.np_random.integers(low=0, high=self.size - 1), self.size - 1))


	def step(self, action):
		
		self.steps +=1
		
		# Map the action (element of {0,1,2,3}) to the direction we walk in
		direction = self._action_to_direction[action]

		# We use `np.clip` to make sure we don't leave the grid
		self._agent_location = np.clip(
			self._agent_location + direction, 0, self.size - 1
		)
		
		distance_fire1 = np.linalg.norm(np.array(self._agent_location) - np.array(self.fire_locations[0]))
		if(distance_fire1<=self.steps):
			self._agent_health -= self.fire_impact * self.steps * (distance_fire1)

		distance_fire2 = np.linalg.norm(np.array(self._agent_location) - np.array(self.fire_locations[1]))
		if(distance_fire1<=self.steps):
			self._agent_health -= self.fire_impact * self.steps * (distance_fire2)

		self._agent_health = max(self._agent_health , 0)
		
		# Compute next obs
		next_obs = {"agent": self._agent_location, "target": self._target_location, "health":self._agent_health}

		# Compute reward
		distance_exit = np.linalg.norm(np.array(self._agent_location) - np.array(self._target_location))
		distance_fires = distance_fire1 + distance_fire2

		reward = self.exit_impact*(20 - distance_exit) - self.fire_impact * (20-distance_fires)

        # Compute done

		terminated = (self._agent_health<=0)
		if(np.array_equal(self._agent_location, self._target_location)): terminated = True
		if(np.array_equal(self._agent_location, self.fire_locations[0]) or np.array_equal(self._agent_location, self.fire_locations[0])): terminated = True

		reward += 1 if terminated else 0  # Binary sparse rewards
		observation = self._get_obs()
		info = self._get_info()

		if self.render_mode == "human":
			self._render_frame()

		return observation, reward, terminated,False, info

	def render(self):
		if self.render_mode == "rgb_array":\
			return self._render_frame()

	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((self.window_size, self.window_size))
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()

		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255))
		pix_square_size = (
			self.window_size / self.size
		)  # The size of a single grid square in pixels

		# First we draw the target
		pygame.draw.rect(
			canvas,
			(255, 0, 0),
			pygame.Rect(
				pix_square_size * self._target_location,
				(pix_square_size, pix_square_size),
			),
		)
		# Now we draw the agent
		pygame.draw.circle(
			canvas,
			(0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
		for x in range(self.size + 1):
			pygame.draw.line(
				canvas,
				0,
				(0, pix_square_size * x),
				(self.window_size, pix_square_size * x),
				width=3,
			)
		pygame.draw.line(
				canvas,
				0,
				(pix_square_size * x, 0),
				(pix_square_size * x, self.window_size),
				width=3,
			)

		if self.render_mode == "human":
		# The following line copies our drawings from `canvas` to the visible window
			self.window.blit(canvas, canvas.get_rect())
			pygame.event.pump()
			pygame.display.update()
			
			# We need to ensure that human-rendering occurs at the predefined framerate.
			# The following line will automatically add a delay to keep the framerate stable.
			self.clock.tick(self.metadata["render_fps"])
		else:  # rgb_array
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
			)

	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()
