import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from utils.telos_joints import TelosJoints


class CustomEnvironment(gym.Env):
    def __init__(self):
        super(CustomEnvironment, self).__init__()

        # Initialize PyBullet and load URDF files
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Additional initialization
        self.seed()

    def step(self, action):
        # Apply action, update simulation, and get new observation
        # Calculate reward and check if episode is done
        return observation, reward, done, info

    def reset(self):
        # Reset environment to initial state
        return initial_observation

    def render(self, mode="human"):
        # Render the environment
        pass

    def close(self):
        # Clean up resources
        p.disconnect(self.client)

    def seed(self, seed=None):
        # Seed the environment for reproducibility
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
