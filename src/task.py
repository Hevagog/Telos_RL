import numpy as np
import pybullet as p
import pybullet_data
import yaml
import math
import time


class TelosTask:
    def __init__(self, agent):
        with open("pybullet_config.yaml", "r", encoding="utf-8") as file:
            _config = yaml.safe_load(file)
        self.agent = agent
        self.goal_radius = _config["task"]["goal_radius"]
        self.fall_reward = _config["task"]["fall_reward"]
        self.dist_threshold = _config["task"]["distance_threshold"]
        self.fall_threshold = _config["task"]["fall_threshold"]
        self.smoothing_factor = _config["task"]["smoothing_factor"]
        self.angle_bounds = np.deg2rad([*_config["task"]["goal_angle_bounds"]])
        theta = np.random.uniform(*self.angle_bounds)
        self.goal = np.array(
            [self.goal_radius * np.cos(theta), self.goal_radius * np.sin(theta), 0]
        )

    def reset(self, seed=None):
        theta = np.random.uniform(*self.angle_bounds)
        self.goal = np.array(
            [self.goal_radius * np.cos(theta), self.goal_radius * np.sin(theta), 0]
        )

    def get_obs(self):
        return self.agent.get_obs()

    def get_achieved_goal(self):
        return self.agent.get_obs()

    def is_success(self, achieved_goal, desired_goal, info={}):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return np.array(distance < self.dist_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        healthy_reward = (
            0 if achieved_goal[2] > self.fall_threshold else self.fall_reward
        )
        distance_reward = -np.linalg.norm(achieved_goal - desired_goal)
        smoothing_reward = self.smoothing_factor * np.sum(
            self.agent.get_acceleration_from_rotary()
        )
        return np.array(
            healthy_reward + distance_reward + smoothing_reward, dtype=np.float32
        )
