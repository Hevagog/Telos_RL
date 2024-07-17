import math
import time
import numpy as np

from utils.helper import load_yaml


class StandingTelosTask:
    def __init__(self, agent, sim_engine):
        _config = load_yaml("pybullet_config.yaml")
        self.agent = agent
        self.sim = sim_engine
        self.pitch_bias = _config["task"]["pitch_bias"]
        self.fall_reward = _config["task"]["fall_reward"]
        self.fall_threshold = _config["task"]["fall_threshold"]
        self.dist_threshold = _config["task"]["distance_threshold"]
        self.smoothing_factor = _config["task"]["smoothing_factor"]
        self.max_angle_dip = _config["standing_task"]["max_angle_dip"]
        self.time_emphasis = _config["standing_task"]["time_emphasis"]
        self.angle_bounds = np.deg2rad([*_config["task"]["goal_angle_bounds"]])
        self.time_threshold = _config["standing_task"]["time_threshold"]
        self.agent_start_pos = np.array(
            [*_config["pybullet"]["robot"]["start_orientation"]]
        )
        self.start_time = time.time()

    def reset(self, seed=None):
        self.start_time = time.time()

    def get_obs(self):
        return self.agent.get_obs()

    def get_episode_time(self) -> float:
        return time.time() - self.start_time

    def is_terminated(self) -> bool:
        is_terminated = False
        is_terminated = (
            abs(self.sim.get_pitch_angle(self.agent.robot_agent)) > self.max_angle_dip
        )
        is_terminated = (
            abs(self.sim.get_roll_angle(self.agent.robot_agent)) > self.max_angle_dip
        )
        is_terminated = (
            abs(self.sim.get_yaw_angle(self.agent.robot_agent)) > self.max_angle_dip
        )
        is_terminated = self.get_episode_time() > self.time_threshold
        is_terminated = self.agent.get_obs()[2] < self.fall_threshold

        return is_terminated

    def compute_reward(self, achieved_goal, info={}) -> float:
        healthy_reward = (
            0 if achieved_goal[2] > self.fall_threshold else self.fall_reward
        )
        smoothing_reward = -self.smoothing_factor * np.sum(
            self.sim.get_acceleration_from_rotary(self.agent.robot_agent)
        )
        pitch_reward = -self.pitch_bias * math.pow(
            self.sim.get_pitch_angle(self.agent.robot_agent), 2
        )
        time_reward = self.time_emphasis * self.get_episode_time()
        distance_reward = -np.linalg.norm(
            self.agent.get_obs()[:3] - self.agent_start_pos
        )
        return (
            healthy_reward
            + smoothing_reward
            + pitch_reward
            + time_reward
            + distance_reward
        )
