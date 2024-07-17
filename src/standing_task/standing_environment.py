import math
import numpy as np
import gymnasium as gym
from typing import Tuple

from typing import Optional
import utils.telos_joints as tj
from utils.helper import load_yaml
from utils.PyBullet import PyBullet


class StandingTelosTaskEnv(gym.Env):
    def __init__(self, task, agent, sim_engine: PyBullet) -> None:
        _config = load_yaml("pybullet_config.yaml")
        self.task = task
        self.agent = agent
        self.sim = sim_engine
        self.plane = self.sim.load_plane()
        self.plane_angle_bounds = math.radians(
            _config["standing_task"]["plane_angle_bounds"]
        )
        observation, _ = self.reset()
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(observation["agent"]),),
                    dtype=np.float32,
                )
            }
        )
        self.low_angles = tj.low_angles
        self.high_angles = tj.high_angles
        self.action_space = gym.spaces.Box(
            low=self.low_angles, high=self.high_angles, shape=(12,), dtype=np.float32
        )

    def reset_plane(self):
        theta = np.random.uniform(
            low=-self.plane_angle_bounds, high=self.plane_angle_bounds
        )
        self.orientation = self.sim.get_quaternion_from_euler([theta, 0, 0])
        self.sim.reset_base_pos(self.plane, [0, 0, 0], self.orientation)

    def _get_obs(self):
        return {"agent": self.agent.get_obs()}

    def _get_info(self):
        agent_pos = self._get_obs()["agent"][0:3]
        return agent_pos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.task.reset()
        self.agent.reset()
        self.reset_plane()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.agent.set_action(action)
        self.sim.step_simulation()
        obs = self._get_obs()
        reward = self.task.compute_reward(obs["agent"][0:3])
        done = self.task.is_terminated()
        info = self._get_info()
        return obs, reward, done, False, info

    def close(self):
        self.sim.close()

    def render(self):
        return


def make_env(task, agent, sim):
    return StandingTelosTaskEnv(task, agent, sim)
