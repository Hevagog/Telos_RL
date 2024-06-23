import numpy as np
import pybullet as p
import gymnasium as gym

from typing import Optional
import utils.telos_joints as tj


class TelosTaskEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task,
        agent,
        render_mode: str = "rgb_array",
    ) -> None:
        self.task = task
        self.agent = agent
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )
        self.low_angles = np.array(
            [
                tj.HIP_MIN_ANGLE,
                tj.THIGH_MIN_ANGLE,
                tj.KNEE_MIN_ANGLE,
            ]
            * 4
        )
        self.high_angles = np.array(
            [
                tj.HIP_MAX_ANGLE,
                tj.THIGH_MAX_ANGLE,
                tj.KNEE_MAX_ANGLE,
            ]
            * 4
        )
        self.action_space = gym.spaces.Box(
            low=self.low_angles, high=self.high_angles, shape=(12,), dtype=np.float32
        )

    def _get_obs(self):
        return self.agent._get_obs()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.task.reset()
        info = {
            "is_success": self.task.is_success(self._get_obs()[0:3], self.task.goal)
        }
        return self._get_obs(), info

    def step(self, action):
        self.agent.set_action(action)
        self.agent.step_simulation()
        obs = self._get_obs()
        reward = self.task.compute_reward(obs[0:3], self.task.goal)
        truncated = False
        done = bool(self.task.is_success(obs[0:3], self.task.goal))
        info = {"is_success": done}
        return obs, reward, done, truncated, info

    def close(self):
        p.disconnect()

    # def render(self, mode="human"):
