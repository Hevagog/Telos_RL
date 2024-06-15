import os
import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot


class TelosEnv(PyBulletRobot):
    _current_dir = os.path.dirname(os.path.realpath(__file__))
    _urdf_root_path = _current_dir + "/../urdf"
    _urdf_robot_path = _urdf_root_path + "/tt.urdf"

    def __init__(self, sim):
        action_dim = 12
        action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="telos",
            file_name=self._urdf_robot_path,
            base_position=[0, 0, 1],
            action_space=action_space,
            joint_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            joint_forces=[1.5] * 12,  # in Nm
        )

    def set_action(self, action):
        self.control_joints(action)

    def get_obs(self):
        angles = np.empty(12)
        for i in range(12):
            angles[i] = self.get_joint_angle(i)
        return np.array(angles)

    def reset(self) -> None:
        neutral_angles = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.set_joint_angles(neutral_angles)


##################################
from panda_gym.pybullet import PyBullet
from time import sleep

if __name__ == "__main__":
    sim = PyBullet(render_mode="human")
    robot = TelosEnv(sim)

    for _ in range(50):
        robot.set_action(np.array([0.5] * 12))
        sleep(0.2)
        sim.step()
