"""
agent
"""

import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from utils.telos_joints import TelosJoints as tj, KNEE_ANGLE, THIGH_HIP_ANGLE, HIP_ANGLE


class TelosAgent:
    def __init__(self, set_gravity=True):
        """
        Initializes the quadruped agent.
        """
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        _urdf_root_path = _current_dir + "/urdf"
        _urdf_robot_path = _urdf_root_path + "/tt.urdf"

        # self.forces = np.full(12, tj.HIP_THIGH_FORCE)
        # self.forces[[3, 6, 9, 12]] = tj.KNEE_FORCE
        self.forces = np.zeros(16)

        self.default_angles = [0, HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE] * 4
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 60.0,
            solverResidualThreshold=1 - 10,
            numSolverIterations=50,
            numSubSteps=4,
        )
        if set_gravity:
            p.setGravity(0, 0, -9.8)
        self.start_pos = [0, 0, 0.5]
        self.cube_start_orientation = p.getQuaternionFromEuler(
            [1.5707963267948966, 0, 0]
        )
        self.agent = p.loadURDF(
            _urdf_robot_path, self.start_pos, self.cube_start_orientation
        )
        self.reset_angles()

    def reset_angles(self):
        default_angles = self.default_angles.copy()
        for joint in range(16):
            p.resetJointState(
                self.agent,
                joint,
                default_angles.pop(0),
            )

    def move_joint(self, leg_id, position, force=1.5):
        """
        Moves a specific leg of the quadruped robot.
        :param leg_id: ID of the leg joint to move.
        :param position: Target position for the leg.
        :param force: Force to apply for moving the leg.
        """
        p.setJointMotorControlArray(
            self.agent,
            [leg_id],
            p.POSITION_CONTROL,
            [position],
            [force],
        )

    def move_legs(self, joint_ids, positions, forces):
        """
        Moves all the legs of the quadruped robot.
        :param leg_id: ID of the leg joint to move.
        :param positions: Target positions for the legs.
        :param forces: Forces to apply for moving the legs.
        """
        p.setJointMotorControlArray(
            self.agent,
            joint_ids,
            p.POSITION_CONTROL,
            positions,
            forces,
        )

    def step_simulation(self):
        """
        Steps the simulation forward by one time step.
        """
        p.stepSimulation()

    def disconnect(self):
        """
        Disconnects from PyBullet.
        """
        p.disconnect(self.physics_client)


# Example usage
if __name__ == "__main__":
    quadruped_agent = TelosAgent(False)
    time.sleep(10)

    for _ in range(100):
        for angle in range(-45, 46):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            print(angles)
            quadruped_agent.move_legs(range(16), angles, quadruped_agent.forces)
            quadruped_agent.step_simulation()
            time.sleep(0.01)
        for angle in range(45, -46, -1):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            print(angles)
            quadruped_agent.move_legs(range(16), angles, quadruped_agent.forces)
            quadruped_agent.step_simulation()
            time.sleep(0.01)

        # Step simulation
        # Sleep for a short duration to observe the movement
        time.sleep(0.5)

    quadruped_agent.disconnect()
