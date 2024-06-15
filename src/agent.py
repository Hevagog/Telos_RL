"""
agent
"""

import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from utils.telos_joints import KNEE_ANGLE, THIGH_HIP_ANGLE, HIP_ANGLE


class TelosAgent:
    def __init__(
        self,
        render_mode: str = "rgb_array",
        set_gravity: bool = True,
        renderer: str = "Tiny",
    ) -> None:
        """
        Initializes the quadruped agent.
        """
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        _urdf_root_path = _current_dir + "/urdf"
        _urdf_robot_path = _urdf_root_path + "/tt.urdf"
        self.default_angles = [0, HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE] * 4
        self.rotational_links = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        self.render_mode = render_mode

        # Set the render mode
        if self.render_mode == "human":
            self.connection_mode = p.GUI
        elif self.render_mode == "rgb_array":
            if renderer == "OpenGL":
                self.connection_mode = p.GUI
            elif renderer == "Tiny":
                self.connection_mode = p.DIRECT

        self.physics_client = p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")

        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 60.0,
            solverResidualThreshold=1 - 10,
            numSolverIterations=50,
            numSubSteps=4,
        )
        if set_gravity:
            p.setGravity(0, 0, -9.8)

        self.start_pos = [0, 0, 0.3]
        self.cube_start_orientation = p.getQuaternionFromEuler(
            [1.5707963267948966, 0, -1.5707963267948966 * 2]
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

    def reset_position(self):
        p.resetBasePositionAndOrientation(
            self.agent, self.start_pos, self.cube_start_orientation
        )
        self.reset_angles()

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

    def set_action(self, action):
        """
        Sets the action for the quadruped robot.
        :param action: Action to set for the quadruped robot.
        """
        p.setJointMotorControlArray(
            self.agent,
            range(16),
            p.POSITION_CONTROL,
            action,
            np.zeros(16),
        )

    def get_observation(self):
        """
        Gets the observation for the quadruped robot.
        :return: Observation for the quadruped robot.
        """
        observation = []
        position, orientation = p.getBasePositionAndOrientation(self.agent)
        observation = [
            *position,  # x, y, z coordinates
            *orientation,  # x, y, z, w orientation
        ]

        for joint in self.rotational_links:
            joint_state = p.getJointState(self.agent, joint)
            observation.append(joint_state[0])  # Joint angle
            observation.append(joint_state[1])  # Joint velocity
            # observation.append(joint_state[2])  # Joint reaction forces For now no torque sensor!

        return observation

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

    quadruped_agent = TelosAgent(renderer="OpenGL")

    time.sleep(1)

    for _ in range(100):
        for angle in range(-45, 46):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            quadruped_agent.set_action(angles)
            quadruped_agent.step_simulation()
            print(quadruped_agent.get_observation())
            time.sleep(0.01)
        for angle in range(45, -46, -1):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            quadruped_agent.set_action(angles)
            quadruped_agent.step_simulation()
            time.sleep(0.01)

        # Step simulation
        # Sleep for a short duration to observe the movement
        quadruped_agent.reset_position()
        time.sleep(1)

    quadruped_agent.disconnect()
