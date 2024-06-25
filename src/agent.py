import os
import math
import time
import yaml
import numpy as np
import pybullet as p
import pybullet_data

from utils.telos_joints import (
    KNEE_ANGLE,
    THIGH_HIP_ANGLE,
    HIP_ANGLE,
    MOVING_JOINTS,
    TelosJoints,
)


class TelosAgent:
    def __init__(
        self,
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
    ) -> None:
        with open("pybullet_config.yaml", "r", encoding="utf-8") as file:
            _config = yaml.safe_load(file)
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        _urdf_root_path = _current_dir + _config["pybullet"]["robot"]["urdf_path"]
        self.n_substeps = _config["pybullet"]["simulation"]["num_substeps"]
        self.default_angles = [0, HIP_ANGLE, THIGH_HIP_ANGLE, KNEE_ANGLE] * 4
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.connection_mode = p.GUI
        elif self.render_mode == "rgb_array":
            if renderer == "OpenGL":
                self.connection_mode = p.GUI
            elif renderer == "Tiny":
                self.connection_mode = p.DIRECT

        self.physics_client = p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_plane = p.loadURDF("plane.urdf")
        p.setPhysicsEngineParameter(
            fixedTimeStep=_config["pybullet"]["simulation"]["time_step"],
            numSolverIterations=_config["pybullet"]["simulation"][
                "num_solver_iterations"
            ],
            numSubSteps=_config["pybullet"]["simulation"]["num_substeps"],
        )
        p.setGravity(*_config["pybullet"]["simulation"]["gravity"])

        self.start_pos = [*_config["pybullet"]["robot"]["start_position"]]
        self.cube_start_orientation = p.getQuaternionFromEuler(
            [*_config["pybullet"]["robot"]["start_orientation"]]
        )
        self.agent = p.loadURDF(
            _urdf_root_path, self.start_pos, self.cube_start_orientation
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

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.agent, self.start_pos, self.cube_start_orientation
        )
        self.reset_angles()

    def set_action(self, action):
        p.setJointMotorControlArray(
            self.agent,
            MOVING_JOINTS,
            p.POSITION_CONTROL,
            action,
            np.zeros(len(MOVING_JOINTS)),
        )

    def _get_obs(self):
        """
        Gets the observation for the quadruped robot.
        :return: Observation for the quadruped robot as a list of shape (31,).
        """
        observation = []
        position, orientation = p.getBasePositionAndOrientation(self.agent)
        observation = [
            *position,  # x, y, z coordinates
            *orientation,  # x, y, z, w orientation
        ]

        for joint in MOVING_JOINTS:
            joint_state = p.getJointState(self.agent, joint)
            observation.append(joint_state[0])  # Joint angle
            observation.append(joint_state[1])  # Joint velocity
            # observation.append(joint_state[2])  # Joint reaction forces For now no torque sensor!

        return observation

    def get_joint_acceleration(self, joint_id):
        """
        Gets the joint acceleration for the specified joint.
        :param joint_id: ID of the joint.
        :return: Joint acceleration.
        """
        return p.getJointState(self.agent, joint_id)[3]

    def get_acceleration_from_rotary(self):
        """
        Gets the acceleration from rotary joints.
        :return: Acceleration from rotary joints.
        """
        acceleration = []
        for joint in MOVING_JOINTS:
            acceleration.append(self.get_joint_acceleration(joint))
        return acceleration

    def get_center_of_mass(self):
        """
        Gets the center of mass of the agent.
        :return: Center of mass of the agent.
        """
        return p.getBasePositionAndOrientation(self.agent)[0]

    def get_contact_points_with_ground(self):
        """
        Gets the contact points of the agent with the ground.
        :return: Contact points of the agent with the ground.
        """
        contact_points = p.getContactPoints(self.agent, self.ground_plane)
        euc_contact_points = []
        for contact_point in contact_points:
            euc_contact_points.append(contact_point[5])
        if not euc_contact_points:
            return False, None
        euc_contact_points = np.array(euc_contact_points)
        euc_contact_points[:, 2] = 0
        return True, euc_contact_points

    def step_simulation(self):
        """
        Steps the simulation forward by one time step.
        """
        for _ in range(self.n_substeps):
            p.stepSimulation()

    def disconnect(self):
        """
        Disconnects from PyBullet.
        """
        p.disconnect(self.physics_client)
