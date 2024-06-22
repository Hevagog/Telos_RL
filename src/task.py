import numpy as np
import pybullet as p
import pybullet_data

import math
import time

from agent import TelosAgent


class TelosTask:
    def __init__(self, agent: TelosAgent):
        self.agent = agent
        self.goal_radius = 10
        self.angle_bounds = np.deg2rad([-45, 45])
        theta = np.random.uniform(*self.angle_bounds)
        self.goal = np.array(
            [self.goal_radius * np.cos(theta), self.goal_radius * np.sin(theta), 0]
        )
        self.dist_threshold = 0.1

    def reset(self):
        theta = np.random.uniform(*self.angle_bounds)
        self.goal = np.array(
            [self.goal_radius * np.cos(theta), self.goal_radius * np.sin(theta), 0]
        )
        self.agent.reset_position()

    def get_obs(self):
        return self.agent.get_obs()

    def get_achieved_goal(self):
        return self.agent.get_obs()

    def is_success(self, achieved_goal, desired_goal, info={}):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return np.array(distance < self.dist_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        healthy_reward = 0 if achieved_goal[2] > 0.2 else -100
        distance_reward = -np.linalg.norm(achieved_goal - desired_goal)
        return np.array(healthy_reward + distance_reward, dtype=np.float32)


####################################################################################################

if __name__ == "__main__":
    quadruped_agent = TelosAgent(renderer="OpenGL")
    task = TelosTask(quadruped_agent)

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CAPSULE,
        rgbaColor=[
            1.0,
            0.25,
            1.0,
            1.0,
        ],
        radius=0.1,
        length=100,
    )
    body_id = p.createMultiBody(
        basePosition=[*task.goal], baseVisualShapeIndex=visual_shape_id
    )
    # while True:
    #     # p.stepSimulation()
    #     # quadruped_agent.step_simulation()
    #     # print(task.compute_reward(quadruped_agent.get_obs()[0:3], task.goal))
    #     time.sleep(1.0 / 60.0)
    for _ in range(4):
        for angle in range(-45, 46):
            angles = np.array([math.radians(angle)] * 16)
            angles = quadruped_agent.default_angles + angles
            indices = [0, 4, 8, 12]
            angles[indices] = 0
            quadruped_agent.set_action(angles)
            quadruped_agent.step_simulation()
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
        # quadruped_agent.reset_position()

        time.sleep(1)
    task.reset()
    quadruped_agent.step_simulation()
    time.sleep(10)

    quadruped_agent.disconnect()
