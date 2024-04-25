import time
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)  # or DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF(
    "tt.urdf",
    startPos,
    cubeStartOrientation,
)

# for i in range(p.getNumJoints(robotId)):
#     print(p.getJointInfo(robotId, i))

while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
