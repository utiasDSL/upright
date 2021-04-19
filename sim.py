import pybullet as pyb
import time
import pybullet_data

# import pybullet_data

import IPython


SIM_DT = 0.001

UR10_JOINT_NAMES = [
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]

UR10_HOME = [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]


def main():
    clid = pyb.connect(pyb.GUI)

    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # pyb.setPhysicsEngineParameter(enableConeFriction=0)
    pyb.setTimeStep(SIM_DT)

    # ridgeback_position = [0, 0, 0]
    # ridgeback_orientation = [0, 0, 0, 1]
    # ridgeback = pyb.loadURDF(
    #     "assets/urdf/ridgeback.urdf", ridgeback_position, ridgeback_orientation
    # )
    #
    # ur10_base_position = [0, 0, 1]
    # ur10_base_orientation = [0, 0, 0, 1]
    # ur10 = pyb.loadURDF(
    #     "assets/urdf/ur10.urdf", ur10_base_position, ur10_base_orientation
    # )

    pyb.loadURDF("plane.urdf", [0, 0, 0])

    mm_base_position = [0, 0, 0]
    mm_base_orientation = [0, 0, 0, 1]
    mm = pyb.loadURDF(
        "assets/urdf/mm.urdf",
        mm_base_position,
        mm_base_orientation,
        flags=pyb.URDF_MERGE_FIXED_LINKS,
    )

    # build a dict of all joints, keyed by name
    joints = {}
    for i in range(pyb.getNumJoints(mm)):
        info = pyb.getJointInfo(mm, i)
        name = info[1].decode("utf-8")
        joints[name] = info

    # get the indices for the UR10 joints
    ur10_joint_indices = []
    for name in UR10_JOINT_NAMES:
        idx = joints[name][0]
        ur10_joint_indices.append(idx)

    # set the UR10 to the home position
    for idx, value in zip(ur10_joint_indices, UR10_HOME):
        pyb.resetJointState(mm, idx, value)

    pyb.setGravity(0, 0, -9.81)

    t = 0

    # simulation loop
    while True:
        # base is controlled directly in Cartesian coordinates
        pyb.resetBaseVelocity(mm, [0.1, 0, 0])

        # UR10 joint control
        pyb.setJointMotorControlArray(
            mm,
            ur10_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=[0.1, 0, 0, 0, 0, 0],
        )

        pyb.stepSimulation()

        t += SIM_DT
        # TODO smart sleep a la ROS - is there a standalone package for this?
        time.sleep(SIM_DT)


if __name__ == "__main__":
    main()
