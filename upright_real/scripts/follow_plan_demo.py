import numpy as np
import time

from perls2.robots.real_panda_interface import RealPandaInterface

import tray_balance_constraints as core

import IPython


def main():
    config = core.parsing.load_config("../config/follow_plan_demo.yaml")

    K = np.eye(7)

    # load trajectory data
    with np.load("trajectory.npz") as data:
        ts = data["ts"]
        xds = data["xs"]

    dt = ts[1] - ts[0]
    qds = xds[:, :7]
    vds = xds[:, 7:14]

    rate = core.util.Rate.from_timestep_secs(dt)

    robot = RealPandaInterface(config, controlType="JointVelocity")
    robot.reset()

    for i in range(ts.shape[0]):
        # joint feedback
        q = robot.q

        # desired joint angles
        qd = qds[i, :]
        vd = vds[i, :]

        # joint velocity controller
        v = K @ (qd - q) + vd

        # print(f"error = {qd - q}")

        # send the command
        robot.set_joint_velocities(v)

        # time.sleep(dt)
        rate.sleep()

    # put robot back to home position
    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
