import numpy as np
import time
import datetime

from perls2.robots.real_panda_interface import RealPandaInterface

import tray_balance_constraints as core
from tray_balance_constraints.logging import DataLogger, DataPlotter

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

    # data logging
    log_dir = Path(config["logging"]["log_dir"])
    log_dt = config["logging"]["timestep"]
    logger = DataLogger(config)

    now = datetime.datetime.now()

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

        # send the command
        robot.set_joint_velocities(v)

        if i % log_dt == 0:
            logger.append("qs", q)
            logger.append("qds", qd)
            logger.append("v_cmds", v)

        rate.sleep()

    # put robot back to home position
    robot.reset()
    robot.disconnect()

    logger.save(log_dir, now, name="real_panda")


if __name__ == "__main__":
    main()
