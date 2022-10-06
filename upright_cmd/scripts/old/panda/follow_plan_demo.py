import numpy as np
import time
import datetime
from pathlib import Path

from perls2.robots.real_panda_interface import RealPandaInterface

import upright_core as core
from upright_core.logging import DataLogger, DataPlotter
import upright_cmd as cmd

import IPython


def main():
    args = cmd.cli.basic_arg_parser().parse_args()
    config = core.parsing.load_config(args.config)

    # load trajectory data
    with np.load("trajectory.npz") as data:
        ts = data["ts"]
        xds = data["xs"]

    nq = nv = 7  # TODO maybe load these from the trajectory file?
    step = 2
    dt = step * (ts[1] - ts[0])  # timestep

    # desired trajectory
    qds = xds[:, :nq]
    vds = xds[:, nq:nq+nv]

    # position gain
    K = 10 * np.eye(7)

    rate = core.util.Rate.from_timestep_secs(dt, quiet=True)

    # data logging
    now = datetime.datetime.now()
    log_dir = Path(config["logging"]["log_dir"])
    log_dt = config["logging"]["timestep"]
    logger = DataLogger(config)
    logger.add("log_dt", log_dt)

    # TODO: note it would be better to actually acquire the robot using a
    # `with` block, which would then take care of the required clean-up
    # automatically
    robot = RealPandaInterface(config["perls2"], controlType="JointVelocity")
    robot.reset()

    try:
        # M = robot.mass_matrix
        # c = robot.coriolis_vector
        # g = robot.gravity_vector
        #
        # print(f"M = {M}")
        # print(f"c = {c}")
        # print(f"g = {g}")

        for i in range(0, ts.shape[0], step):
            # start = time.time_ns()

            # joint feedback
            q = robot.q
            v = robot.dq
            τ = robot.tau

            # desired joint angles
            qd = qds[i, :]
            vd = vds[i, :]

            # joint velocity controller
            v_cmd = K @ (qd - q) + vd

            # send the command
            robot.set_joint_velocities(v_cmd)

            # loop_duration = time.time_ns() - start
            # print(f"loop duration = {loop_duration / 1e6} ms")

            if i % log_dt == 0:
                logger.append("qs", q)
                logger.append("vs", v)
                logger.append("qds", qd)
                logger.append("v_cmds", v_cmd)
                logger.append("taus", τ)

            rate.sleep()
    finally:
        # put robot back to home position and close connection
        robot.reset()
        robot.disconnect()

    logger.save(log_dir, now, name="real_panda")


if __name__ == "__main__":
    main()
