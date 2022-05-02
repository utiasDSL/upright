import numpy as np
import time
import datetime
from pathlib import Path

from perls2.robots.real_panda_interface import RealPandaInterface

import tray_balance_constraints as core
from tray_balance_constraints.logging import DataLogger, DataPlotter
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
    ads = xds[:, -nv:]

    # position and velocity gain
    Kp = 50 * np.eye(7)
    damping = 1
    Kv = 2 * np.sqrt(Kp) * damping

    rate = core.util.Rate.from_timestep_secs(dt, quiet=True)

    # data logging
    now = datetime.datetime.now()
    log_dir = Path(config["logging"]["log_dir"])
    log_dt = config["logging"]["timestep"]
    logger = DataLogger(config)
    logger.add("log_dt", log_dt)

    robot = RealPandaInterface(config["perls2"], controlType="JointVelocity")
    robot.reset()

    try:
        for i in range(0, ts.shape[0], step):
            # joint feedback
            q = robot.q
            v = robot.dq
            τ = robot.tau

            M = robot.mass_matrix

            # desired joint angles
            qd = qds[i, :]
            vd = vds[i, :]
            ad = ads[i, :]

            a_cmd = Kp @ (qd - q) + Kv @ (vd - v) + ad

            # TODO Panda compensates for gravity internally, but we could add
            # Coriolis effects here
            τ_cmd = M @ a_cmd

            robot.set_joint_torques(τ_cmd)

            if i % log_dt == 0:
                logger.append("qs", q)
                logger.append("vs", v)
                logger.append("taus", τ)

                logger.append("vds", vd)
                logger.append("qds", qd)
                logger.append("ads", ad)

                logger.append("a_cmds", a_cmd)
                logger.append("tau_cmds", τ_cmd)

            rate.sleep()
    finally:
        # put robot back to home position and close connection
        robot.reset()
        robot.disconnect()

    logger.save(log_dir, now, name="real_panda")


if __name__ == "__main__":
    main()
