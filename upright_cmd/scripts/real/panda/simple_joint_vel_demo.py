import numpy as np

from perls2.robots.real_panda_interface import RealPandaInterface

import upright_core as core
import upright_cmd as cmd

import IPython


def main():
    args = cmd.cli.basic_arg_parser().parse_args()
    config = core.parsing.load_config(args.config)

    robot = RealPandaInterface(config["perls2"], controlType="JointVelocity")
    robot.reset()

    q0 = np.copy(robot.q)
    K = 0.5 * np.eye(7)
    amp = 0.2
    freq = 2

    dt = 0.01  # seconds
    N = 1000  # duration = N * dt

    rate = core.util.Rate.from_timestep_secs(dt)

    t = 0
    for i in range(N):
        # joint feedback
        q = robot.q

        # desired joint angles
        qd = q0 + [0, 0, 0, 0, 0, amp * (1 - np.cos(freq * t)), 0]
        vd = np.array([0, 0, 0, 0, 0, amp * freq * np.sin(freq * t), 0])

        # joint velocity controller
        v_cmd = K @ (qd - q) + vd

        # send the command
        robot.set_joint_velocities(v_cmd)

        t += dt
        rate.sleep()

    # put robot back to home position
    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
