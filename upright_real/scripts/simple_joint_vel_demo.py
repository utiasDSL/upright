import numpy as np
import time

from perls2.robots.real_panda_interface import RealPandaInterface
from perls2.utils.yaml_config import YamlConfig

import tray_balance_constraints as core

import IPython


def main():
    config = core.parsing.load_config("../config/simple_joint_vel_demo.yaml")
    robot = RealPandaInterface(config, controlType="JointVelocity")
    robot.reset()

    q0 = np.copy(robot.q)
    K = 0.5 * np.eye(7)
    amp = 0.2
    freq = 2

    dt = 0.01  # seconds
    N = 600  # duration = N * dt

    rate = core.util.Rate.from_timestep_secs(dt)
    dt = rate.timestep_secs

    t = 0
    for i in range(N):
        # joint feedback
        q = robot.q

        # desired joint angles
        qd = q0 + [0, 0, 0, 0, 0, amp * (1 - np.cos(freq * t)), 0]
        vd = np.array([0, 0, 0, 0, 0, amp * freq * np.sin(freq * t), 0])

        # joint velocity controller
        v = K @ (qd - q) + vd

        print(f"error = {qd - q}")

        # send the command
        robot.set_joint_velocities(v)

        t += dt
        # time.sleep(dt)
        rate.sleep()

    # put robot back to home position
    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
