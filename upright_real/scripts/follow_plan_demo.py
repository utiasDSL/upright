import numpy as np
import time
from collections import OrderedDict

from perls2.robots.real_panda_interface import RealPandaInterface
from perls2.utils.yaml_config import YamlConfig

import tray_balance_constraints as core

import IPython


class Rate:
    def __init__(self, hz):
        self.last_time = time.time()
        self.secs = 1. / hz

    def sleep(self):
        elapsed = time.time() - self.last_time
        duration = self.secs - elapsed
        if duration > 0:
            time.sleep(duration)
        self.last_time = time.time()


def unorder_dict(d):
    if type(d) == OrderedDict:
        d = dict(d)

    if type(d) == dict:
        for key, val in d.items():
            d[key] = unorder_dict(val)

    return d


def main():
    config_file = "../config/follow_plan_demo.yaml"
    config = YamlConfig(config_file)

    # config_mine = core.parsing.load_config(config_mine)
    # assert config_mine == unorder_dict(config.config)

    IPython.embed()
    return

    robot = RealPandaInterface(config, controlType="JointVelocity")
    robot.reset()

    K = np.eye(7)

    # load trajectory data
    with np.load("trajectory.npz") as data:
        ts = data["ts"]
        xds = data["xs"]

    dt = ts[1] - ts[0]
    qds = xds[:, :7]
    vds = xds[:, 7:14]

    rate = Rate(1. / dt)

    for i in range(ts.shape[0]):
        # joint feedback
        q = robot.q

        # desired joint angles
        qd = qds[i, :]
        vd = vds[i, :]

        # joint velocity controller
        v = K @ (qd - q) + vd

        print(f"error = {qd - q}")

        # send the command
        robot.set_joint_velocities(v)

        time.sleep(dt)
        # rate.sleep()

    # put robot back to home position
    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
