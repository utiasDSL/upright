# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
import matplotlib.pyplot as plt

from mm_pybullet_sim.recording import DATA_DRIVE_PATH
from mm_pybullet_sim.robot import RobotModel
import mm_pybullet_sim.util as util
from liegroups import SO3

import IPython

# DATA_PATH = DATA_DRIVE_PATH / "dynamic-obs/dynamic_obs2_rate50_2021-09-13_01-53-39.npz"
DATA_PATH = DATA_DRIVE_PATH / "multi-object/cups/cups3_goal3_2021-09-11_02-42-52.npz"


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.ts = data["ts"]
            self.xs = data["xs"]
            self.us = data["us"]

            self.v_ew_ws = data["v_ew_ws"]

            # self.r_ew_ws = data["r_ew_ws"]
            # self.r_ew_wds = data["r_ew_wds"]
            # self.dynamic_obs_distance = data["dynamic_obs_distance"]

        # self.r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        # self.r_ew_w_err_norm = np.linalg.norm(self.r_ew_w_err, axis=1)

        data_length = self.ts.shape[0]
        if tf is not None:
            for i in range(self.ts.shape[0]):
                if self.ts[i] > tf:
                    data_length = i
                    break

        self.ts_cut = self.ts[:data_length]
        self.vel_norms_real = np.linalg.norm(self.v_ew_ws[:data_length, :], axis=1)

        # compute acceleration via finite difference
        acc_diffed = (self.v_ew_ws[1:data_length, :] - self.v_ew_ws[:data_length - 1, :]) / (self.ts[1] - self.ts[0])
        self.acc_norms_diffed = np.linalg.norm(acc_diffed, axis=1)

        # print(f"max vel (sim) = {np.max(self.vel_norms_real)}")
        # print(f"max acc (sim) = {np.max(self.acc_norms_diffed)}")
        # import sys
        # sys.exit()


        # self.xs_cut = self.xs[:data_length]
        # self.us_cut = self.us[:data_length]
        # self.dynamic_obs_distance_cut = self.dynamic_obs_distance[:data_length]
        # self.min_clearance = np.min(self.dynamic_obs_distance_cut, axis=1)

        # self.r_ew_ws_cut = self.r_ew_ws[:data_length, :]
        # self.r_ew_wds_cut = self.r_ew_wds[:data_length, :]
        # self.r_ew_w_err_cut = self.r_ew_wds_cut - self.r_ew_ws_cut

        model = RobotModel(dt=0.1)  # dt doesn't matter here
        vel = np.zeros((data_length, 3))
        acc = np.zeros((data_length, 3))
        for i in range(data_length):
            x = self.xs[i, :]
            u = self.us[i, :]
            q, v = x[:9], x[9:]
            V = model.tool_velocity(q, v)
            A = model.tool_acceleration(x, u)
            vel[i, :] = V[0]  # TODO not sure why V is returned as a tuple
            acc[i, :] = A[:3]

        self.vel_norms = np.linalg.norm(vel, axis=1)
        self.acc_norms = np.linalg.norm(acc, axis=1)


def main():
    # load the data
    tf = 5
    data = TrajectoryData(DATA_PATH, tf=tf)

    print(f"max vel (model) = {np.max(data.vel_norms)}")
    print(f"max acc (model) = {np.max(data.acc_norms)}")

    # also check that sim/diffed versions correspond to those from the model
    print(f"max vel (sim) = {np.max(data.vel_norms_real)}")
    print(f"max acc (sim) = {np.max(data.acc_norms_diffed)}")

    plt.plot(data.ts_cut, data.vel_norms, label="vel")
    plt.plot(data.ts_cut, data.acc_norms, label="acc")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("motion")
    plt.show()


if __name__ == "__main__":
    main()
