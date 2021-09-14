# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
import matplotlib.pyplot as plt

from mm_pybullet_sim.recording import DATA_DRIVE_PATH
import mm_pybullet_sim.util as util
from liegroups import SO3

import IPython

DATA_PATH = DATA_DRIVE_PATH / "static_obs_2021-09-13_02-25-11.npz"

FIG_PATH = "/home/adam/phd/papers/icra22/figures/static_obstacles.pdf"


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.ts = data["ts"]
            self.r_ew_ws = data["r_ew_ws"]
            self.r_ew_wds = data["r_ew_wds"]
            # self.Q_wes = data["Q_wes"]
            # self.Q_weds = data["Q_weds"]
            self.collision_pair_distance = data["collision_pair_distance"]

        self.r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        self.r_ew_w_err_norm = np.linalg.norm(self.r_ew_w_err, axis=1)

        # self.Q_des = np.zeros_like(self.Q_wes)
        # self.rpy_des = np.zeros((self.Q_wes.shape[0], 3))
        # self.angle_err = np.zeros(self.Q_wes.shape[0])
        # for i in range(self.Q_des.shape[0]):
        #     self.Q_des[i, :] = util.quat_multiply(
        #         util.quat_inverse(self.Q_weds[i, :]), self.Q_wes[i, :]
        #     )
        #     self.rpy_des[i, :] = SO3.from_quaternion(
        #         self.Q_des[i, :], ordering="xyzw"
        #     ).to_rpy()
        #     self.angle_err[i] = util.quat_error(self.Q_des[i, :])

        data_length = self.ts.shape[0]
        if tf is not None:
            for i in range(self.ts.shape[0]):
                if self.ts[i] > tf:
                    data_length = i
                    break

        self.ts_cut = self.ts[:data_length]
        self.collision_pair_distance_cut = self.collision_pair_distance[:data_length]
        self.min_clearance = np.min(self.collision_pair_distance_cut, axis=1)
        self.min_base_clearance = np.min(self.collision_pair_distance_cut[:, 1:13], axis=1)
        self.min_arm_clearance = np.min(self.collision_pair_distance_cut[:, 16:29], axis=1)
        self.min_obj_clearance = np.min(self.collision_pair_distance_cut[:, 13:16], axis=1)

        self.r_ew_ws_cut = self.r_ew_ws[:data_length, :]
        self.r_ew_wds_cut = self.r_ew_wds[:data_length, :]

        # TODO hard-coded for now to match the trajectory
        self.r_ew_wds_cut[:, 0] = self.r_ew_wds_cut[0, 0] + 0.5 * self.ts_cut
        self.r_ew_w_err_cut = self.r_ew_wds_cut - self.r_ew_ws_cut

        # self.r_ew_w_err_norm_cut = self.r_ew_w_err_norm[:data_length]
        # self.angle_err_cut = self.angle_err[:data_length]


def main():
    # load the data
    tf = 10
    data = TrajectoryData(DATA_PATH, tf=tf)

    fig = plt.figure(figsize=(3.25, 1.8))
    plt.rcParams.update(
        {"font.size": 8, "text.usetex": True, "legend.fontsize": 8, "axes.titlesize": 8}
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax1 = plt.subplot(211)
    plt.plot(data.ts_cut, data.min_base_clearance, label=r"$\mathrm{Base}$")
    plt.plot(data.ts_cut, data.min_arm_clearance, label=r"$\mathrm{Arm}$")
    plt.plot(data.ts_cut, data.min_obj_clearance, label=r"$\mathrm{Objects}$")

    plt.ylabel(r"$\mathrm{Clearance}$" "\n" r"$\mathrm{(m)}$")
    plt.legend(labelspacing=0.3, handlelength=1, loc=(0.47, 0.3))
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([0, 0.5, 1])
    ax1.yaxis.set_label_coords(-0.11, 0.5)

    ax2 = plt.subplot(212)
    ax2.axhline(0, color=(0.75, 0.75, 0.75))
    plt.plot(data.ts_cut, data.r_ew_w_err_cut[:, 0], label="$x$", color=colors[3])
    plt.plot(data.ts_cut, data.r_ew_w_err_cut[:, 1], label="$y$", color=colors[4])
    plt.plot(data.ts_cut, data.r_ew_w_err_cut[:, 2], label="$z$", color=colors[5])
    plt.legend(labelspacing=0.3, handlelength=1, loc=(0.75, 0.3))
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.ylabel(r"$\mathrm{EE\ error}$" "\n" r"$\mathrm{(m)}$")
    ax2.yaxis.set_label_coords(-0.11, 0.5)

    # fig.align_ylabels()
    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
