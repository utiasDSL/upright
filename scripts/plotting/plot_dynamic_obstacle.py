# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
import matplotlib.pyplot as plt

from mm_pybullet_sim.recording import DATA_DRIVE_PATH
import mm_pybullet_sim.util as util
from liegroups import SO3

import IPython

DATA_PATH = DATA_DRIVE_PATH / "dynamic_obs2_rate50_2021-09-13_01-53-39.npz"

FIG_PATH = "/home/adam/phd/papers/icra22/figures/dynamic_obstacles.pdf"


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.ts = data["ts"]
            self.r_ew_ws = data["r_ew_ws"]
            self.r_ew_wds = data["r_ew_wds"]
            self.dynamic_obs_distance = data["dynamic_obs_distance"]

        self.r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        self.r_ew_w_err_norm = np.linalg.norm(self.r_ew_w_err, axis=1)

        data_length = self.ts.shape[0]
        if tf is not None:
            for i in range(self.ts.shape[0]):
                if self.ts[i] > tf:
                    data_length = i
                    break

        self.ts_cut = self.ts[:data_length]
        self.dynamic_obs_distance_cut = self.dynamic_obs_distance[:data_length]
        self.min_clearance = np.min(self.dynamic_obs_distance_cut, axis=1)

        # 0th position is just a placeholder
        # TODO what I should do is just not plot the first bit until the controller updates
        self.min_clearance[0] = self.min_clearance[1]
        # self.min_base_clearance = np.min(self.collision_pair_distance_cut[:, 1:13], axis=1)
        # self.min_arm_clearance = np.min(self.collision_pair_distance_cut[:, 16:29], axis=1)
        # self.min_obj_clearance = np.min(self.collision_pair_distance_cut[:, 13:16], axis=1)

        self.r_ew_ws_cut = self.r_ew_ws[:data_length, :]
        self.r_ew_wds_cut = self.r_ew_wds[:data_length, :]
        self.r_ew_w_err_cut = self.r_ew_wds_cut - self.r_ew_ws_cut


def main():
    # load the data
    tf = 10
    data = TrajectoryData(DATA_PATH, tf=tf)

    fig = plt.figure(figsize=(3.25, 2))
    plt.rcParams.update(
        {"font.size": 8, "text.usetex": True, "legend.fontsize": 8, "axes.titlesize": 8}
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax1 = plt.subplot(211)
    plt.plot(data.ts_cut, data.min_clearance)
    plt.ylabel(r"$\mathrm{Clearance\ (m)}$")
    # plt.grid()
    ax1.set_xticks([])
    ax1.set_xticklabels([])

    ax2 = plt.subplot(212)
    ax2.axhline(0, color=(0.75, 0.75, 0.75))
    plt.plot(data.ts_cut, data.r_ew_w_err_cut[:, 0], label="$x$", color=colors[3])
    plt.plot(data.ts_cut, data.r_ew_w_err_cut[:, 1], label="$y$", color=colors[4])
    plt.plot(data.ts_cut, data.r_ew_w_err_cut[:, 2], label="$z$", color=colors[5])
    plt.legend(labelspacing=0.3)
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.ylabel(r"$\mathrm{EE\ error\ (m)}$")

    fig.align_ylabels()
    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_PATH)
    # print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
