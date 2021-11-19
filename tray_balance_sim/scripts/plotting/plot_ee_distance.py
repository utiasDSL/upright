# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
import matplotlib.pyplot as plt

from tray_balance_sim.recording import DATA_DRIVE_PATH
import tray_balance_sim.util as util
from liegroups import SO3

import IPython

DATA_PATH = DATA_DRIVE_PATH / "single-object"
NO_OBJ_PATHS = [
    DATA_DRIVE_PATH / "new-multi-object" / filename
    for filename in [
        "no_obj_goal1_2021-09-14_20-51-52.npz",
        "no_obj_goal2_2021-09-14_20-46-27.npz",
        "no_obj_goal3_2021-09-14_20-49-17.npz",
    ]
]
HEIGHT_2CM_PATHS = [
    DATA_PATH / "height-0.02" / filename
    for filename in (
        "h0.02_goal1_all_2021-09-08_22-25-44.npz",
        "h0.02_goal2_all_2021-09-08_22-27-05.npz",
        "h0.02_goal3_all_2021-09-08_22-28-36.npz",
    )
]
HEIGHT_20CM_PATHS = [
    DATA_PATH / "height-0.20" / filename
    for filename in (
        "h0.2_goal1_all_2021-09-08_22-55-45.npz",
        "h0.2_goal2_all_2021-09-08_22-58-10.npz",
        "h0.2_goal3_all_2021-09-08_23-00-33.npz",
    )
]
HEIGHT_100CM_PATHS = [
    DATA_PATH / "height-1.00" / filename
    for filename in (
        "h1.0_goal1_all_2021-09-08_23-33-47.npz",
        "h1.0_goal2_all_2021-09-08_23-35-55.npz",
        "h1.0_goal3_all_2021-09-08_23-42-04.npz",
    )
]

# EXTRA_PATH = DATA_DRIVE_PATH / "h1.7_goal1_all_2021-09-09_17-04-22.npz"

FIG_PATH = "/home/adam/phd/papers/icra22/figures/ee_convergence.pdf"


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.ts = data["ts"]
            self.r_ew_ws = data["r_ew_ws"]
            self.r_ew_wds = data["r_ew_wds"]
            self.Q_wes = data["Q_wes"]
            self.Q_weds = data["Q_weds"]

        self.r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        self.r_ew_w_err_norm = np.linalg.norm(self.r_ew_w_err, axis=1)

        self.Q_des = np.zeros_like(self.Q_wes)
        self.rpy_des = np.zeros((self.Q_wes.shape[0], 3))
        self.angle_err = np.zeros(self.Q_wes.shape[0])
        for i in range(self.Q_des.shape[0]):
            self.Q_des[i, :] = util.quat_multiply(
                util.quat_inverse(self.Q_weds[i, :]), self.Q_wes[i, :]
            )
            self.rpy_des[i, :] = SO3.from_quaternion(
                self.Q_des[i, :], ordering="xyzw"
            ).to_rpy()
            self.angle_err[i] = util.quat_error(self.Q_des[i, :])

        if tf is not None:
            for i in range(self.ts.shape[0]):
                if self.ts[i] > tf:
                    data_length = i
                    break
        else:
            data_length = self.ts.shape[0]

        self.ts_cut = self.ts[:data_length]
        self.r_ew_w_err_norm_cut = self.r_ew_w_err_norm[:data_length]
        self.angle_err_cut = self.angle_err[:data_length]

        # for i in range(err_norm.shape[0]):
        #     if err_norm[i] <= 0.01:
        #         print(f"convergence time = {ts[i]} s")
        #         break


def main():
    # load the data
    tf = 3
    height_2cm_data = [TrajectoryData(path, tf=tf) for path in HEIGHT_2CM_PATHS]
    # height_20cm_data = [TrajectoryData(path, tf=tf) for path in HEIGHT_20CM_PATHS]
    height_100cm_data = [TrajectoryData(path, tf=tf) for path in HEIGHT_100CM_PATHS]
    no_obj_data = [TrajectoryData(path, tf=tf) for path in NO_OBJ_PATHS]

    # extra_data = TrajectoryData(EXTRA_PATH, tf=4)

    # plt.figure()
    # plt.plot(
    #     height_2cm_data[0].ts_cut,
    #     height_2cm_data[0].angle_err_cut,
    #     label=r"$2\ \mathrm{cm}$",
    # )
    # plt.plot(
    #     height_20cm_data[0].ts_cut,
    #     height_20cm_data[0].angle_err_cut,
    #     label=r"$20\ \mathrm{cm}$",
    # )
    # plt.plot(
    #     height_100cm_data[0].ts_cut,
    #     height_100cm_data[0].angle_err_cut,
    #     label=r"$100\ \mathrm{cm}$",
    # )
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Orientation error (rad)")
    # plt.title("Orientation error norm")
    #
    # plt.figure()
    # plt.plot(height_2cm_data[0].ts, height_2cm_data[0].rpy_des[:, 0], label="x")
    # plt.plot(height_2cm_data[0].ts, height_2cm_data[0].rpy_des[:, 1], label="y")
    # plt.plot(height_2cm_data[0].ts, height_2cm_data[0].rpy_des[:, 2], label="z")
    # plt.legend()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Orientation error (rad)")
    # plt.title("Orientation error (2cm)")
    #
    # plt.figure()
    # plt.plot(height_100cm_data[0].ts, height_100cm_data[0].rpy_des[:, 0], label="x")
    # plt.plot(height_100cm_data[0].ts, height_100cm_data[0].rpy_des[:, 1], label="y")
    # plt.plot(height_100cm_data[0].ts, height_100cm_data[0].rpy_des[:, 2], label="z")
    # plt.legend()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Orientation error (rad)")
    # plt.title("Orientation error (100cm)")

    fig = plt.figure(figsize=(3.25, 1.25))
    plt.rcParams.update(
        {"font.size": 8, "text.usetex": True, "legend.fontsize": 8, "axes.titlesize": 8}
    )

    ax1 = plt.subplot(131)
    plt.plot(
        no_obj_data[0].ts_cut,
        no_obj_data[0].r_ew_w_err_norm_cut,
        label=r"$\mathrm{None}$",
    )
    plt.plot(
        height_2cm_data[0].ts_cut,
        height_2cm_data[0].r_ew_w_err_norm_cut,
        label=r"$\mathrm{Short}$",
    )
    # plt.plot(
    #     height_20cm_data[0].ts_cut,
    #     height_20cm_data[0].r_ew_w_err_norm_cut,
    #     label=r"$20\ \mathrm{cm}$",
    # )
    plt.plot(
        height_100cm_data[0].ts_cut,
        height_100cm_data[0].r_ew_w_err_norm_cut,
        label=r"$\mathrm{Tall}$",
    )
    plt.ylabel(r"$\mathrm{EE\ distance}$" "\n" r"$\mathrm{to\ goal\ (m)}$")
    plt.title(r"$\mathrm{Goal\ 1}$")
    ax1.set_xticks([0, 1, 2, 3])

    ax2 = plt.subplot(132)
    plt.plot(
        no_obj_data[1].ts_cut,
        no_obj_data[1].r_ew_w_err_norm_cut,
        label=r"$\mathrm{None}$",
    )
    plt.plot(
        height_2cm_data[1].ts_cut,
        height_2cm_data[1].r_ew_w_err_norm_cut,
        label=r"$\mathrm{Short}$",
    )
    # plt.plot(
    #     height_20cm_data[1].ts_cut,
    #     height_20cm_data[1].r_ew_w_err_norm_cut,
    #     label=r"$20\ \mathrm{cm}$",
    # )
    plt.plot(
        height_100cm_data[1].ts_cut,
        height_100cm_data[1].r_ew_w_err_norm_cut,
        label=r"$\mathrm{Tall}$",
    )
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.title(r"$\mathrm{Goal\ 2}$")
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0, 1, 2, 3])

    ax3 = plt.subplot(133)
    plt.plot(
        no_obj_data[2].ts_cut,
        no_obj_data[2].r_ew_w_err_norm_cut,
        label=r"$\mathrm{None}$",
    )
    plt.plot(
        height_2cm_data[2].ts_cut,
        height_2cm_data[2].r_ew_w_err_norm_cut,
        label=r"$\mathrm{Short}$",
    )
    # plt.plot(
    #     height_20cm_data[2].ts_cut,
    #     height_20cm_data[2].r_ew_w_err_norm_cut,
    #     label=r"$20\ \mathrm{cm}$",
    # )
    plt.plot(
        height_100cm_data[2].ts_cut,
        height_100cm_data[2].r_ew_w_err_norm_cut,
        label=r"$\mathrm{Tall}$",
    )
    plt.legend(handlelength=1)
    plt.title(r"$\mathrm{Goal\ 3}$")
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    ax3.set_xticks([0, 1, 2, 3])

    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
