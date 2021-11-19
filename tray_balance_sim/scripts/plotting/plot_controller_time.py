# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from mm_pybullet_sim.recording import DATA_DRIVE_PATH
import mm_pybullet_sim.util as util
from liegroups import SO3

import IPython

# DATA_PATH = DATA_DRIVE_PATH / "box1_goal1_2021-09-09_20-54-18.npz"
# DATA_PATH = DATA_DRIVE_PATH / "box2_goal1_2021-09-09_21-06-18.npz"
DATA_PATH = DATA_DRIVE_PATH / "multi-object"

TRAY_ONLY_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/tray_only_goal1_2021-09-11_18-38-39.npz",
    DATA_DRIVE_PATH / "new-multi-object/tray_only_goal1_2021-09-14_15-15-28.npz",
    DATA_DRIVE_PATH / "new-multi-object/tray_only_goal1_2021-09-14_15-16-35.npz",
]
TRAY_ONLY_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/tray_only_goal2_2021-09-11_18-35-52.npz",
    DATA_DRIVE_PATH / "new-multi-object/tray_only_goal2_2021-09-14_15-00-03.npz",
    DATA_DRIVE_PATH / "new-multi-object/tray_only_goal2_2021-09-14_15-05-13.npz",
]
TRAY_ONLY_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/tray_only_goal3_2021-09-11_18-37-19.npz",
    DATA_DRIVE_PATH / "new-multi-object/tray_only_goal3_2021-09-14_15-10-14.npz",
    DATA_DRIVE_PATH / "new-multi-object/tray_only_goal3_2021-09-14_15-12-23.npz",
]

STACK1_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack1_goal1_2021-09-11_18-31-16.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack1_goal1_2021-09-14_14-53-47.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack1_goal1_2021-09-14_14-55-36.npz",
]
STACK1_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack1_goal2_2021-09-11_18-27-08.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack1_goal2_2021-09-14_14-45-50.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack1_goal2_2021-09-14_14-47-38.npz",
]
STACK1_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack1_goal3_2021-09-11_18-29-26.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack1_goal3_2021-09-14_14-49-50.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack1_goal3_2021-09-14_14-51-48.npz",
]

STACK2_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack2_goal1_2021-09-11_18-22-04.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack2_goal1_2021-09-14_14-34-03.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack2_goal1_2021-09-14_14-36-59.npz",
]
STACK2_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack2_goal2_2021-09-11_18-13-28.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack2_goal2_2021-09-14_14-22-55.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack2_goal2_2021-09-14_14-25-27.npz",
]
STACK2_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack2_goal3_2021-09-11_18-17-15.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack2_goal3_2021-09-14_14-28-16.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack2_goal3_2021-09-14_14-31-14.npz",
]

STACK3_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack3_goal1_2021-09-11_18-07-38.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack3_goal1_2021-09-14_14-03-12.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack3_goal1_2021-09-14_14-08-15.npz",
]
STACK3_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack3_goal2_2021-09-11_17-58-57.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack3_goal2_2021-09-14_13-41-31.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack3_goal2_2021-09-14_13-45-52.npz",
]
STACK3_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/stack/stack3_goal3_2021-09-11_18-03-15.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack3_goal3_2021-09-14_13-54-59.npz",
    DATA_DRIVE_PATH / "new-multi-object/stack/stack3_goal3_2021-09-14_13-59-04.npz",
]

FLAT1_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups1_goal1_2021-09-11_03-12-01.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat1_goal1_2021-09-14_20-35-42.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat1_goal1_2021-09-14_20-38-33.npz",
]
FLAT1_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups1_goal2_2021-09-11_03-06-55.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat1_goal2_2021-09-14_20-26-23.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat1_goal2_2021-09-14_20-28-22.npz",
]
FLAT1_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups1_goal3_2021-09-11_03-09-49.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat1_goal3_2021-09-14_20-31-53.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat1_goal3_2021-09-14_20-33-52.npz",
]

FLAT2_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups2_goal1_2021-09-11_03-00-36.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat2_goal1_2021-09-14_20-09-13.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat2_goal1_2021-09-14_20-13-05.npz",
]
FLAT2_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups2_goal2_2021-09-11_02-53-28.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat2_goal2_2021-09-14_19-54-43.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat2_goal2_2021-09-14_19-57-47.npz",
]
FLAT2_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups2_goal3_2021-09-11_02-56-58.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat2_goal3_2021-09-14_20-00-58.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat2_goal3_2021-09-14_20-04-40.npz",
]

FLAT3_GOAL1_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups3_goal1_2021-09-11_02-47-29.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat3_goal1_2021-09-14_19-37-18.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat3_goal1_2021-09-14_19-41-23.npz",
]
FLAT3_GOAL2_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups3_goal2_2021-09-11_02-38-10.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat3_goal2_2021-09-14_15-27-02.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat3_goal2_2021-09-14_15-31-27.npz",
]
FLAT3_GOAL3_PATHS = [
    DATA_DRIVE_PATH / "multi-object/cups/cups3_goal3_2021-09-11_02-42-52.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat3_goal3_2021-09-14_15-47-26.npz",
    DATA_DRIVE_PATH / "new-multi-object/flat/flat3_goal3_2021-09-14_15-51-34.npz",
]

FIG_PATH = "/home/adam/phd/papers/icra22/figures/controller_time_box_all.pdf"
# FIG_PATH = "controller_time_box.pdf"

RECORD_PERIOD = 10
CTRL_PERIOD = 50


def compute_object_error(ts, r_ew_ws, Q_wes, r_ow_ws, Q_wos, tf=None):
    data_length = ts.shape[0]
    if tf is not None:
        for i in range(ts.shape[0]):
            if ts[i] > tf:
                data_length = i
                break

    r_oe_es = np.zeros_like(r_ow_ws)
    for i in range(r_oe_es.shape[0]):
        C_ew = SO3.from_quaternion(Q_wes[i, :], ordering="xyzw").inv()
        r_oe_es[i, :] = C_ew.dot(r_ow_ws[i, :] - r_ew_ws[i, :])
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    r_err_norm = np.linalg.norm(r_oe_e_err, axis=1)
    return np.max(r_err_norm[:data_length])


def compute_max_object_error(paths_of_paths, tf):
    errs = []
    for paths in paths_of_paths:
        for path in paths:
            with np.load(path) as data:
                ts = data["ts"]
                r_ew_ws = data["r_ew_ws"]
                Q_wes = data["Q_wes"]
                r_ow_ws = data["r_ow_ws"]
                Q_wos = data["Q_wos"]
                for i in range(r_ow_ws.shape[0]):
                    err = compute_object_error(
                        ts, r_ew_ws, Q_wes, r_ow_ws[i, :, :], Q_wos[i, :, :], tf
                    )
                    errs.append(err)
    print(f"max error = {np.max(errs) * 1000} mm")


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.control_durations = data["control_durations"]

            self.ts = data["ts"]
            self.r_ew_ws = data["r_ew_ws"]
            self.r_ew_wds = data["r_ew_wds"]

        # r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        # r_ew_w_err_norm = np.linalg.norm(r_ew_w_err, axis=1)

        data_length = self.control_durations.shape[0]
        if tf is not None:
            for i in range(self.ts.shape[0]):
                if self.ts[i] > tf:
                    # we have to translate from record time to control time
                    data_length = int(i * RECORD_PERIOD / CTRL_PERIOD) + 1
                    break

        # for i in range(r_ew_w_err_norm.shape[0]):
        #     if r_ew_w_err_norm[i] <= 0.01:
        #         print(f"convergence time = {self.ts[i]} s")
        #         data_length = int(i * RECORD_PERIOD / CTRL_PERIOD) + 1
        #         break
        # print(f"data_length = {data_length}")

        # print(f"avg all = {np.mean(self.control_durations) * 1000} ms")
        # print(f"avg all no first = {np.mean(self.control_durations[1:]) * 1000} ms")
        # print(f"avg cut = {np.mean(self.control_durations[:data_length]) * 1000} ms")
        # print(
        #     f"avg cut no first = {np.mean(self.control_durations[1:data_length]) * 1000} ms"
        # )

        self.control_durations_cut = self.control_durations[1:data_length] * 1000
        self.avg_control_time = np.mean(self.control_durations_cut)  # in ms


def set_box_colors(bp, colors):
    plt.setp(bp["boxes"][0], color=colors[0])
    plt.setp(bp["caps"][0], color=colors[0])
    plt.setp(bp["caps"][1], color=colors[0])
    plt.setp(bp["whiskers"][0], color=colors[0])
    plt.setp(bp["whiskers"][1], color=colors[0])
    plt.setp(bp["medians"][0], color=colors[0])

    for flier in bp["fliers"]:
        plt.setp(flier, markersize=1)

    plt.setp(bp["boxes"][1], color=colors[1])
    plt.setp(bp["caps"][2], color=colors[1])
    plt.setp(bp["caps"][3], color=colors[1])
    plt.setp(bp["whiskers"][2], color=colors[1])
    plt.setp(bp["whiskers"][3], color=colors[1])
    plt.setp(bp["medians"][1], color=colors[1])

    plt.setp(bp["boxes"][2], color=colors[2])
    plt.setp(bp["caps"][4], color=colors[2])
    plt.setp(bp["caps"][5], color=colors[2])
    plt.setp(bp["whiskers"][4], color=colors[2])
    plt.setp(bp["whiskers"][5], color=colors[2])
    plt.setp(bp["medians"][2], color=colors[2])


def build_data_from_paths(goal1_paths, goal2_paths, goal3_paths, tf):
    return [
        np.concatenate(
            [TrajectoryData(path, tf=tf).control_durations_cut for path in goal1_paths]
        ),
        np.concatenate(
            [TrajectoryData(path, tf=tf).control_durations_cut for path in goal2_paths]
        ),
        np.concatenate(
            [TrajectoryData(path, tf=tf).control_durations_cut for path in goal3_paths]
        ),
    ]


def main():
    # load the data
    # tray_only_data = [TrajectoryData(path, tf=4) for path in TRAY_ONLY_PATHS]
    tf = 4

    tray_only_data = build_data_from_paths(
        TRAY_ONLY_GOAL1_PATHS, TRAY_ONLY_GOAL2_PATHS, TRAY_ONLY_GOAL3_PATHS, tf
    )
    stack1_data = build_data_from_paths(
        STACK1_GOAL1_PATHS, STACK1_GOAL2_PATHS, STACK1_GOAL3_PATHS, tf
    )
    stack2_data = build_data_from_paths(
        STACK2_GOAL1_PATHS, STACK2_GOAL2_PATHS, STACK2_GOAL3_PATHS, tf
    )
    stack3_data = build_data_from_paths(
        STACK3_GOAL1_PATHS, STACK3_GOAL2_PATHS, STACK3_GOAL3_PATHS, tf
    )

    flat1_data = build_data_from_paths(
        FLAT1_GOAL1_PATHS, FLAT1_GOAL2_PATHS, FLAT1_GOAL3_PATHS, tf
    )
    flat2_data = build_data_from_paths(
        FLAT2_GOAL1_PATHS, FLAT2_GOAL2_PATHS, FLAT2_GOAL3_PATHS, tf
    )
    flat3_data = build_data_from_paths(
        FLAT3_GOAL1_PATHS, FLAT3_GOAL2_PATHS, FLAT3_GOAL3_PATHS, tf
    )

    # compute_max_object_error(
    #     [
    #         TRAY_ONLY_GOAL1_PATHS,
    #         TRAY_ONLY_GOAL2_PATHS,
    #         TRAY_ONLY_GOAL3_PATHS,
    #         STACK1_GOAL1_PATHS,
    #         STACK1_GOAL2_PATHS,
    #         STACK1_GOAL3_PATHS,
    #         STACK2_GOAL1_PATHS,
    #         STACK2_GOAL2_PATHS,
    #         STACK2_GOAL3_PATHS,
    #         STACK3_GOAL1_PATHS,
    #         STACK3_GOAL2_PATHS,
    #         STACK3_GOAL3_PATHS,
    #         FLAT1_GOAL1_PATHS,
    #         FLAT1_GOAL2_PATHS,
    #         FLAT1_GOAL3_PATHS,
    #         FLAT2_GOAL1_PATHS,
    #         FLAT2_GOAL2_PATHS,
    #         FLAT2_GOAL3_PATHS,
    #         FLAT3_GOAL1_PATHS,
    #         FLAT3_GOAL2_PATHS,
    #         FLAT3_GOAL3_PATHS,
    #     ],
    #     tf,
    # )
    # return

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig = plt.figure(figsize=(3.25, 1.5))
    # fig, ax = plt.subplots(1, 2, sharex=True, figsize=(3.25, 1.5))
    plt.rcParams.update(
        {"font.size": 8, "text.usetex": True, "legend.fontsize": 8, "axes.titlesize": 8}
    )

    # plot for objects in a flat configuration
    ax1 = plt.subplot(121)
    width = 0.2
    offset = np.array([-width, 0, width])
    bp1 = plt.boxplot(tray_only_data, positions=1 + offset)
    set_box_colors(bp1, colors)

    bp2 = plt.boxplot(flat1_data, positions=2 + offset)
    set_box_colors(bp2, colors)

    bp3 = plt.boxplot(flat2_data, positions=3 + offset)
    set_box_colors(bp3, colors)

    bp4 = plt.boxplot(flat3_data, positions=4 + offset)
    set_box_colors(bp4, colors)
    plt.ylabel(r"$\mathrm{Computation}$" "\n" r"$\mathrm{time\ (ms)}$")
    plt.title(r"$\mathrm{Flat}$")
    ylim = [5, 70]
    yticks = [16, 20, 24, 28]
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels([1, 2, 3, 4])
    # ax1.set_yticks(yticks)
    plt.ylim(ylim)

    # plot for objects stacked atop one another
    ax2 = plt.subplot(122)
    bp1 = plt.boxplot(tray_only_data, positions=1 + offset)
    set_box_colors(bp1, colors)

    bp2 = plt.boxplot(stack1_data, positions=2 + offset)
    set_box_colors(bp2, colors)

    bp3 = plt.boxplot(stack2_data, positions=3 + offset)
    set_box_colors(bp3, colors)

    bp4 = plt.boxplot(stack3_data, positions=4 + offset)
    set_box_colors(bp4, colors)

    plt.title(r"$\mathrm{Stacked}$")
    # ax2.set_yticks(yticks)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels([1, 2, 3, 4])
    plt.xlabel("N")
    plt.ylim(ylim)
    ax2.xaxis.label.set_color((0, 0, 0, 0))

    goal1_patch = mpatches.Patch(color=colors[0], label=r"$\mathrm{Goal\ 1}$")
    goal2_patch = mpatches.Patch(color=colors[1], label=r"$\mathrm{Goal\ 2}$")
    goal3_patch = mpatches.Patch(color=colors[2], label=r"$\mathrm{Goal\ 3}$")
    handles = [goal1_patch, goal2_patch, goal3_patch]

    plt.legend(
        handles=handles,
        loc=(0.5, 0.54),
        labelspacing=0.3,
        borderpad=0.3,
        handlelength=1,
    )

    # manually specify the common x-label
    fig.text(0.5, 0.04, r"$\mathrm{Number\ of\ objects}$", ha="center", va="center")

    fig.tight_layout(pad=0.1, w_pad=0.5)
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
