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

TRAY_ONLY_PATHS = [
    DATA_PATH / filename
    for filename in (
        "tray_only_goal1_2021-09-11_18-38-39.npz",
        "tray_only_goal2_2021-09-11_18-35-52.npz",
        "tray_only_goal3_2021-09-11_18-37-19.npz",
    )
]

STACK1_PATHS = [
    DATA_PATH / "stack" / filename
    for filename in (
        "stack1_goal1_2021-09-11_18-31-16.npz",
        "stack1_goal2_2021-09-11_18-27-08.npz",
        "stack1_goal3_2021-09-11_18-29-26.npz",
    )
]
STACK2_PATHS = [
    DATA_PATH / "stack" / filename
    for filename in (
        "stack2_goal1_2021-09-11_18-22-04.npz",
        "stack2_goal2_2021-09-11_18-13-28.npz",
        "stack2_goal3_2021-09-11_18-17-15.npz",
    )
]
STACK3_PATHS = [
    DATA_PATH / "stack" / filename
    for filename in (
        "stack3_goal1_2021-09-11_18-07-38.npz",
        "stack3_goal2_2021-09-11_17-58-57.npz",
        "stack3_goal3_2021-09-11_18-03-15.npz",
    )
]

STACK2_GOAL2_PATHS_EXTRA = [
    DATA_DRIVE_PATH / filename
    for filename in (
        "stack2_goal2_2021-09-11_19-28-32.npz",
        "stack2_goal2_2021-09-11_19-31-24.npz",
    )
]

CUPS1_PATHS = [
    DATA_PATH / "cups" / filename
    for filename in (
        "cups1_goal1_2021-09-11_03-12-01.npz",
        "cups1_goal2_2021-09-11_03-06-55.npz",
        "cups1_goal3_2021-09-11_03-09-49.npz",
    )
]
CUPS2_PATHS = [
    DATA_PATH / "cups" / filename
    for filename in (
        "cups2_goal1_2021-09-11_03-00-36.npz",
        "cups2_goal2_2021-09-11_02-53-28.npz",
        "cups2_goal3_2021-09-11_02-56-58.npz",
    )
]
CUPS3_PATHS = [
    DATA_PATH / "cups" / filename
    for filename in (
        "cups3_goal1_2021-09-11_02-47-29.npz",
        "cups3_goal2_2021-09-11_02-38-10.npz",
        "cups3_goal3_2021-09-11_02-42-52.npz",
    )
]

FIG_PATH = "/home/adam/phd/papers/icra22/figures/controller_time_box.pdf"
# FIG_PATH = "controller_time_box.pdf"

RECORD_PERIOD = 10
CTRL_PERIOD = 50


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.control_durations = data["control_durations"]

            self.ts = data["ts"]
            self.r_ew_ws = data["r_ew_ws"]
            self.r_ew_wds = data["r_ew_wds"]

        r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        r_ew_w_err_norm = np.linalg.norm(r_ew_w_err, axis=1)

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
    # try:
    #     plt.setp(bp['fliers'][0], markeredgecolor='blue')
    #     plt.setp(bp['fliers'][1], markeredgecolor='blue')
    # except IndexError:
    #     pass
    plt.setp(bp["medians"][0], color=colors[0])

    for flier in bp["fliers"]:
        plt.setp(flier, markersize=1)

    plt.setp(bp["boxes"][1], color=colors[1])
    plt.setp(bp["caps"][2], color=colors[1])
    plt.setp(bp["caps"][3], color=colors[1])
    plt.setp(bp["whiskers"][2], color=colors[1])
    plt.setp(bp["whiskers"][3], color=colors[1])
    # try:
    #     plt.setp(bp['fliers'][2], markeredgecolor='red')
    #     plt.setp(bp['fliers'][3], markeredgecolor='red')
    # except IndexError:
    #     pass
    plt.setp(bp["medians"][1], color=colors[1])

    plt.setp(bp["boxes"][2], color=colors[2])
    plt.setp(bp["caps"][4], color=colors[2])
    plt.setp(bp["caps"][5], color=colors[2])
    plt.setp(bp["whiskers"][4], color=colors[2])
    plt.setp(bp["whiskers"][5], color=colors[2])
    plt.setp(bp["medians"][2], color=colors[2])


def main():
    # load the data
    tray_only_data = [TrajectoryData(path, tf=4) for path in TRAY_ONLY_PATHS]

    # NOTE: I renamed this to "flat" in the paper
    cups1_data = [TrajectoryData(path, tf=4) for path in CUPS1_PATHS]
    cups2_data = [TrajectoryData(path, tf=4) for path in CUPS2_PATHS]
    cups3_data = [TrajectoryData(path, tf=4) for path in CUPS3_PATHS]

    stack1_data = [TrajectoryData(path, tf=4) for path in STACK1_PATHS]
    stack2_data = [TrajectoryData(path, tf=4) for path in STACK2_PATHS]
    stack3_data = [TrajectoryData(path, tf=4) for path in STACK3_PATHS]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    ### Flat Configuration ###

    # plt.figure()
    # # goal 1 data = index 0
    # ax = plt.gca()
    # width = 0.2
    # offset = np.array([-width, 0, width])
    # bp1 = plt.boxplot(
    #     [data.control_durations_cut for data in tray_only_data], positions=1 + offset
    # )
    # set_box_colors(bp1, colors)
    #
    # bp2 = plt.boxplot(
    #     [data.control_durations_cut for data in cups1_data], positions=2 + offset
    # )
    # set_box_colors(bp2, colors)
    #
    # bp3 = plt.boxplot(
    #     [data.control_durations_cut for data in cups2_data], positions=3 + offset,
    # )
    # set_box_colors(bp3, colors)
    #
    # bp4 = plt.boxplot(
    #     [data.control_durations_cut for data in cups3_data], positions=4 + offset,
    # )
    # set_box_colors(bp4, colors)
    #
    # plt.xlabel("Number of objects")
    # ax.set_xticks([1, 2, 3, 4])
    # ax.set_xticklabels([1, 2, 3, 4])
    # plt.show()
    # return

    fig = plt.figure(figsize=(3.25, 1.5))
    # fig, ax = plt.subplots(1, 2, sharex=True, figsize=(3.25, 1.5))
    plt.rcParams.update(
        {"font.size": 8, "text.usetex": True, "legend.fontsize": 8, "axes.titlesize": 8}
    )

    # plot for objects in a flat configuration
    ax1 = plt.subplot(121)
    width = 0.2
    offset = np.array([-width, 0, width])
    bp1 = plt.boxplot(
        [data.control_durations_cut for data in tray_only_data], positions=1 + offset
    )
    set_box_colors(bp1, colors)

    bp2 = plt.boxplot(
        [data.control_durations_cut for data in cups1_data], positions=2 + offset
    )
    set_box_colors(bp2, colors)

    bp3 = plt.boxplot(
        [data.control_durations_cut for data in cups2_data],
        positions=3 + offset,
    )
    set_box_colors(bp3, colors)

    bp4 = plt.boxplot(
        [data.control_durations_cut for data in cups3_data],
        positions=4 + offset,
    )
    set_box_colors(bp4, colors)
    plt.ylabel(r"$\mathrm{Computation}$" "\n" r"$\mathrm{time\ (ms)}$")
    plt.title(r"$\mathrm{Flat}$")
    ylim = [10, 70]
    yticks = [16, 20, 24, 28]
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels([1, 2, 3, 4])
    # ax1.set_yticks(yticks)
    plt.ylim(ylim)

    # plot for objects stacked atop one another
    ax2 = plt.subplot(122)
    bp1 = plt.boxplot(
        [data.control_durations_cut for data in tray_only_data], positions=1 + offset
    )
    set_box_colors(bp1, colors)

    bp2 = plt.boxplot(
        [data.control_durations_cut for data in stack1_data], positions=2 + offset
    )
    set_box_colors(bp2, colors)

    bp3 = plt.boxplot(
        [data.control_durations_cut for data in stack2_data],
        positions=3 + offset,
    )
    set_box_colors(bp3, colors)

    bp4 = plt.boxplot(
        [data.control_durations_cut for data in stack3_data],
        positions=4 + offset,
    )
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
