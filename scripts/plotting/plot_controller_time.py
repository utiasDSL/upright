# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
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

FIG_PATH = "/home/adam/phd/papers/icra22/figures/controller_time.pdf"

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

        self.avg_control_time = (
            np.mean(self.control_durations[1:data_length]) * 1000
        )  # in ms


def main():
    # load the data
    tray_only_data = [TrajectoryData(path, tf=4) for path in TRAY_ONLY_PATHS]

    cups1_data = [TrajectoryData(path, tf=4) for path in CUPS1_PATHS]
    cups2_data = [TrajectoryData(path, tf=4) for path in CUPS2_PATHS]
    cups3_data = [TrajectoryData(path, tf=4) for path in CUPS3_PATHS]

    stack1_data = [TrajectoryData(path, tf=4) for path in STACK1_PATHS]
    stack2_data = [TrajectoryData(path, tf=4) for path in STACK2_PATHS]
    stack3_data = [TrajectoryData(path, tf=4) for path in STACK3_PATHS]

    # stack2_goal2_data = [
    #     stack2_data[1],
    #     TrajectoryData(STACK2_GOAL2_PATHS_EXTRA[0], tf=4),
    #     TrajectoryData(STACK2_GOAL2_PATHS_EXTRA[1], tf=4),
    # ]
    # for datum in stack2_goal2_data:
    #     print(datum.avg_control_time)

    fig = plt.figure(figsize=(3.25, 1.5))
    # fig, ax = plt.subplots(1, 2, sharex=True, figsize=(3.25, 1.5))
    plt.rcParams.update({"font.size": 8, "text.usetex": True, "legend.fontsize": 8})

    # plot for objects in a flat configuration
    ax1 = plt.subplot(121)
    plt.plot(
        [1, 2, 3, 4],
        [
            tray_only_data[0].avg_control_time,
            cups1_data[0].avg_control_time,
            cups2_data[0].avg_control_time,
            cups3_data[0].avg_control_time,
        ],
        label=r"$\mathrm{Goal\ 1}$",
    )
    plt.plot(
        [1, 2, 3, 4],
        [
            tray_only_data[1].avg_control_time,
            cups1_data[1].avg_control_time,
            cups2_data[1].avg_control_time,
            cups3_data[1].avg_control_time,
        ],
        label=r"$\mathrm{Goal\ 2}$",
    )
    plt.plot(
        [1, 2, 3, 4],
        [
            tray_only_data[2].avg_control_time,
            cups1_data[2].avg_control_time,
            cups2_data[2].avg_control_time,
            cups3_data[2].avg_control_time,
        ],
        label=r"$\mathrm{Goal\ 3}$",
    )
    # plt.xlabel(r"$\mathrm{Number\ of\ Objects}$")
    plt.ylabel(r"$\mathrm{Time\ (ms)}$")
    plt.title(r"$\mathrm{Flat}$")
    ylim = [15, 29]
    yticks = [16, 20, 24, 28]
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_yticks(yticks)
    plt.ylim(ylim)

    # plot for objects stacked atop one another
    ax2 = plt.subplot(122)
    plt.plot(
        [1, 2, 3, 4],
        [
            tray_only_data[0].avg_control_time,
            stack1_data[0].avg_control_time,
            stack2_data[0].avg_control_time,
            stack3_data[0].avg_control_time,
        ],
        label=r"$\mathrm{Goal\ 1}$",
    )
    plt.plot(
        [1, 2, 3, 4],
        [
            tray_only_data[1].avg_control_time,
            stack1_data[1].avg_control_time,
            stack2_data[1].avg_control_time,
            stack3_data[1].avg_control_time,
        ],
        label=r"$\mathrm{Goal\ 2}$",
    )
    plt.plot(
        [1, 2, 3, 4],
        [
            tray_only_data[2].avg_control_time,
            stack1_data[2].avg_control_time,
            stack2_data[2].avg_control_time,
            stack3_data[2].avg_control_time,
        ],
        label=r"$\mathrm{Goal\ 3}$",
    )
    # plt.xlabel(r"$\mathrm{Time\ (ms)}$")
    plt.title(r"$\mathrm{Stacked}$")
    # ax2.set_yticks(yticks)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([1, 2, 3, 4])
    plt.xlabel("N")
    plt.ylim(ylim)
    ax2.xaxis.label.set_color((0, 0, 0, 0))
    plt.legend(loc="upper right", labelspacing=0.3, borderpad=0.3)

    # manually specify the common x-label
    fig.text(0.5, 0.04, r"$\mathrm{Number\ of\ Objects}$", ha="center", va="center")

    fig.tight_layout(pad=0.1, w_pad=0.5)
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
