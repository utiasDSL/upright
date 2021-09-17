# This script plots the EE distance (to the goal) vs. time for the single
# object of varying heights.
import numpy as np
import matplotlib.pyplot as plt

from mm_pybullet_sim.recording import DATA_DRIVE_PATH
import mm_pybullet_sim.util as util
from liegroups import SO3

import IPython

# DATA_PATH = DATA_DRIVE_PATH / "dynamic_obs_photo_op_2021-09-13_20-19-18.npz"
# CTRL_PERIOD = 100
# TF = 4

DATA_PATH = DATA_DRIVE_PATH / "static_obs_2021-09-13_02-25-11.npz"
CTRL_PERIOD = 50
TF = 10

RECORD_PERIOD = 10


class TrajectoryData:
    def __init__(self, path, tf=None):
        with np.load(path) as data:
            self.control_durations = data["control_durations"][:-1]  # last element is zero?

            self.ts = data["ts"]
            self.r_ew_ws = data["r_ew_ws"]
            self.r_ew_wds = data["r_ew_wds"]

        r_ew_w_err = self.r_ew_wds - self.r_ew_ws
        r_ew_w_err_norm = np.linalg.norm(r_ew_w_err, axis=1)

        data_length = self.control_durations.shape[0]
        record_data_length = self.ts.shape[0]
        if tf is not None:
            for i in range(self.ts.shape[0]):
                if self.ts[i] > tf:
                    # we have to translate from record time to control time
                    data_length = int(i * RECORD_PERIOD / CTRL_PERIOD) + 1
                    record_data_length = i
                    break
        print(f"data_length = {data_length}")

        print(f"avg all = {np.mean(self.control_durations) * 1000} ms")
        print(f"avg all no first = {np.mean(self.control_durations[1:]) * 1000} ms")
        print(f"avg cut = {np.mean(self.control_durations[:data_length]) * 1000} ms")
        print(
            f"avg cut no first = {np.mean(self.control_durations[1:data_length]) * 1000} ms"
        )

        self.ts_cut = self.ts[slice(0, record_data_length, int(CTRL_PERIOD / RECORD_PERIOD))]
        self.control_durations_cut = self.control_durations[:data_length] * 1000
        self.avg_control_time = (
            np.mean(self.control_durations[1:data_length]) * 1000
        )  # in ms


def main():
    # load the data
    data = TrajectoryData(DATA_PATH, tf=TF)

    plt.figure()
    plt.plot(data.ts_cut, data.control_durations_cut)
    plt.xlabel("Time (s)")
    plt.ylabel("Control duration (ms)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
