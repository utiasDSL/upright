"""Plot position of Vicon objects from a ROS bag."""
import argparse
from scipy import optimize

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils

import IPython


# we only use data above this height
H = 1.11
DRAG = 0.05


def rollout_drag(ts, b, r0, v0, g):
    n = ts.shape[0]
    rs = np.zeros((n, 3))
    vs = np.zeros((n, 3))

    rs[0, :] = r0
    vs[0, :] = v0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        a = g - b * np.linalg.norm(vs[i - 1, :]) * vs[i - 1, :]
        vs[i, :] = vs[i - 1, :] + dt * a
        rs[i, :] = rs[i - 1, :] + dt * vs[i, :]
    return rs, vs


def identify_drag(ts, rs, r0, v0, g, method="trf", p0=[0.01]):
    """Fit a second-order model to the inputs us and outputs ys at times ts."""

    def residuals(b):
        rm, _ = rollout_drag(ts, b, r0, v0, g)
        Δs = np.linalg.norm(rm - rs, axis=1)
        return Δs

    bounds = ([0], [np.inf])
    res = optimize.least_squares(residuals, x0=p0, bounds=bounds)
    return res.x[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    msgs = [msg for _, msg, _ in bag.read_messages("/vicon/Projectile/Projectile")]
    positions = []
    for msg in msgs:
        p = msg.transform.translation
        positions.append([p.x, p.y, p.z])
    positions = np.array(positions)
    times = ros_utils.parse_time(msgs)

    idx = np.flatnonzero(positions[:, 2] >= H)
    rp = positions[idx, :]
    tp = times[idx]

    i = 2
    r0 = rp[0, :]
    v0 = (rp[i, :] - r0) / (tp[i] - tp[0])
    a0 = np.array([0, 0, -9.81])

    tm = (tp - tp[0])[:, None]
    rm = r0 + v0 * tm + 0.5 * tm ** 2 * a0

    b = identify_drag(tp, rp, r0, v0, a0)
    print(f"b = {b}")
    rd, _ = rollout_drag(tp, b, r0, v0, a0)

    plt.figure()
    plt.plot(tp, rp[:, 0], label="x")
    plt.plot(tp, rp[:, 1], label="y")
    plt.plot(tp, rp[:, 2], label="z")
    # plt.plot(tp, rm[:, 0], label="x_m")
    # plt.plot(tp, rm[:, 1], label="y_m")
    # plt.plot(tp, rm[:, 2], label="z_m")
    plt.plot(tp, rd[:, 0], label="x_d")
    plt.plot(tp, rd[:, 1], label="y_d")
    plt.plot(tp, rd[:, 2], label="z_d")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time")
    plt.legend()
    plt.grid()

    # # x, y, z position vs. time
    # plt.figure()
    # plt.plot(times, positions[:, 0], label="x")
    # plt.plot(times, positions[:, 1], label="y")
    # plt.plot(times, positions[:, 2], label="z")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Position (m)")
    # plt.title("Projectile position vs. time")
    # plt.legend()
    # plt.grid()
    #
    # # x, y, z position in 3D space
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.plot(positions[:, 0], positions[:, 1], zs=positions[:, 2])
    # ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], "o", color="g", label="Start")
    # ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], "o", color="r", label="End")
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel("x (m)")
    # ax.set_ylabel("y (m)")
    # ax.set_zlabel("z (m)")
    #
    # ax.set_xlim([-3, 3])
    # ax.set_ylim([-3, 3])

    plt.show()


if __name__ == "__main__":
    main()
