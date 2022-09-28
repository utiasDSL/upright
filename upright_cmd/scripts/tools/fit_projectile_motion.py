#!/usr/bin/env python3
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


def rollout_drag(ts, b, r0, v0, g):
    """Roll out model including a nonlinear drag term with coefficient b."""
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


def rollout_numerical_diff(ts, rs, r0, v0, τ):
    """Rollout velocity using exponential smoothing."""
    n = ts.shape[0]
    vs = np.zeros((n, 3))
    vs[0, :] = v0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        α = 1 - np.exp(-dt / τ)
        v_meas = (rs[i, :] - rs[i - 1, :]) / dt
        vs[i, :] = α * v_meas + (1 - α) * vs[i - 1, :]
    return vs


def rollout_kalman(ts, rs, r0, v0, g):
    """Estimate ball trajectory using a Kalman filter."""
    n = ts.shape[0]

    # noise covariance
    Q = 1e-2 * np.eye(6)
    R = np.diag([1e-2, 1e-2, 1e-2, 1, 1, 1])

    # (linear) motion model
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)

    # initial state
    xs = np.zeros((n, 6))
    xc = np.concatenate((r0, v0))
    Pc = R

    xs[0, :] = xc

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]

        # predictor
        Pc = dt ** 2 * A @ Pc @ A.T + Q
        xc = xs[i - 1, :] + dt * (A @ xs[i - 1, :] + B @ g)

        # velocity portion of R can be directly calculated from the position
        # portion, since velocity is computed directly from position
        R[3:, 3:] = R[0, 0] * 2 * np.eye(3) / dt

        # measurement
        r = rs[i, :]
        v = (rs[i, :] - rs[i - 1, :]) / dt
        y = np.concatenate((r, v))

        # corrector
        K = Pc @ np.linalg.inv(Pc + R)
        P = (np.eye(6) - K) @ Pc
        xs[i, :] = xc + K @ (y - xc)

    return xs


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

    # find portion of the ball undergoing projectile motion
    idx = np.flatnonzero(positions[:, 2] >= H)
    rp = positions[idx, :]
    tp = times[idx]

    # use the first two timesteps to estimate initial state
    r0 = rp[1, :]
    v0 = (rp[1, :] - rp[0, :]) / (tp[1] - tp[0])
    g = np.array([0, 0, -9.81])

    # discard first timestep now that we've "used it up"
    rp = rp[1:, :]
    tp = tp[1:]

    # nominal model (perfect projectile motion)
    tm = (tp - tp[0])[:, None]
    rm = r0 + v0 * tm + 0.5 * tm ** 2 * g
    vm = v0 + tm * g

    # drag model
    b = identify_drag(tp, rp, r0, v0, g)
    print(f"b = {b}")
    rd, vd = rollout_drag(tp, b, r0, v0, g)

    # numerical diff to get velocity
    vn = rollout_numerical_diff(tp, rp, r0, v0, τ=0.05)

    # rollout with Kalman filter
    xk = rollout_kalman(tp, rp, r0, v0, g)
    rk, vk = xk[:, :3], xk[:, 3:]

    # Position models

    plt.figure()

    # ground truth
    plt.plot(tp, rp[:, 0], label="x", color="r")
    plt.plot(tp, rp[:, 1], label="y", color="g")
    plt.plot(tp, rp[:, 2], label="z", color="b")

    # assume perfect projectile motion
    plt.plot(tp, rm[:, 0], label="x_m", color="r", linestyle=":")
    plt.plot(tp, rm[:, 1], label="y_m", color="g", linestyle=":")
    plt.plot(tp, rm[:, 2], label="z_m", color="b", linestyle=":")

    # Kalman filter (no drag)
    plt.plot(tp, rd[:, 0], label="x_k", color="r", linestyle="--")
    plt.plot(tp, rd[:, 1], label="y_k", color="g", linestyle="--")
    plt.plot(tp, rd[:, 2], label="z_k", color="b", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time")
    plt.legend()
    plt.grid()

    # Position model with drag

    plt.figure()

    # ground truth
    plt.plot(tp, rp[:, 0], label="x", color="r")
    plt.plot(tp, rp[:, 1], label="y", color="g")
    plt.plot(tp, rp[:, 2], label="z", color="b")

    # assume perfect projectile motion
    plt.plot(tp, rm[:, 0], label="x_m", color="r", linestyle=":")
    plt.plot(tp, rm[:, 1], label="y_m", color="g", linestyle=":")
    plt.plot(tp, rm[:, 2], label="z_m", color="b", linestyle=":")

    # projectile motion with drag
    plt.plot(tp, rd[:, 0], label="x_d", color="r", linestyle="-.")
    plt.plot(tp, rd[:, 1], label="y_d", color="g", linestyle="-.")
    plt.plot(tp, rd[:, 2], label="z_d", color="b", linestyle="-.")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time: drag")
    plt.legend()
    plt.grid()

    # Velocity models

    plt.figure()
    plt.plot(tp, vm[:, 0], label="x_m", color="r")
    plt.plot(tp, vm[:, 1], label="y_m", color="g")
    plt.plot(tp, vm[:, 2], label="z_m", color="b")
    plt.plot(tp, vn[:, 0], label="x_n", color="r", linestyle=":")
    plt.plot(tp, vn[:, 1], label="y_n", color="g", linestyle=":")
    plt.plot(tp, vn[:, 2], label="z_n", color="b", linestyle=":")
    plt.plot(tp, vk[:, 0], label="x_k", color="r", linestyle="--")
    plt.plot(tp, vk[:, 1], label="y_k", color="g", linestyle="--")
    plt.plot(tp, vk[:, 2], label="z_k", color="b", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Projectile velocity vs. time")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
