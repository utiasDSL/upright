#!/usr/bin/env python3
"""Play with KF for projectile motion."""
import argparse
from scipy import optimize

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils

import IPython


# we only use data above this height
H = 0.1


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
    dt_nom = 0.01
    R = dt_nom**2 * np.eye(3)

    # acceleration variance
    var_a = 1000

    # (linear) motion model
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    C = np.zeros((3, 6))
    C[:, :3] = np.eye(3)

    # initial state
    xs = np.zeros((n, 6))
    xc = np.concatenate((r0, v0))
    Pc = np.eye(6)  # P0

    xs[0, :] = xc
    Ps = [Pc]

    active = False

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]

        Ai = np.eye(6) + dt * A
        Bi = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))

        # NOTE: doing this is key! (rather than having no off-diagonal elements)
        Qi = var_a * Bi @ Bi.T
        Ri = R

        if rs[i, 2] >= 0.8:
            active = True
        elif rs[i, 2] <= 0.2:
            active = False

        if active:
            u = g
        else:
            u = np.zeros(3)

        # predictor
        xc = Ai @ xs[i - 1, :] + Bi @ u
        Pc = Ai @ Ps[i - 1] @ Ai.T + Qi

        # measurement
        y = rs[i, :]

        # corrector
        K = Pc @ C.T @ np.linalg.inv(C @ Pc @ C.T + Ri)
        P = (np.eye(6) - K @ C) @ Pc
        xs[i, :] = xc + K @ (y - C @ xc)
        Ps.append(P)

    return xs, np.array(Ps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    msgs = [
        msg for _, msg, _ in bag.read_messages("/vicon/ThingVolleyBall/ThingVolleyBall")
    ]
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

    # numerical diff to get velocity
    vn = rollout_numerical_diff(tp, rp, r0, v0, τ=0.0)

    # rollout with Kalman filter
    xk, Pks = rollout_kalman(tp, rp, r0, v0, g)
    rk, vk = xk[:, :3], xk[:, 3:6]

    # Position models

    plt.figure()

    # ground truth
    plt.plot(tp, rp[:, 0], label="x (vicon)", color="r")
    plt.plot(tp, rp[:, 1], label="y (vicon)", color="g")
    plt.plot(tp, rp[:, 2], label="z (vicon)", color="b")

    # Kalman filter (no drag)
    plt.plot(tp, rk[:, 0], label="x (kalman)", color="r", linestyle="--")
    plt.plot(tp, rk[:, 1], label="y (kalman)", color="g", linestyle="--")
    plt.plot(tp, rk[:, 2], label="z (kalman)", color="b", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(tp, Pks[:, 0, 0], label="var(x)", color="r")
    plt.plot(tp, Pks[:, 3, 3], label="var(vx)", color="r", linestyle="--")
    plt.legend()
    plt.grid()

    # Velocity models

    plt.figure()

    # numerically diffed model
    plt.plot(tp, vn[:, 0], label="x (num diff)", color="r")
    plt.plot(tp, vn[:, 1], label="y (num diff)", color="g")
    plt.plot(tp, vn[:, 2], label="z (num diff)", color="b")

    # kalman filter
    plt.plot(tp, vk[:, 0], label="x (kalman)", color="r", linestyle="--")
    plt.plot(tp, vk[:, 1], label="y (kalman)", color="g", linestyle="--")
    plt.plot(tp, vk[:, 2], label="z (kalman)", color="b", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Projectile velocity vs. time")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
