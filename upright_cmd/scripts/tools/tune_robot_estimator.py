#!/usr/bin/env python3
"""Tune Kalman filter for estimation of robot state."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils

import IPython


ROBOT_PROC_VAR = 1000
ROBOT_MEAS_VAR = 0.0001


def parse_mpc_observation_msgs(msgs, normalize_time=True):
    ts = []
    xs = []
    us = []

    for msg in msgs:
        ts.append(msg.time)
        xs.append(msg.state.value)
        us.append(msg.input.value)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(xs), np.array(us)


class GaussianEstimate:
    def __init__(self, x, P):
        self.x = x
        self.P = P


def kf_predict(estimate, A, Q, v):
    x = A @ estimate.x + v
    P = A @ estimate.P @ A.T + Q
    return GaussianEstimate(x, P)


def kf_correct(estimate, C, R, y):
    # Innovation covariance
    CP = C @ estimate.P
    S = CP @ C.T + R

    # Correct using measurement model
    P = estimate.P - CP.T @ np.linalg.solve(S, CP)
    x = estimate.x + CP.T @ np.linalg.solve(S, y - C @ estimate.x)
    return GaussianEstimate(x, P)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    ur10_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/joint_states")]
    ridgeback_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]
    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ]

    tas, qas, vas = ros_utils.parse_ur10_joint_state_msgs(
        ur10_msgs, normalize_time=False
    )
    tbs, qbs, vbs = ros_utils.parse_ridgeback_joint_state_msgs(
        ridgeback_msgs, normalize_time=False
    )
    tms, xms, ums = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=False)

    n = 9

    ts = tas
    qbs_aligned = ros_utils.interpolate_list(ts, tbs, qbs)
    vbs_aligned = ros_utils.interpolate_list(ts, tbs, vbs)

    # observations
    qs = np.hstack((qbs_aligned, qas))
    vs = np.hstack((vbs_aligned, vas))

    # inputs
    us = np.array(ros_utils.interpolate_list(ts, tms, ums))

    # integrated state (assumes perfect model)
    xms_aligned = np.array(ros_utils.interpolate_list(ts, tms, xms))
    vs_int = xms_aligned[:, n : 2 * n]

    ts -= ts[0]

    # initial state
    x0 = np.concatenate((qs[0, :], np.zeros(2 * n)))
    estimate = GaussianEstimate(x0, np.eye(x0.shape[0]))

    I = np.eye(n)
    Z = np.zeros((n, n))
    C = np.hstack((I, Z, Z))

    # noise covariance
    Q0 = ROBOT_PROC_VAR * I
    R = ROBOT_MEAS_VAR * I

    # do estimation using the Kalman filter
    xs_est = [x0]
    for i in range(1, ts.shape[0]):
        dt = ts[i] - ts[i - 1]

        A = np.block([[I, dt * I, 0.5 * dt * dt * I], [Z, I, dt * I], [Z, Z, I]])
        B = np.vstack((dt * dt * dt * I / 6, 0.5 * dt * dt * I, dt * I))
        Q = B @ Q0 @ B.T
        u = us[i, :n]

        estimate = kf_predict(estimate, A, Q, B @ u)
        estimate = kf_correct(estimate, C, R, qs[i, :])
        xs_est.append(estimate.x.copy())

    xs_est = np.array(xs_est)
    qs_est = xs_est[:, :n]
    vs_est = xs_est[:, n : 2 * n]

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs[:, i], label=f"q_{i+1}")
    plt.title("Measured Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_est[:, i], label=f"q_{i+1}")
    plt.title("Estimated Joint Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs[:, i], label=f"v_{i+1}")
    plt.title("Measured Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_est[:, i], label=f"v_{i+1}")
    plt.title("Estimated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_int[:, i], label=f"v_{i+1}")
    plt.title("Integrated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs[:, i] - vs_est[:, i], label=f"v_{i+1}")
    plt.title("Measured - Estimated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs[:, i] - vs_int[:, i], label=f"v_{i+1}")
    plt.title("Measured - Integrated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
