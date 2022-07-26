import sys
from functools import partial

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from scipy import signal, optimize

import IPython


JOINT_INDEX_MAP = {
    "ur10_arm_shoulder_pan_joint": 0,
    "ur10_arm_shoulder_lift_joint": 1,
    "ur10_arm_elbow_joint": 2,
    "ur10_arm_wrist_1_joint": 3,
    "ur10_arm_wrist_2_joint": 4,
    "ur10_arm_wrist_3_joint": 5,
}

BAG_DIR = "/media/adam/Data/PhD/Data/upright/real-thing/bags/2022-07-25/velocity_test"
BAG_PATHS = [BAG_DIR + "/" + path for path in [
    "velocity_test_joint0_v0.5_2022-07-25-15-32-43.bag",
    "velocity_test_joint1_v0.5_2022-07-25-15-39-15.bag",
    "velocity_test_joint2_v0.5_2022-07-25-15-46-14.bag",
    "velocity_test_joint3_v0.5_2022-07-25-15-55-46.bag",
    "velocity_test_joint4_v0.5_2022-07-25-16-01-38.bag",
    "velocity_test_joint5_v0.5_2022-07-25-16-04-08.bag",
]]


def msg_time(msg):
    """Extract message timestamp as float in seconds."""
    return msg.header.stamp.to_sec()


def parse_time(msgs, normalize_time=True, t0=None):
    """Parse time in seconds from a list of messages.

    If normalize_time is True (the default), the array of time values will be
    normalized such that t[0] = t0. If t0 is not provided, it defaults to t[0].
    """
    t = np.array([msg_time(msg) for msg in msgs])
    if normalize_time:
        if t0:
            t -= t0
        else:
            t -= t[0]
    return t


def parse_feedback_msgs(feedback_msgs):
    """Parse feedback messages.

    This involves reordering the joints in the correct order.

    Returns:
        ts  times
        qs  joint positions
        vs  joint velocities
    """
    ts = parse_time(feedback_msgs, normalize_time=False)
    qs_unordered = np.array([msg.position for msg in feedback_msgs])
    vs_unordered = np.array([msg.velocity for msg in feedback_msgs])

    qs = np.zeros_like(qs_unordered)
    vs = np.zeros_like(vs_unordered)

    joint_names = feedback_msgs[0].name

    # re-order joint names so the names correspond to indices given in
    # JOINT_INDEX_MAP
    for i in range(qs.shape[1]):
        j = JOINT_INDEX_MAP[joint_names[i]]
        qs[:, j] = qs_unordered[:, i]
        vs[:, j] = vs_unordered[:, i]

    return ts, qs, vs


def trim_msgs(msgs, t0=None, t1=None):
    """Trim messages that so only those in the time interval [t0, t1] are included."""
    ts = parse_time(msgs, normalize_time=False)
    start = 0
    if t0 is not None:
        for i in range(ts.shape[0]):
            if ts[i] >= t0:
                start = i
                break

    end = ts.shape[0]
    if t1 is not None:
        for i in range(start, ts.shape[0]):
            if ts[i] > t1:
                end = i
                break

    return msgs[start:end]


def parse_velocity_data(feedback_msgs, cmd_msgs, cmd_ts, idx):
    t0 = cmd_ts[0] - 0.1  # small extra buffer before commands start
    t1 = cmd_ts[-1]
    feedback_msgs = trim_msgs(feedback_msgs, t0=t0, t1=t1)
    ts, _, vs = parse_feedback_msgs(feedback_msgs)

    # all of the commands should be the same
    cmd_value = cmd_msgs[0].data[idx]

    us = np.zeros_like(ts)
    for i in range(ts.shape[0]):
        us[i] = 0 if ts[i] < cmd_ts[0] else cmd_value

    # normalize time so that ts[0] = 0
    ts -= ts[0]

    # take only the velocity from joint idx
    vs = vs[:, idx]

    return ts, us, vs


# system identification routines adapted from
# https://medium.com/robotics-devs/system-identification-with-python-2079088b4d03
def simulate_second_order_system(ts, k, ωn, ζ, us):
    """Simulate a second-order system with parameters (k, ωn, ζ) and inputs us at times ts."""
    sys = signal.TransferFunction(k * (ωn ** 2), [1, 2 * ζ * ωn, ωn ** 2])
    _, ys, _ = signal.lsim2(sys, U=us, T=ts)
    return ys


def identify_second_order_system(ts, us, ys, method="trf", p0=[1.0, 10.0, 0.1]):
    """Fit a second-order model to the inputs us and outputs ys at times ts."""
    # bounds: assume system is not overdamped
    bounds = ([0, 0, 0], [np.inf, np.inf, 1.0])
    model = partial(simulate_second_order_system, us=us)
    (k, ωn, ζ), covariance = optimize.curve_fit(
        model, ts, ys, method=method, p0=p0, bounds=bounds,
    )
    return k, ωn, ζ


def process_one_bag(path, joint_idx):
    bag = rosbag.Bag(path)

    feedback_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_joint_states")
    ]

    cmd_msgs = [msg for _, msg, _ in bag.read_messages("/mobile_manipulator_cmd_vel")]

    # get times of the command messagse, since these do not contain a timestamp
    cmd_ts = np.array(
        [t.to_sec() for _, _, t in bag.read_messages("/mobile_manipulator_cmd_vel")]
    )

    ts, us, ys_actual = parse_velocity_data(
        feedback_msgs, cmd_msgs, cmd_ts, idx=joint_idx
    )

    # fit a second-order model to the data
    k, ωn, ζ = identify_second_order_system(ts, us, ys_actual)
    ys_fit = simulate_second_order_system(ts, k, ωn, ζ, us)

    # make sure all steps are in positive direction for consistency
    if us[-1] < 0:
        us = -us
        ys_actual = -ys_actual
        ys_fit = -ys_fit

    plt.figure()

    # actual output
    plt.plot(ts, ys_actual, label="Actual")

    # commands
    plt.plot(ts, us, linestyle="--", label="Commanded")

    # fitted model
    plt.plot(ts, ys_fit, color="k", label="Model")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.legend()
    plt.grid()
    plt.title(f"Joint {joint_idx} Velocity")


def main():
    for i, path in enumerate(BAG_PATHS):
        if i == 4:
            process_one_bag(path, i)
    plt.show()


if __name__ == "__main__":
    main()
