import numpy as np
import sys
import rosbag
import matplotlib.pyplot as plt

import IPython


# TODO this stuff can do in the ros_interface library

JOINT_INDEX_MAP = {
    "ur10_arm_shoulder_pan_joint": 0,
    "ur10_arm_shoulder_lift_joint": 1,
    "ur10_arm_elbow_joint": 2,
    "ur10_arm_wrist_1_joint": 3,
    "ur10_arm_wrist_2_joint": 4,
    "ur10_arm_wrist_3_joint": 5,
}


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


def plot_mpc_observations(mpc_msgs):
    ts = np.array([msg.time for msg in mpc_msgs])
    ts -= ts[0]  # normalize time

    xs = np.array([msg.state.value for msg in mpc_msgs])
    us = np.array([msg.input.value for msg in mpc_msgs])

    plt.figure()
    plt.grid()
    for i in range(us.shape[1]):
        plt.plot(ts, xs[:, 12 + i], label=f"a_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint acceleration (rad/s^2)")
    plt.legend()
    plt.title("Joint acceleration")

    plt.figure()
    plt.grid()
    for i in range(us.shape[1]):
        plt.plot(ts, us[:, i], label=f"u_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint jerk input (rad/s^3)")
    plt.legend()
    plt.title("Joint jerk inputs")


def plot_feedback(feedback_msgs):
    ts = parse_time(feedback_msgs)
    qs = np.array([msg.position for msg in feedback_msgs])
    vs = np.array([msg.velocity for msg in feedback_msgs])

    plt.figure()
    plt.grid()
    for i in range(qs.shape[1]):
        plt.plot(ts, qs[:, i], label=f"q_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position (rad)")
    plt.legend()
    plt.title("Joint positions")

    plt.figure()
    plt.grid()
    for i in range(vs.shape[1]):
        plt.plot(ts, vs[:, i], label=f"v_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.legend()
    plt.title("Joint velocities")


def plot_cmds(cmd_ts, cmd_msgs):
    cmds = np.array([msg.data for msg in cmd_msgs])

    cmd_ts = cmd_ts - cmd_ts[0]

    plt.figure()
    plt.grid()
    for i in range(cmds.shape[1]):
        plt.plot(cmd_ts, cmds[:, i], label=f"cmd_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint command (rad/s)")
    plt.legend()
    plt.title("Joint velocity commands")

    plt.figure()
    plt.grid()
    plt.plot(cmd_ts[1:] - cmd_ts[:-1])
    plt.xlabel("Step")
    plt.ylabel("Times (s)")
    plt.title("Command time")


def parse_feedback_msgs(feedback_msgs):
    """Parse feedback messages.

    This involves reordering the joints in the correct order.

    Returns:
        ts  times
        qs  joint positions
        vs  joint velocities
    """
    ts = parse_time(feedback_msgs)
    qs_unordered = np.array([msg.position for msg in feedback_msgs])
    vs_unordered = np.array([msg.velocity for msg in feedback_msgs])

    qs = np.zeros_like(qs_unordered)
    vs = np.zeros_like(vs_unordered)

    joint_names = feedback_msgs[0].name

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


def plot_velocity(feedback_msgs, cmd_msgs, cmd_ts, idx):
    feedback_msgs = trim_msgs(feedback_msgs, t0=cmd_ts[0], t1=cmd_ts[-1])
    ts, _, vs = parse_feedback_msgs(feedback_msgs)

    cmds = np.array([msg.data for msg in cmd_msgs])
    cmd_ts -= cmd_ts[0]

    plt.figure()
    plt.grid()
    plt.plot(ts, vs[:, idx], label=f"v_{idx}")
    plt.plot(cmd_ts[:-1], cmds[:-1, idx], linestyle="--", label=f"v^d_{idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.legend()
    plt.title("Joint velocities")


def main():
    bag = rosbag.Bag(sys.argv[1])

    feedback_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_joint_states")
    ]
    cmd_msgs = [msg for _, msg, _ in bag.read_messages("/mobile_manipulator_cmd_vel")]

    # this only uses real wall time, not simulated time, so is not accurate for
    # simulation
    cmd_ts = np.array(
        [t.to_sec() for _, _, t in bag.read_messages("/mobile_manipulator_cmd_vel")]
    )

    mpc_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ]

    # plot_feedback(feedback_msgs)
    # plot_cmds(cmd_ts, cmd_msgs)
    # plot_mpc_observations(mpc_msgs)

    # plot_velocity(feedback_msgs, cmd_msgs, cmd_ts, idx=0)

    plt.show()


if __name__ == "__main__":
    main()
