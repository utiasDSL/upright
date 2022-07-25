import numpy as np
import sys
import rosbag
import matplotlib.pyplot as plt

import IPython


# TODO this stuff can do in the ros_interface library

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
        plt.plot(ts, xs[:, 12+i], label=f"a_{i}")
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


def main():
    bag = rosbag.Bag(sys.argv[1])

    feedback_msgs = [msg for _, msg, _ in bag.read_messages("/mobile_manipulator_joint_states")]
    cmd_msgs = [msg for _, msg, _ in bag.read_messages("/mobile_manipulator_cmd_vel")]

    # this only uses real wall time, not simulated time, so is not accurate for
    # simulation
    cmd_ts = np.array([t.to_sec() for _, _, t in bag.read_messages("/mobile_manipulator_cmd_vel")])

    mpc_msgs = [msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")]

    plot_feedback(feedback_msgs)
    # plot_cmds(cmd_ts, cmd_msgs)
    # plot_mpc_observations(mpc_msgs)

    plt.show()


if __name__ == "__main__":
    main()
