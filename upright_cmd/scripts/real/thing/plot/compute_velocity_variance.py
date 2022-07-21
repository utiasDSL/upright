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


def plot_feedback(feedback_msgs):
    ts = parse_time(feedback_msgs)
    qs = np.array([msg.position for msg in feedback_msgs])
    vs = np.array([msg.velocity for msg in feedback_msgs])

    for i, v in enumerate(vs):
        if np.linalg.norm(v) > 1e-6:
            start = i
            break

    for i, v in enumerate(vs[::-1]):
        if np.linalg.norm(v) > 1e-6:
            end = vs.shape[0] - i
            break
    std = np.std(vs[start:end], axis=0)
    IPython.embed()

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


def compute_velocity_std(feedback_msgs):
    vs = np.array([msg.velocity for msg in feedback_msgs])

    for i, v in enumerate(vs):
        if np.linalg.norm(v) > 1e-6:
            start = i
            break

    for i, v in enumerate(vs[::-1]):
        if np.linalg.norm(v) > 1e-6:
            end = vs.shape[0] - i
            break

    return np.std(vs[start:end], axis=0)


def main():
    bag = rosbag.Bag(sys.argv[1])

    feedback_msgs = [msg for _, msg, _ in bag.read_messages("/ur10_joint_states")]

    std = compute_velocity_std(feedback_msgs)
    print(f"Standard deviation of true joint velocities = {std} rad/s")
    plot_feedback(feedback_msgs)


if __name__ == "__main__":
    main()
