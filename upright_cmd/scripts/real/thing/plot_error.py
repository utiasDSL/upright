import numpy as np
import sys
import rosbag
import matplotlib.pyplot as plt

import IPython


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


def main():
    bag = rosbag.Bag(sys.argv[1])

    feedback_msgs = [
        msg
        for _, msg, _ in bag.read_messages(
            "/scaled_vel_joint_traj_controller/follow_joint_trajectory/feedback"
        )
    ]
    ts = parse_time(feedback_msgs)

    # positions
    qs = np.array([msg.feedback.actual.positions for msg in feedback_msgs])
    qds = np.array([msg.feedback.desired.positions for msg in feedback_msgs])
    Δqs = np.array([msg.feedback.error.positions for msg in feedback_msgs])

    # velocities
    vs = np.array([msg.feedback.actual.velocities for msg in feedback_msgs])
    vds = np.array([msg.feedback.desired.velocities for msg in feedback_msgs])
    Δvs = np.array([msg.feedback.error.velocities for msg in feedback_msgs])

    nq = qs.shape[1]
    nv = vs.shape[1]

    plt.figure()
    plt.grid()
    for i in range(nq):
        plt.plot(ts, Δqs[:, i], label=f"Δq_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (rad)")
    plt.legend()
    plt.title("Joint position error")

    plt.figure()
    plt.grid()
    for i in range(nv):
        plt.plot(ts, Δvs[:, i], label=f"Δv_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity error (rad/s)")
    plt.legend()
    plt.title("Joint velocity error")

    plt.figure()
    plt.grid()
    for i in range(nq):
        plt.plot(ts, qds[:, i], label=f"qd_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired position (rad)")
    plt.legend()
    plt.title("Desired joint position")

    plt.figure()
    plt.grid()
    for i in range(nv):
        plt.plot(ts, vds[:, i], label=f"vd_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired velocity (rad/s)")
    plt.legend()
    plt.title("Desired joint velocity")

    plt.show()


if __name__ == "__main__":
    main()
