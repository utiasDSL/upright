import numpy as np
import sys
import rosbag
import matplotlib.pyplot as plt

import upright_control as ctrl

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


def plot_goal(msg, actual_interpolator, t0, mapping):
    ts = np.array([point.time_from_start.to_sec() for point in msg.goal.trajectory.points]) + t0
    q_goals = np.array([point.positions for point in msg.goal.trajectory.points])
    v_goals = np.array([point.velocities for point in msg.goal.trajectory.points])
    a_goals = np.array([point.accelerations for point in msg.goal.trajectory.points])
    u_goals = np.array([point.effort for point in msg.goal.trajectory.points])

    # get actual value at this point
    x0 = actual_interpolator.interpolate(t0)

    # interpolate the goal trajectory
    x_goals = np.hstack((q_goals, v_goals, a_goals))
    ts = np.concatenate(([t0], ts))
    xs = np.vstack((x0, x_goals))
    us = np.vstack((np.zeros_like(u_goals[0, :]), u_goals))

    trajectory = ctrl.trajectory.StateInputTrajectory(ts, xs, us)
    interpolator = ctrl.trajectory.TrajectoryInterpolator(mapping, trajectory)

    nq = q_goals.shape[1]
    nv = v_goals.shape[1]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # plt.figure()
    # plt.grid()
    # for i in range(nq):
    #     plt.plot(ts, q_goals[:, i], label=f"qd_{i}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Desired position (rad)")
    # plt.legend()
    # plt.title("Desired joint position")

    # TODO we also want to interpolate the trajectory
    t_interp = np.linspace(ts[0], ts[-1], 100)
    x_interp = np.zeros((100, 18))
    for i in range(100):
        x_interp[i, :] = interpolator.interpolate(t_interp[i])

    plt.figure()
    plt.grid()
    for i in range(nv):
        plt.plot(t_interp, x_interp[:, 6+i], label=f"vg_{i}", color=colors[i])
        # plt.plot(ts_fb, vds_fb[:, i], label=f"vd_{i}", color=colors[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Goal velocity (rad/s)")
    plt.legend()
    plt.title("Goal joint velocity")

    # plt.figure()
    # plt.grid()
    # for i in range(nv):
    #     plt.plot(ts, a_goals[:, i], label=f"ag_{i}", linestyle="--", color=colors[i])
    #     plt.plot(ts_fb, vds_fb[:, i], label=f"ad_{i}", color=colors[i])
    # plt.xlabel("Time (s)")
    # plt.ylabel("Goal acceleration (rad/s)")
    # plt.legend()
    # plt.title("Goal joint acceleration")


def main():
    bag = rosbag.Bag(sys.argv[1])

    dims = ctrl.bindings.RobotDimensions()
    dims.q = 6
    dims.v = 6
    mapping = ctrl.trajectory.StateInputMapping(dims)

    goal_msgs = [
        msg
        for _, msg, _ in bag.read_messages(
            "/scaled_vel_joint_traj_controller/follow_joint_trajectory/goal"
        )
    ]

    feedback_msgs = [
        msg
        for _, msg, _ in bag.read_messages(
            "/scaled_vel_joint_traj_controller/follow_joint_trajectory/feedback"
        )
    ]
    ts = parse_time(feedback_msgs)
    qs = np.array([msg.feedback.actual.positions for msg in feedback_msgs])
    vs = np.array([msg.feedback.actual.velocities for msg in feedback_msgs])

    # we don't get actual feedback on these
    acs = np.array([np.zeros(6) for msg in feedback_msgs])
    es = np.array([np.zeros(6) for msg in feedback_msgs])

    xs = np.hstack((qs, vs, acs))

    # TODO we should really use a cubic interpolator here
    actual_trajectory = ctrl.trajectory.StateInputTrajectory(ts, xs, es)
    actual_interpolator = ctrl.trajectory.TrajectoryInterpolator(mapping, actual_trajectory)

    for msg in goal_msgs:
        t0 = msg.header.stamp.to_sec() - goal_msgs[0].header.stamp.to_sec()
        plot_goal(msg, actual_interpolator, t0, mapping)

    plt.show()


if __name__ == "__main__":
    main()
