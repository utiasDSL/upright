import numpy as np
import sys
import rosbag
import matplotlib.pyplot as plt

import tray_balance_ocs2 as ctrl

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


def plot_goal(msg, ts_fb, vds_fb, t0, mapping):
    ts = np.array([point.time_from_start.to_sec() for point in msg.goal.trajectory.points]) + t0
    q_goals = np.array([point.positions for point in msg.goal.trajectory.points])
    v_goals = np.array([point.velocities for point in msg.goal.trajectory.points])
    a_goals = np.array([point.accelerations for point in msg.goal.trajectory.points])
    u_goals = np.array([point.effort for point in msg.goal.trajectory.points])

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
    # x_goals = np.hstack((q_goals, v_goals, a_goals))
    # trajectory = ctrl.trajectory.StateInputTrajectory(ts, x_goals, u_goals)
    # interpolator = ctrl.trajectory.TrajectoryInterpolator(mapping, trajectory)

    plt.figure()
    plt.grid()
    for i in range(nv):
        plt.plot(ts, v_goals[:, i], label=f"vg_{i}", linestyle="--", color=colors[i])
        plt.plot(ts_fb, vds_fb[:, i], label=f"vd_{i}", color=colors[i])
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


def plot_both(traj_msg, t0_traj, goal_msg, t0_goal):
    # plot goals published by MPC and received by controller to make sure they
    # are the same
    t_goals = np.array([point.time_from_start.to_sec() for point in goal_msg.goal.trajectory.points]) + t0_goal
    # q_goals = np.array([point.positions for point in msg.goal.trajectory.points])
    v_goals = np.array([point.velocities for point in goal_msg.goal.trajectory.points])
    # a_goals = np.array([point.accelerations for point in msg.goal.trajectory.points])

    t_mpcs = np.array([point.time_from_start.to_sec() for point in traj_msg.points]) + t0_traj
    v_mpcs = np.array([point.velocities for point in traj_msg.points])

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure()
    plt.grid()
    for i in range(v_goals.shape[1]):
        plt.plot(t_goals, v_goals[:, i], label=f"vg_{i}", linestyle="--", color=colors[i])
        plt.plot(t_mpcs, v_mpcs[:, i], label=f"vm_{i}", color=colors[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Goal velocity (rad/s)")
    plt.legend()
    plt.title("Goal joint velocity")


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

    mpc_traj_msgs = [msg for _, msg, _ in bag.read_messages("/mobile_manipulator_joint_trajectory")]

    feedback_msgs = [
        msg
        for _, msg, _ in bag.read_messages(
            "/scaled_vel_joint_traj_controller/follow_joint_trajectory/feedback"
        )
    ]
    ts = parse_time(feedback_msgs)
    vs = np.array([msg.feedback.actual.velocities for msg in feedback_msgs])
    vds = np.array([msg.feedback.desired.velocities for msg in feedback_msgs])
    Δvs = np.array([msg.feedback.error.velocities for msg in feedback_msgs])

    ads = np.array([msg.feedback.desired.accelerations for msg in feedback_msgs])

    # for msg in goal_msgs:
    #     t0 = msg.header.stamp.to_sec() - goal_msgs[0].header.stamp.to_sec()
    #     plot_goal(msg, ts, ads, t0)

    for traj_msg, goal_msg in zip(mpc_traj_msgs, goal_msgs):
        t0_traj = traj_msg.header.stamp.to_sec() - mpc_traj_msgs[0].header.stamp.to_sec()
        t0_goal = goal_msg.header.stamp.to_sec() - goal_msgs[0].header.stamp.to_sec()
        plot_both(traj_msg, t0_traj, goal_msg, t0_goal)

    # q_goals = np.array([msg.goal.trajectory.points[0].positions for msg in goal_msgs])
    # v_goals = np.array([msg.goal.trajectory.points[0].velocities for msg in goal_msgs])

    # idx = -5
    # ts = np.array([point.time_from_start.to_sec() for point in goal_msgs[idx].goal.trajectory.points])
    # q_goals = np.array([point.positions for point in goal_msgs[idx].goal.trajectory.points])
    # v_goals = np.array([point.velocities for point in goal_msgs[idx].goal.trajectory.points])
    #
    # nq = q_goals.shape[1]
    # nv = v_goals.shape[1]
    #
    # plt.figure()
    # plt.grid()
    # for i in range(nq):
    #     plt.plot(ts, q_goals[:, i], label=f"qd_{i}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Desired position (rad)")
    # plt.legend()
    # plt.title("Desired joint position")
    #
    # plt.figure()
    # plt.grid()
    # for i in range(nv):
    #     plt.plot(ts, v_goals[:, i], label=f"vd_{i}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Desired velocity (rad/s)")
    # plt.legend()
    # plt.title("Desired joint velocity")

    plt.show()


if __name__ == "__main__":
    main()
