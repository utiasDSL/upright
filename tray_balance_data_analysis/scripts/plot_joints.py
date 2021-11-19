#!/usr/bin/env python3
import sys
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import tray_balance_constraints as tbc
import pinocchio

import IPython

URDF_PATH = "/home/adam/phd/code/mm/ocs2_noetic/catkin_ws/src/ocs2_mobile_manipulator_modified/urdf/mm.urdf"
EE_FRAME_NAME = "thing_tool"


def msg_time(msg):
    """Extract message timestamp as float in seconds."""
    return msg.header.stamp.to_sec()


def parse_t0(msgs):
    return msg_time(msgs[0])


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


def compute_forward_kinematics(model, q, v, a):
    """Compute EE pose, velocity, and acceleration from joint motion."""
    data = model.createData()
    pinocchio.forwardKinematics(model, data, q, v, a)
    ee_id = model.getFrameId(EE_FRAME_NAME)

    r = data.oMf[ee_id].translation
    C = data.oMf[ee_id].rotation

    V = pinocchio.getFrameVelocity(model, data, ee_id)  # TODO reference frame needed?
    A = pinocchio.getFrameAcceleration(model, data, ee_id)

    v = V.linear
    ω = V.angular
    a = A.linear
    α = A.angular

    return r, C, v, ω, a, α


def compute_inequality_constraints(model, q, v, a):
    r, C, v, ω, a, α = compute_forward_kinematics(model, q, v, a)
    params = tbc.TrayBalanceParameters()  # default params for now
    h = tbc.inequality_constraints(C, ω, a, α)
    return h


def main():
    bagname = sys.argv[1]
    bag = rosbag.Bag(bagname)

    model = pinocchio.buildModelFromUrdf(URDF_PATH, pinocchio.JointModelPlanar())

    IPython.embed()

    control_info_msgs = [msg for _, msg, _ in bag.read_messages("/mm/control_info")]
    t = parse_time(control_info_msgs)
    names = control_info_msgs[0].joints.name
    qs = np.array([msg.joints.position for msg in control_info_msgs])
    dqs = np.array([msg.joints.velocity for msg in control_info_msgs])
    us = np.array([msg.command for msg in control_info_msgs])

    ineq_cons = np.zeros((len(control_info_msgs), 4))
    for i in range(len(control_info_msgs)):
        ineq_cons[i, :] = compute_inequality_constraints(
            model, qs[i, :], dqs[i, :], us[i, :]
        )

    plt.figure()
    for i in range(9):
        plt.plot(t, qs[:, i], label=names[i])
    plt.grid()
    plt.legend()
    plt.title("Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")

    plt.figure()
    for i in range(9):
        plt.plot(t, dqs[:, i], label=names[i])
    plt.grid()
    plt.legend()
    plt.title("Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")

    plt.figure()
    for i in range(9):
        plt.plot(t, us[:, i], label=names[i])
    plt.grid()
    plt.legend()
    plt.title("Joint Commands")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")

    plt.figure()
    plt.plot(t, ineq_cons[:, 0], label="Friction 1")
    plt.plot(t, ineq_cons[:, 1], label="Friction 2")
    plt.plot(t, ineq_cons[:, 2], label="Contact")
    plt.plot(t, ineq_cons[:, 3], label="Tipping")
    plt.grid()
    plt.legend()
    plt.title("Inequality constraints")
    plt.xlabel("Time (s)")
    plt.ylabel("Constraint value")

    plt.show()


if __name__ == "__main__":
    main()
