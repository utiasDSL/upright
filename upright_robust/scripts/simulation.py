#!/usr/bin/env python3
"""Closed-loop upright reactive simulation using Pybullet."""
import datetime
import time
import signal
import sys

import numpy as np
import pybullet as pyb
from pyb_utils.frame import debug_frame_world
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import mobile_manipulation_central as mm
import upright_sim as sim
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd
import upright_robust as rob

import IPython


def sigint_handler(sig, frame):
    print("Ctrl-C pressed: exiting.")
    sys.exit(0)


def parse_controller_from_config(ctrl_config, robot, objects, contacts, timestep):
    use_balancing_constraints = ctrl_config["balancing"]["enabled"]
    tilting_type = ctrl_config["reactive"]["tilting"]
    use_robust_constraints = ctrl_config["reactive"]["robust"]
    use_face_form = ctrl_config["reactive"]["face_form"]

    # rotational tracking gains
    kθ = 1
    kω = 2 * np.sqrt(kθ)
    use_dvdt_scaling = False

    if tilting_type == "full":
        return rob.ReactiveBalancingControllerFullTilting(
            robot,
            objects,
            contacts,
            timestep,
            kθ=kθ,
            kω=kω,
            use_dvdt_scaling=use_dvdt_scaling,
            use_face_form=use_face_form,
            use_robust_constraints=use_robust_constraints,
        )
    elif tilting_type == "tray":
        return rob.NominalReactiveBalancingControllerTrayTilting(
            robot,
            objects,
            contacts,
            timestep,
            kθ=kθ,
            kω=kω,
            use_balancing_constraints=use_balancing_constraints,
        )
    elif tilting_type == "flat":
        return rob.NominalReactiveBalancingControllerFlat(
            robot,
            objects,
            contacts,
            timestep,
            use_balancing_constraints=use_balancing_constraints,
        )
    else:
        raise ValueError(f"Unknown tilting type {tilting_type}")


def main():
    np.set_printoptions(precision=6, suppress=True)
    signal.signal(signal.SIGINT, sigint_handler)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    timestamp = datetime.datetime.now()
    env = sim.simulation.UprightSimulation(
        config=sim_config,
        timestamp=timestamp,
        video_name=cli_args.video,
        extra_gui=sim_config.get("extra_gui", False),
    )
    env.settle(5.0)

    # controller
    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot

    # make EE origin the reference point for all contacts
    objects = model.settings.balancing_settings.objects
    for c in model.settings.balancing_settings.contacts:
        if c.object1_name != "ee":
            o1 = objects[c.object1_name]
            c.r_co_o1 = c.r_co_o1 + o1.body.com
        o2 = objects[c.object2_name]
        c.r_co_o2 = c.r_co_o2 + o2.body.com

    contacts = [
        rob.RobustContactPoint(c) for c in model.settings.balancing_settings.contacts
    ]

    # parse the bounds for each object
    uncertain_objects = {}
    arrangement_name = ctrl_config["balancing"]["arrangement"]
    arrangement_config = ctrl_config["arrangements"][arrangement_name]
    for conf in arrangement_config["objects"]:
        name = conf["name"]
        type_ = conf["type"]
        bounds_config = ctrl_config["objects"][type_].get("bounds", {})
        bounds = rob.ObjectBounds.from_config(bounds_config)
        uncertain_objects[name] = rob.UncertainObject(objects[name], bounds)

    # translational tracking gains
    kp = 1
    kv = 2 * np.sqrt(kp)

    controller = parse_controller_from_config(
        ctrl_config, model.robot, uncertain_objects, contacts, env.timestep
    )

    t = 0.0
    q, v = env.robot.joint_states()
    a = np.zeros(env.robot.nv)

    v_cmd = np.zeros(env.robot.nv)

    robot.forward(q, v)
    r_ew_w_0, Q_we_0 = robot.link_pose()
    r_ew_w_d = r_ew_w_0 + ctrl_config["waypoints"][0]["position"]

    # desired trajectory
    trajectory = mm.PointToPointTrajectory.quintic(
        r_ew_w_0, r_ew_w_d, max_vel=2, max_acc=4
    )

    # goal position
    debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_0, line_width=3)

    # simulation loop
    while t <= env.duration:
        # current joint state
        q, v = env.robot.joint_states(add_noise=False)

        # current EE state
        robot.forward(q, v)
        r_ew_w, C_we = robot.link_pose(rotation_matrix=True)
        v_ew_w, ω_ew_w = robot.link_velocity()

        # desired EE state
        # rd, vd, ad = trajectory.sample(t)

        rd = r_ew_w_d
        vd = np.zeros(3)
        ad = np.zeros(3)

        # commanded EE acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command
        t0 = time.time()
        u, A_e = controller.solve(q, v, a_ew_w_cmd)
        t1 = time.time()

        A_w = block_diag(C_we, C_we) @ A_e

        print(f"solve took {1000 * (t1 - t0)} ms")
        print(f"a_cmd = {a_ew_w_cmd}")
        print(f"A_w = {A_w}")

        # NOTE: we want to use v_cmd rather than v here because PyBullet
        # doesn't respond to small velocities well, and it screws up angular
        # velocity tracking
        v_cmd = v_cmd + env.timestep * u
        env.robot.command_velocity(v_cmd, bodyframe=False)
        t = env.step(t, step_robot=False)[0]

    IPython.embed()


if __name__ == "__main__":
    main()
