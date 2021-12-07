#!/usr/bin/env python
"""Testing of the robust balancing constraints"""
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
from PIL import Image
import rospkg
from jaxlie import SO3

import tray_balance_sim.util as util
from tray_balance_sim.simulation import MobileManipulatorSimulation
from tray_balance_sim.recording import Recorder

import IPython

# hook into the bindings from the OCS2-based controller
from tray_balance_ocs2.MobileManipulatorPyBindings import (
    mpc_interface,
    scalar_array,
    vector_array,
    matrix_array,
    TargetTrajectories,
)


# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 50  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 6.0  # duration of trajectory (s)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_DIR = Path("/media/adam/Data/PhD/Videos/ICRA22/")
VIDEO_PATH = VIDEO_DIR / ("no_object_dist0.1_" + TIMESTAMP)

FRAMES_PATH = VIDEO_PATH
VIDEO_PERIOD = 40  # 25 frames per second with 1000 steps per second
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

RECORD_VIDEO = False


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=SIM_DT)

    N = int(DURATION / sim.dt)

    # simulation objects and model
    robot, objects, composites = sim.setup(
        obj_names=["tray", "stacked_cylinder1", "stacked_cylinder2", "stacked_cylinder3"]
        # obj_names=["tray", "flat_cylinder1", "flat_cylinder2", "flat_cylinder3"]
    )

    q, v = robot.joint_states()
    r_ew_w, Q_we = robot.link_pose()
    v_ew_w, ω_ew_w = robot.link_velocity()

    # data recorder and plotter
    recorder = Recorder(
        sim.dt,
        DURATION,
        RECORD_PERIOD,
        ns=robot.ns,
        ni=robot.ni,
        n_objects=len(objects),
        control_period=CTRL_PERIOD,
        n_balance_con=3 * 2,
        n_collision_pair=1,
        n_dynamic_obs=0,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    for name, obj in objects.items():
        print(f"{name} CoM = {obj.body.com}")
    # IPython.embed()

    if RECORD_VIDEO:
        os.makedirs(FRAMES_PATH)
        cam_view_matrix = pyb.computeViewMatrixFromYawPitchRoll(
            distance=4,
            yaw=42,
            pitch=-35.8,
            roll=0,
            cameraTargetPosition=[1.28, 0.045, 0.647],
            upAxisIndex=2,
        )

        cam_proj_matrix = pyb.computeProjectionMatrixFOV(
            fov=60.0, aspect=VIDEO_WIDTH / VIDEO_HEIGHT, nearVal=0.1, farVal=1000.0
        )

    # initial time, state, and input
    t = 0.0
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    target_times = [0]
    r_obs0 = np.array(r_ew_w) + [0, -10, 0]

    # setup MPC and initial EE target pose
    rospack = rospkg.RosPack()
    task_info_path = os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )
    mpc = mpc_interface(task_info_path, "/tmp/ocs2")
    t_target = scalar_array()
    for target_time in target_times:
        t_target.push_back(target_time)

    input_target = vector_array()
    for _ in target_times:
        input_target.push_back(u)

    # goal 1
    r_ew_w_d = np.array(r_ew_w) + [2, 0, -0.5]
    Qd = Q_we

    # goal 2
    # r_ew_w_d = np.array(r_ew_w) + [0, 2, 0.5]
    # Qd = Q_we

    # goal 3
    # r_ew_w_d = np.array(r_ew_w) + [0, -2, 0]
    # Qd = util.quat_multiply(Q_we, np.array([0, 0, 1, 0]))

    # NOTE: doesn't show up in the recording
    # util.debug_frame_world(0.2, list(r_ew_w_d), orientation=Qd, line_width=3)

    goal_visual_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=(0, 1, 0, 1),
    )
    goal_uid = pyb.createMultiBody(
        baseMass=0,  # non-dynamic body
        baseVisualShapeIndex=goal_visual_uid,
        basePosition=list(r_ew_w_d),
        baseOrientation=list(Qd),
    )

    state_target = vector_array()
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs0)))

    target_trajectories = TargetTrajectories(t_target, state_target, input_target)
    mpc.reset(target_trajectories)

    target_idx = 0

    assert len(state_target) == len(target_times)
    assert len(t_target) == len(target_times)
    assert len(input_target) == len(target_times)

    frame_num = 0

    for i in range(N):
        q, v = robot.joint_states()
        x = np.concatenate((q, v))
        mpc.setObservation(t, x, u)

        # TODO this should be set to reflect the MPC time step
        # we can increase it if the MPC rate is faster
        if i % CTRL_PERIOD == 0:
            # robot.cmd_vel = v  # NOTE
            try:
                t0 = time.time()
                mpc.advanceMpc()
                t1 = time.time()
            except RuntimeError as e:
                print(e)
                IPython.embed()
                i -= 1  # for the recorder
                break
            recorder.control_durations[i // CTRL_PERIOD] = t1 - t0

        # As far as I can tell, evaluateMpcSolution actually computes the input
        # for the particular time and state (the input is often at least
        # state-varying in DDP, with linear feedback on state error). OTOH,
        # getMpcSolution just gives the current MPC policy trajectory over the
        # entire time horizon, without accounting for the given state. So it is
        # like doing feedforward input only, which is bad.
        x_opt = np.zeros(robot.ns)
        u = np.zeros(robot.ni)
        mpc.evaluateMpcSolution(t, x, x_opt, u)

        robot.command_acceleration(u)

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()
            recorder.ineq_cons[idx, :] = mpc.stateInputInequalityConstraint(
                "trayBalance", t, x, u
            )

            r_ew_w_d = state_target[target_idx][:3]
            Q_we_d = state_target[target_idx][3:7]

            # record
            recorder.us[idx, :] = u
            recorder.xs[idx, :] = x
            recorder.r_ew_wds[idx, :] = r_ew_w_d
            recorder.r_ew_ws[idx, :] = r_ew_w
            recorder.Q_wes[idx, :] = Q_we
            recorder.Q_weds[idx, :] = Q_we_d
            recorder.v_ew_ws[idx, :] = v_ew_w
            recorder.ω_ew_ws[idx, :] = ω_ew_w

            # SO3_we = SO3.from_quaternion_xyzw(Q_we)
            # print(np.array(SO3_we.as_matrix()) @ [0, 0, -9.81])

            for j, obj in enumerate(objects.values()):
                r, Q = obj.bullet.get_pose()
                recorder.r_ow_ws[j, idx, :] = r
                recorder.Q_wos[j, idx, :] = Q

            recorder.cmd_vels[idx, :] = robot.cmd_vel

        sim.step(step_robot=True)
        t += sim.dt

        if RECORD_VIDEO and i % VIDEO_PERIOD == 0:
            (w, h, rgb, dep, seg) = pyb.getCameraImage(
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                shadow=1,
                viewMatrix=cam_view_matrix,
                projectionMatrix=cam_proj_matrix,
                flags=pyb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                renderer=pyb.ER_BULLET_HARDWARE_OPENGL,
            )
            img = Image.fromarray(np.reshape(rgb, (h, w, 4)), "RGBA")
            img.save(FRAMES_PATH / ("frame_" + str(frame_num) + ".png"))
            frame_num += 1

    if recorder.ineq_cons.shape[1] > 0:
        print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    # save logged data
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        if len(sys.argv) > 2:
            prefix = sys.argv[2]
        else:
            prefix = "data"
        fname = prefix + "_" + TIMESTAMP
        recorder.save(fname)

    last_sim_index = i
    recorder.plot_ee_position(last_sim_index)
    recorder.plot_ee_orientation(last_sim_index)
    recorder.plot_ee_velocity(last_sim_index)
    for j in range(len(objects)):
        recorder.plot_object_error(last_sim_index, j)
    recorder.plot_balancing_constraints(last_sim_index)
    recorder.plot_commands(last_sim_index)
    recorder.plot_control_durations(last_sim_index)
    recorder.plot_cmd_vs_real_vel(last_sim_index)
    recorder.plot_joint_config(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
