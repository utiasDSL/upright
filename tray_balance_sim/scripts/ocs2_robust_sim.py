#!/usr/bin/env python3
"""Testing of the robust balancing constraints"""
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as pyb
from PIL import Image
import rospkg
from scipy.spatial.distance import pdist, squareform
import liegroups

from tray_balance_sim import util, ocs2_util, pyb_util, clustering, geometry
from tray_balance_sim.simulation import MobileManipulatorSimulation
from tray_balance_sim.recording import Recorder, VideoRecorder
from tray_balance_sim.camera import Camera

import IPython

# hook into the bindings from the OCS2-based controller
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 50  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 8.0  # duration of trajectory (s)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_DIR = Path("/media/adam/Data/PhD/Videos/tray-balance/robust/")
VIDEO_PATH = VIDEO_DIR / ("robust_stack3_" + TIMESTAMP)
VIDEO_PERIOD = 40  # 25 frames per second with 1000 steps per second
RECORD_VIDEO = False


def set_bounding_spheres(robot, objects, settings, k=2, plot_point_cloud=False):
    target = objects["stacked_cylinder2"].bullet.get_pose()[0]
    cam_pos = [target[0], target[1] - 1, target[2]]
    camera = Camera(
        camera_position=cam_pos,
        target_position=target,
        width=200,
        height=200,
        fov=50,
        near=0.1,
        far=5,
    )
    w, h, rgb, dep, seg = camera.get_frame()
    camera.save_frame("testframe.png")
    points = camera.get_point_cloud(dep)

    # mask out everything except balanced objects
    mask = np.zeros_like(seg)
    for obj in objects.values():
        mask = np.logical_or(seg == obj.bullet.uid, mask)
    points = points[mask.T, :]

    # compute max_radius for robust inertia
    max_radius = 0.5 * np.max(pdist(points))
    print(f"max_radius = {max_radius}")

    # cluster point cloud points and bound with spheres
    centers, radii = clustering.cluster_and_bound(points, k=k, cluster_type="greedy")
    # centers, radii = clustering.iterative_ritter(points, k=k)
    # volume = 4 * np.pi * np.sum(radii ** 3) / 3
    # print(f"Volume = {volume}")

    if plot_point_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points[:, 0], points[:, 1], zs=points[:, 2])
        ax.scatter(camera.target[0], camera.target[1], zs=camera.target[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    r_ew_w, Q_we = robot.link_pose()
    C_we = util.quaternion_to_matrix(Q_we)

    balls = []
    for i in range(k):
        r_bw_w = centers[i, :]
        r_be_e = C_we.T @ (r_bw_w - r_ew_w)
        balls.append(ocs2.Ball(r_be_e, radii[i]))
    settings.tray_balance_settings.robust_params.balls = balls
    settings.tray_balance_settings.robust_params.max_radius = max_radius


def get_task_settings():
    settings = ocs2.TaskSettings()

    settings.method = ocs2.TaskSettings.Method.DDP
    settings.dynamic_obstacle_enabled = False
    settings.collision_avoidance_enabled = False

    # tray balance settings
    settings.tray_balance_settings.enabled = True
    settings.tray_balance_settings.robust = False
    settings.tray_balance_settings.constraint_type = ocs2.ConstraintType.Soft

    config = ocs2.TrayBalanceConfiguration()
    config.arrangement = ocs2.TrayBalanceConfiguration.Arrangement.Stacked
    config.num = 3
    settings.tray_balance_settings.config = config

    # robust settings
    robust_params = ocs2.RobustParameterSet()
    robust_params.min_support_dist = 0.05
    robust_params.min_mu = 0.5
    robust_params.min_r_tau = geometry.circle_r_tau(robust_params.min_support_dist)

    if config.arrangement == ocs2.TrayBalanceConfiguration.Arrangement.Stacked:
        ball1 = ocs2.Ball([0, 0, 0.1], 0.12)
        ball2 = ocs2.Ball([0, 0, 0.3], 0.12)

        robust_params.max_radius = 0.5 * (
            np.linalg.norm(ball2.center - ball2.center) + ball1.radius + ball2.radius
        )
        robust_params.balls = [ball1, ball2]
    else:
        ball = ocs2.Ball([0, 0, 0.02 + 0.02 + 0.075], 0.1)
        robust_params.max_radius = ball.radius
        robust_params.balls = [ball]

    settings.tray_balance_settings.robust_params = robust_params

    return settings


# TODO: could build a generic object to attach a visual object to a multibody
class RobustSpheres:
    def __init__(self, robot, robust_params, color=(0.5, 0.5, 0.5, 0.5)):
        self.robot = robot
        r_ew_w, _ = robot.link_pose()

        self.spheres = []
        self.centers = []
        for ball in robust_params.balls:
            position = r_ew_w + ball.center
            self.centers.append(ball.center)
            self.spheres.append(
                pyb_util.GhostSphere(radius=ball.radius, position=position, color=color)
            )

    def update(self):
        r_ew_w, Q_we = self.robot.link_pose()
        for i in range(len(self.spheres)):
            position = util.transform_point(r_ew_w, Q_we, self.centers[i])
            self.spheres[i].set_position(position)


def main():
    np.set_printoptions(precision=3, suppress=True)

    settings = get_task_settings()
    sim = MobileManipulatorSimulation(dt=SIM_DT)

    N = int(DURATION / sim.dt)

    # simulation objects and model
    robot, objects, _ = sim.setup(ocs2_util.get_obj_names_from_settings(settings))
    for name, obj in objects.items():
        print(f"{name} CoM = {obj.body.com}")

    if settings.tray_balance_settings.robust:
        set_bounding_spheres(robot, objects, settings)
        robust_spheres = RobustSpheres(
            robot, settings.tray_balance_settings.robust_params
        )
        IPython.embed()

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
        n_balance_con=ocs2_util.get_num_balance_constraints(settings),
        n_collision_pair=0,
        n_dynamic_obs=0,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    if RECORD_VIDEO:
        video = VideoRecorder(
            path=VIDEO_PATH,
            distance=4,
            roll=0,
            pitch=-35.8,
            yaw=42,
            target_position=[1.28, 0.045, 0.647],
        )

    # initial time, state, and input
    t = 0.0
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    target_times = [0]
    r_obs0 = np.array(r_ew_w) + [0, -10, 0]

    mpc = ocs2_util.setup_ocs2_mpc_interface(settings)

    # setup EE target
    t_target = ocs2.scalar_array()
    for target_time in target_times:
        t_target.push_back(target_time)

    input_target = ocs2.vector_array()
    for _ in target_times:
        input_target.push_back(u)

    # goal 1
    # r_ew_w_d = np.array(r_ew_w) + [2, 0, -0.5]
    # Qd = Q_we

    # goal 2
    # r_ew_w_d = np.array(r_ew_w) + [0, 2, 0.5]
    # Qd = Q_we

    # goal 3
    r_ew_w_d = np.array(r_ew_w) + [0, -2, 0]
    Qd = util.quat_multiply(Q_we, np.array([0, 0, 1, 0]))

    # NOTE: doesn't show up in the recording
    pyb_util.debug_frame_world(0.2, list(r_ew_w_d), orientation=Qd, line_width=3)

    pyb_util.GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1))

    state_target = ocs2.vector_array()
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs0)))

    target_trajectories = ocs2.TargetTrajectories(t_target, state_target, input_target)
    mpc.reset(target_trajectories)

    target_idx = 0

    assert len(state_target) == len(target_times)
    assert len(t_target) == len(target_times)
    assert len(input_target) == len(target_times)

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

            if settings.tray_balance_settings.enabled:
                if (
                    settings.tray_balance_settings.constraint_type
                    == ocs2.ConstraintType.Hard
                ):
                    recorder.ineq_cons[idx, :] = mpc.stateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                else:
                    recorder.ineq_cons[idx, :] = mpc.softStateInputInequalityConstraint(
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

            for j, obj in enumerate(objects.values()):
                r, Q = obj.bullet.get_pose()
                recorder.r_ow_ws[j, idx, :] = r
                recorder.Q_wos[j, idx, :] = Q

            recorder.cmd_vels[idx, :] = robot.cmd_vel

        sim.step(step_robot=True)
        if settings.tray_balance_settings.robust:
            robust_spheres.update()
        t += sim.dt
        time.sleep(sim.dt)

        if RECORD_VIDEO and i % VIDEO_PERIOD == 0:
            video.save_frame()

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
