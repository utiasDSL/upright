#!/usr/bin/env python3
"""Testing of the robust balancing constraints"""
import enum
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import rospkg
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from tray_balance_sim import util, ocs2_util, robustness, cameras
from tray_balance_sim.simulation import MobileManipulatorSimulation, DynamicObstacle
from tray_balance_sim.recording import Recorder, VideoRecorder

import tray_balance_constraints as con
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2

import IPython


@enum.unique
class SimType(enum.Enum):
    POSE_TO_POSE = 1
    DYNAMIC_OBSTACLE = 2
    STATIC_OBSTACLE = 3


SIM_TYPE = SimType.POSE_TO_POSE


# simulation parameters
SIM_DT = 0.001
RECORD_PERIOD = 10
DURATION = 12.0  # duration of trajectory (s)
#DURATION = 6.0  # duration of trajectory (s)

# generate new control signal every CTRL_PERIOD timesteps
if SIM_TYPE == SimType.POSE_TO_POSE:
    CTRL_PERIOD = 25
elif SIM_TYPE == SimType.DYNAMIC_OBSTACLE:
    CTRL_PERIOD = 40
elif SIM_TYPE == SimType.STATIC_OBSTACLE:
    CTRL_PERIOD = 100

# measurement and process noise
USE_NOISY_STATE_TO_PLAN = True

# divide by 1000 to convert from mm to meters
# Q_STDEV = 10 / 1000
# V_STDEV = 100 / 1000
# V_CMD_STDEV = 10 / 1000
Q_STDEV = 0
V_STDEV = 0
V_CMD_STDEV = 0

DATA_DIR_PATH = Path("/media/adam/Data/PhD/Data/object-balance/bounded")

# video recording parameters
RECORD_VIDEO = False
VIDEO_NAME = "single_object_r5"
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_DIR = Path("/media/adam/Data/PhD/Videos/tray-balance/bounded")
VIDEO_PERIOD = 40  # 25 frames per second with 1000 steps per second
# select appropriate video recorders here
VIDEO_RECORDER_TYPES = [
    (cameras.PoseToPoseVideoRecorder1, "view1"),
    # (cameras.PoseToPoseVideoRecorder2, "view2"),
]

DO_DYNAMIC_OBSTACLE_PHOTO_SHOOT = False

# robust bounding spheres
USE_ROBUST_CONSTRAINTS = True
NUM_BOUNDING_SPHERES = 1

POSITION_GOAL = np.array([0, -2, -0.5])
ORIENTATION_GOAL = np.array([0, 0, 1, 0])

# stack alernative
# POSITION_GOAL = np.array([1, 1, 0])
# ORIENTATION_GOAL = np.array([0, 0, 0, 1])

# cups alternative
# POSITION_GOAL = np.array([1.5, 0, 0.5])
# ORIENTATION_GOAL = np.array([0, 0, 0, 1])

# object configurations
SHORT_CONFIG = ["tray", "cuboid_short"]
TALL_CONFIG = ["tray", "cuboid_tall"]
STACK_CONFIG = [
    "cylinder_base_stack",
    "cuboid1_stack",
    "cuboid2_stack",
    "cylinder3_stack",
]

# note rearrangement to remove the most challenging cup first
CUPS_CONFIG = ["tray", "cylinder3_cup", "cylinder1_cup", "cylinder2_cup"]


def main():
    np.set_printoptions(precision=3, suppress=True)

    N = int(DURATION / SIM_DT)

    # simulation, objects, and model
    sim = MobileManipulatorSimulation(dt=SIM_DT)
    robot, objects, composites = sim.setup(
        STACK_CONFIG[:1],
        load_static_obstacles=(SIM_TYPE == SimType.STATIC_OBSTACLE),
    )

    # initial time, state, input
    t = 0.0
    q, v = robot.joint_states()
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    settings_wrapper = ocs2_util.TaskSettingsWrapper(x)
    settings_wrapper.settings.tray_balance_settings.enabled = True
    settings_wrapper.settings.tray_balance_settings.robust = USE_ROBUST_CONSTRAINTS

    if SIM_TYPE == SimType.STATIC_OBSTACLE:
        settings_wrapper.settings.collision_avoidance_settings.enabled = True

    if SIM_TYPE == SimType.DYNAMIC_OBSTACLE:
        settings_wrapper.settings.dynamic_obstacle_settings.enabled = True

    settings_wrapper.settings.tray_balance_settings.config.enabled.normal = True
    settings_wrapper.settings.tray_balance_settings.config.enabled.friction = True
    settings_wrapper.settings.tray_balance_settings.config.enabled.zmp = True

    r_ew_w, Q_we = robot.link_pose()

    # set up video recordings
    # need to set it up here to pass into robust sphere generation
    videos = []
    if RECORD_VIDEO:
        for video_type, postfix in VIDEO_RECORDER_TYPES:
            name = "_".join([VIDEO_NAME, postfix, TIMESTAMP])
            videos.append(video_type(VIDEO_DIR, name))

    ghosts = []  # ghost (i.e., pure visual) objects
    if settings_wrapper.settings.tray_balance_settings.robust:
        # obj = objects[STACK_CONFIG[0]]
        # Δm = 0
        # r_gyr = 0.15 * np.array([1, 1, 1])  # radius of the cylinder
        #
        # # com_ellipsoid = con.Ellipsoid.point(obj.com)
        # com_half_lengths = 0.05 * np.array([1, 1, 1])
        # com_ellipsoid = con.Ellipsoid(obj.com, com_half_lengths, np.eye(3))
        #
        # # convert the object to the bounded one in bindings
        # # TODO it would be nice if there was less duplication between the C++
        # # side and the Python side
        # bounded_body = con.BoundedRigidBody(
        #     obj.mass - Δm, obj.mass + Δm, r_gyr, com_ellipsoid
        # )
        # bounded_obj = con.BoundedBalancedObject(
        #     bounded_body,
        #     obj.com_height,
        #     obj.support_area,
        #     obj.r_tau,
        #     obj.mu,
        # )
        settings_wrapper.settings.tray_balance_settings.bounded_config.objects = composites

    else:
        # if not using robust approach, use a default collision sphere
        balanced_object_collision_sphere = ocs2.CollisionSphere(
            name="thing_tool_collision_link",
            parent_frame_name="thing_tool",
            offset=np.array([0, 0, 0]),
            radius=0.25,
        )
        settings_wrapper.settings.dynamic_obstacle_settings.collision_spheres.push_back(
            balanced_object_collision_sphere
        )

        settings_wrapper.settings.collision_avoidance_settings.extra_spheres.push_back(
            balanced_object_collision_sphere
        )
        # fmt: off
        for pair in [
            ("thing_tool_collision_link", "chair3_1_link_0"),
            ("thing_tool_collision_link", "chair4_2_link_0"),
            ("thing_tool_collision_link", "chair2_1_link_0"),
            ("thing_tool_collision_link", "forearm_collision_link_0"),
        ]:
            settings_wrapper.settings.collision_avoidance_settings.collision_link_pairs.push_back(pair)
        # fmt: on

    q, v = robot.joint_states()
    x = np.concatenate((q, v))
    settings_wrapper.settings.initial_state = x

    # set process noise after initial routine to get robust spheres
    robot.v_cmd_stdev = V_CMD_STDEV

    # data recorder and plotter
    recorder = Recorder(
        sim.dt,
        DURATION,
        RECORD_PERIOD,
        ns=robot.ns,
        ni=robot.ni,
        n_objects=len(objects),
        control_period=CTRL_PERIOD,
        n_balance_con=settings_wrapper.get_num_balance_constraints(),
        n_collision_pair=settings_wrapper.get_num_collision_avoidance_constraints(),
        n_dynamic_obs=settings_wrapper.get_num_dynamic_obstacle_constraints() + 1,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    cameras.BalancedObjectCamera(robot).save_frame()
    cameras.RobotCamera(robot).save_frame()
    dynamic_cam = cameras.DynamicObstacleCamera()

    if SIM_TYPE == SimType.POSE_TO_POSE:
        target_times = [0]
        target_inputs = [u]

        # goal pose
        r_ew_w_d = r_ew_w + POSITION_GOAL
        Qd = util.quat_multiply(Q_we, ORIENTATION_GOAL)
        r_obs0 = np.array(r_ew_w) + [0, -10, 0]

        target_states = [np.concatenate((r_ew_w_d, Qd, r_obs0))]

        # visual indicator for target
        # NOTE: debug frame doesn't show up in the recording
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Qd, line_width=3)
        ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))

    elif SIM_TYPE == SimType.DYNAMIC_OBSTACLE:
        target_times = [0, 5]  # TODO
        target_inputs = [u for _ in target_times]

        # create the dynamic obstacle
        r_obs0 = np.array(r_ew_w) + [0, -1, 0]
        v_obs = np.array([0, 1.0, 0])
        obstacle = DynamicObstacle(r_obs0, radius=0.1, velocity=v_obs)

        # stationary pose
        r_ew_w_d = r_ew_w
        Qd = Q_we

        # start with the obstacle out of the way
        r_obs_out_of_the_way = r_ew_w + [0, -10, 0]

        target_states = [
            np.concatenate((r_ew_w_d, Qd, r_obs_out_of_the_way)),
            np.concatenate((r_ew_w_d, Qd, r_obs_out_of_the_way)),
        ]

        # trajectories are modified to introduce dynamic obstacles
        target_times_obs1 = [0, 5]
        target_times_obs2 = [2, 7]

        r_obsf = obstacle.sample_position(target_times_obs1[-1])
        target_states_obs = [
            np.concatenate((r_ew_w_d, Qd, r_obs0)),
            np.concatenate((r_ew_w_d, Qd, r_obsf)),
        ]

        # obstacle appears at t = 0
        target_trajectories_obs1 = ocs2_util.make_target_trajectories(
            target_times_obs1, target_states_obs, target_inputs
        )

        # obstacle appears again at t = 2
        target_trajectories_obs2 = ocs2_util.make_target_trajectories(
            target_times_obs2, target_states_obs, target_inputs
        )

        if DO_DYNAMIC_OBSTACLE_PHOTO_SHOOT:
            # add visualizations for EE desired pose and obstacle trajectory
            ghosts.append(
                GhostSphere(radius=0.025, position=r_ew_w_d, color=(0, 0, 0, 1))
            )
            dots = np.linspace(r_obs0, r_obs0 + 2 * v_obs, 40)
            for i in range(dots.shape[0]):
                ghosts.append(
                    GhostSphere(radius=0.01, position=dots[i, :], color=(1, 0, 0, 0.8))
                )

        for (
            sphere
        ) in settings_wrapper.settings.dynamic_obstacle_settings.collision_spheres:
            # robust spheres already have ghost spheres
            if sphere.name.startswith("robust"):
                continue
            link_idx = robot.links[sphere.parent_frame_name][0]
            ghosts.append(
                GhostSphere(
                    sphere.radius,
                    position=sphere.offset,
                    parent_body_uid=robot.uid,
                    parent_link_index=link_idx,
                    color=(0, 1, 0, 0.3),
                )
            )

        settings_wrapper.settings.collision_avoidance_settings.collision_link_pairs.clear()
        settings_wrapper.settings.collision_avoidance_settings.collision_link_pairs.push_back(
            ("robust_collision_sphere_0", "forearm_collision_sphere_link1_0"),
        )

    elif SIM_TYPE == SimType.STATIC_OBSTACLE:
        # target_duration = 8
        # num_waypoints = 6
        # target_dt = target_duration / (num_waypoints - 1)
        # # target_times = np.array([0, 2, 4, 6, 8, 10])
        # target_times = np.array([i * target_dt for i in range(num_waypoints)])
        # target_inputs = [u for _ in target_times]
        #
        # Qd = Q_we
        # r_obs0 = np.array(r_ew_w) + [0, -10, 0]
        #
        # target_states = [
        #     np.concatenate((r_ew_w + [0, 0, 0], Qd, r_obs0)),
        #     np.concatenate((r_ew_w + [1, 0, 0], Qd, r_obs0)),
        #     np.concatenate((r_ew_w + [2, 0, 0], Qd, r_obs0)),
        #     np.concatenate((r_ew_w + [3, 0, 0], Qd, r_obs0)),
        #     np.concatenate((r_ew_w + [4, 0, 0], Qd, r_obs0)),
        #     np.concatenate((r_ew_w + [5, 0, 0], Qd, r_obs0)),
        # ]

        target_times = [0]
        target_inputs = [u]

        # goal pose
        r_ew_w_d = r_ew_w + [5, 0, 0]
        Qd = Q_we
        r_obs0 = np.array(r_ew_w) + [0, -10, 0]

        target_states = [np.concatenate((r_ew_w_d, Qd, r_obs0))]

        # ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))

    mpc = ocs2_util.setup_ocs2_mpc_interface(
        settings_wrapper.settings, target_times, target_states, target_inputs
    )

    target_idx = 0

    x_opt = np.copy(x)

    # simulation loop
    for i in range(N):
        q, v = robot.joint_states()
        x = np.concatenate((q, v))

        # add noise to state variables
        q_noisy = q + np.random.normal(scale=Q_STDEV, size=q.shape)
        v_noisy = v + np.random.normal(scale=V_STDEV, size=v.shape)
        x_noisy = np.concatenate((q_noisy, v_noisy))

        # by using x_opt, we're basically just doing pure open-loop planning,
        # since the state never deviates from the optimal trajectory (at least
        # due to noise)
        # this does have the benefit of smoothing out the state used for
        # computation, which is important for constraint handling
        if USE_NOISY_STATE_TO_PLAN:
            mpc.setObservation(t, x_noisy, u)
        else:
            mpc.setObservation(t, x_opt, u)

        # this should be set to reflect the MPC time step
        # we can increase it if the MPC rate is faster
        if i % CTRL_PERIOD == 0:
            try:
                t0 = time.time()
                mpc.advanceMpc()
                t1 = time.time()
            except RuntimeError as e:
                print(e)
                print("exit the interpreter to proceed to plots")
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
        # x_opt_new = np.zeros(robot.ns)
        # u = np.zeros(robot.ni)
        mpc.evaluateMpcSolution(t, x_noisy, x_opt, u)
        robot.command_acceleration(u)
        # robot.command_velocity(x_opt[9:])

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()

            if settings_wrapper.settings.tray_balance_settings.enabled:
                if (
                    settings_wrapper.settings.tray_balance_settings.constraint_type
                    == ocs2.ConstraintType.Hard
                ):
                    recorder.ineq_cons[idx, :] = mpc.stateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                else:
                    recorder.ineq_cons[idx, :] = mpc.softStateInputInequalityConstraint("trayBalance", t, x, u)
            if settings_wrapper.settings.dynamic_obstacle_settings.enabled:
                recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
                    "dynamicObstacleAvoidance", t, x
                )
            if settings_wrapper.settings.collision_avoidance_settings.enabled:
                recorder.collision_pair_distance[
                    idx, :
                ] = mpc.stateInequalityConstraint("collisionAvoidance", t, x)

            r_ew_w_d = target_states[target_idx][:3]
            Q_we_d = target_states[target_idx][3:7]

            # record
            recorder.us[idx, :] = u
            recorder.xs[idx, :] = x
            recorder.xs_noisy[idx, :] = x_noisy
            recorder.r_ew_wds[idx, :] = r_ew_w_d
            recorder.r_ew_ws[idx, :] = r_ew_w
            recorder.Q_wes[idx, :] = Q_we
            recorder.Q_weds[idx, :] = Q_we_d
            recorder.v_ew_ws[idx, :] = v_ew_w
            recorder.ω_ew_ws[idx, :] = ω_ew_w

            for j, obj in enumerate(objects.values()):
                r, Q = obj.sim_obj.get_pose()
                recorder.r_ow_ws[j, idx, :] = r
                recorder.Q_wos[j, idx, :] = Q

            recorder.cmd_vels[idx, :] = robot.cmd_vel

        sim.step(step_robot=True)
        if settings_wrapper.settings.dynamic_obstacle_settings.enabled:
            obstacle.step()
        for ghost in ghosts:
            ghost.update()
        t += sim.dt
        # time.sleep(sim.dt)

        # set the target trajectories to make controller aware of dynamic
        # obstacles
        if SIM_TYPE == SimType.DYNAMIC_OBSTACLE:
            if i == 0:
                mpc.setTargetTrajectories(target_trajectories_obs1)
            elif i == 2000:
                # reset the obstacle to use again
                obstacle.reset_pose(r_obs0, (0, 0, 0, 1))
                obstacle.reset_velocity(obstacle.velocity)
                mpc.setTargetTrajectories(target_trajectories_obs2)

        if SIM_TYPE == SimType.STATIC_OBSTACLE:
            if t >= target_times[target_idx] and target_idx < len(target_times) - 1:
                target_idx += 1

        # if RECORD_VIDEO and i % VIDEO_PERIOD == 0:
        #     VIDEO_RECORDER.save_frame()
        if i % VIDEO_PERIOD == 0:
            for video in videos:
                video.save_frame()

        # every 0.5 seconds
        if DO_DYNAMIC_OBSTACLE_PHOTO_SHOOT and i % 500 == 0:
            dynamic_cam.save_frame(f"t{i}.png")

        # for taking pictures manually during a trajectory
        # if i > 1000 and i % 200 == 0:
        #     IPython.embed()

    if recorder.ineq_cons.shape[1] > 0:
        print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    # save logged data
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        if len(sys.argv) > 2:
            prefix = sys.argv[2]
        else:
            prefix = "data"
        data_file_name = prefix + "_" + TIMESTAMP
        recorder.save(DATA_DIR_PATH / data_file_name)

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

    if settings_wrapper.settings.dynamic_obstacle_settings.enabled:
        print(
            f"Min dynamic obstacle distance = {np.min(recorder.dynamic_obs_distance, axis=0)}"
        )
        recorder.plot_dynamic_obs_dist(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
