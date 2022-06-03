#!/usr/bin/env python3
"""Testing of the robust balancing constraints"""
import sys
from pathlib import Path

import numpy as np
import pybullet as pyb
import pybullet_data
from pyb_utils.ghost import GhostSphere

from upright_sim import util, ocs2_util, robustness
from upright_sim.simulation import MobileManipulatorSimulation
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2

import IPython


DATA_DRIVE_PATH = Path("/media/adam/Data/PhD/Data/IROS-RAL22/v2")
DATA_DRIVE_FILE = "static-obstacle/static_cups4_robust1_ctrlprd100_2022-02-18_23-44-44.npz"

MAX_TIME = 10.0
LINE_WIDTH = 5

SIM_DT = 0.001
CUP_CONFIG = ["tray", "cylinder1_cup", "cylinder2_cup", "cylinder3_cup"]


def compute_data_length(times, max_time):
    """Compute length of data that fits within time max_time."""
    data_length = times.shape[0]

    for i in range(times.shape[0]):
        if times[i] > max_time:
            data_length = i
            break
    return data_length


# def draw_curve(waypoints, rgb=(1, 0, 0), dist=0.05, linewidth=1, dashed=False):
#     # process waypoints to space them (roughly) evenly
#     visual_points = [waypoints[0, :]]
#     for i in range(1, len(waypoints)):
#         d = np.linalg.norm(waypoints[i, :] - visual_points[-1])
#         if d >= dist:
#             visual_points.append(waypoints[i, :])
#
#     step = 2 if dashed else 1
#     for i in range(0, len(visual_points) - 1, step):
#         start = visual_points[i]
#         end = visual_points[i + 1]
#         pyb.addUserDebugLine(
#             list(start),
#             list(end),
#             lineColorRGB=rgb,
#             lineWidth=linewidth,
#         )


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=SIM_DT)
    robot, objects, composites = sim.setup(
        CUP_CONFIG,
        load_static_obstacles=True,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--front":
        # front view
        pyb.resetDebugVisualizerCamera(
            cameraDistance=4.8,
            cameraYaw=78,
            cameraPitch=-28.2,
            cameraTargetPosition=[2.77, 0.043, 0.142],
        )
    else:
        # side view
        pyb.resetDebugVisualizerCamera(
            cameraDistance=2.97,
            cameraYaw=0,
            cameraPitch=-43.8,
            cameraTargetPosition=[2.77, 0.043, 0.142],
        )

    q, v = robot.joint_states()
    x = np.concatenate((q, v))

    settings_wrapper = ocs2_util.TaskSettingsWrapper(composites, x)
    settings_wrapper.settings.tray_balance_settings.enabled = True
    settings_wrapper.settings.tray_balance_settings.robust = True

    r_ew_w, Q_we = robot.link_pose()

    ghosts = []  # ghost (i.e., pure visual) objects
    robustness.set_bounding_spheres(
        robot,
        objects,
        settings_wrapper.settings,
        target=r_ew_w + [0, 0, 0.1],
        sim_timestep=SIM_DT,
        plot_point_cloud=False,
        k=1,
    )

    for ball in settings_wrapper.settings.tray_balance_settings.robust_params.balls:
        ghosts.append(
            GhostSphere(
                ball.radius,
                position=ball.center,
                parent_body_uid=robot.uid,
                parent_link_index=robot.tool_idx,
                color=(0, 0, 1, 0.3),
            )
        )

    with np.load(DATA_DRIVE_PATH / DATA_DRIVE_FILE) as data:
        times = data["ts"]
        xs = data["xs"]
        r_ew_ws = data["r_ew_ws"]
        r_ew_wds = data["r_ew_wds"]

    data_length = compute_data_length(times, MAX_TIME)

    base_position = np.hstack((xs[:, :2], 0.1 * np.ones((xs.shape[0], 1))))

    # interpolate between start and end of desired EE trajectory
    r_ew_wds = np.linspace(r_ew_ws[0, :], r_ew_wds[-1, :], data_length)

    # draw from end to start to ensure the last point is drawn despite being
    # dashed (otherwise it looks like the error is large)
    util.draw_curve(
        np.flip(r_ew_wds, axis=0), linewidth=LINE_WIDTH, dashed=True, rgb=(0, 0, 0)
    )
    util.draw_curve(
        r_ew_ws[:data_length, :],
        linewidth=LINE_WIDTH,
        dashed=False,
        rgb=(0.839, 0.153, 0.157),
    )
    util.draw_curve(
        base_position[:data_length, :],
        linewidth=LINE_WIDTH,
        dashed=False,
        rgb=(0, 1, 0),
    )

    IPython.embed()


if __name__ == "__main__":
    main()
