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

import tray_balance_sim.util as util
from tray_balance_sim.simulation import MobileManipulatorSimulation
from tray_balance_sim.recording import Recorder
from tray_balance_sim.trajectory import QuinticTimeScaling, PointToPoint

import IPython

# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 10  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 30.0  # duration of trajectory (s)
BEAT_DT = 1.0

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_DIR = Path("/media/adam/Data/PhD/Videos/Dance/")
VIDEO_PATH = VIDEO_DIR / ("no_object_dist0.1_" + TIMESTAMP)

FRAMES_PATH = VIDEO_PATH
VIDEO_PERIOD = 40  # 25 frames per second with 1000 steps per second
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

RECORD_VIDEO = False


def generate_target_config(q0, dt, n=1000):
    """Generate a new configuration qd with time interval dt."""
    q_high = np.array([1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    q_low = -q_high
    qds = q0 + np.random.uniform(q_low, q_high, (n, 9))
    qds_max = np.max(np.abs(qds - q0), axis=1)

    # apply velocity limit
    qds = qds[qds_max / dt < 1.0, :]

    # here I'm choosing the largest norm, but perhaps that isn't always what we
    # want (just get rid of some boring small ones)
    qd_norms = np.linalg.norm(qds - q0, axis=1)
    idx = np.argmax(qd_norms)
    return qds[idx, :]


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=SIM_DT)

    N = int(DURATION / sim.dt)

    # simulation objects and model
    robot, objects, composites = sim.setup(obj_names=[])

    q, v = robot.joint_states()
    r_ew_w, Q_we = robot.link_pose()
    v_ew_w, ω_ew_w = robot.link_velocity()

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

    q0 = q
    time_scaling = QuinticTimeScaling(BEAT_DT)
    waypoint = generate_target_config(q0, BEAT_DT)
    print(waypoint)
    trajectory1 = PointToPoint(q0, waypoint, time_scaling, BEAT_DT)
    trajectory2 = PointToPoint(waypoint, q0, time_scaling, BEAT_DT)
    trajectories = [trajectory1, trajectory2]

    Kp = np.eye(9)
    num_traj = 2

    traj_idx = 0
    t_traj = 0.0

    frame_num = 0

    for i in range(N):
        q, v = robot.joint_states()

        if t_traj >= BEAT_DT:
            t_traj = 0
            traj_idx = (traj_idx + 1) % num_traj

        if i % CTRL_PERIOD == 0:
            qd, vd, _ = trajectories[traj_idx].sample(t_traj)
            u = Kp @ (qd.flatten() - q) + vd.flatten()

        robot.command_velocity(u, bodyframe=False)

        sim.step(step_robot=False)
        t += sim.dt
        t_traj += sim.dt
        time.sleep(sim.dt)

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


if __name__ == "__main__":
    main()
