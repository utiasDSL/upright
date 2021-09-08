#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import time
import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

import mm_pybullet_sim.util as util
from mm_pybullet_sim.simulation import MobileManipulatorSimulation
from mm_pybullet_sim.recording import Recorder

from ocs2_mobile_manipulator_modified import (
    mpc_interface,
    scalar_array,
    vector_array,
    matrix_array,
    TargetTrajectories,
)

import IPython


# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 50  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)


class Obstacle:
    def __init__(self, initial_position, collision_uid, visual_uid, velocity=None):
        self.uid = pyb.createMultiBody(
            baseMass=0,  # non-dynamic body
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=list(initial_position),
            baseOrientation=(0, 0, 0, 1),
        )
        self.initial_position = initial_position

        self.velocity = velocity
        if self.velocity is None:
            self.velocity = np.zeros(3)
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(self.velocity))

    # def update_velocity(self, t):
    #     v = self.velocity_func(self.initial_position, t)
    #     pyb.resetBaseVelocity(self.uid, linearVelocity=v)

    def sample_position(self, t):
        """Sample the position of the object at a given time."""
        # assume constant velocity
        return self.initial_position + t * self.velocity
        # positions = np.array([self.initial_position + t * self.velocity for t in ts])
        # return positions


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=SIM_DT)

    N = int(DURATION / sim.dt)

    # simulation objects and model
    robot, objects, composites = sim.setup(
        obj_names=["tray", "cuboid1", "cuboid2", "cuboid3"]
    )
    tray = objects["tray"]
    cuboid1 = objects["cuboid1"]
    # cylinder1 = objects["cylinder1"]

    q, v = robot.joint_states()
    r_ew_w, Q_we = robot.link_pose()
    v_ew_w, ω_ew_w = robot.link_velocity()
    # r_tw_w, Q_wt = tray.bullet.get_pose()
    # r_ow_w, Q_wo = cuboid1.bullet.get_pose()

    # data recorder and plotter
    recorder = Recorder(
        sim.dt,
        DURATION,
        RECORD_PERIOD,
        ns=robot.ns,
        ni=robot.ni,
        n_objects=len(objects),
        control_period=CTRL_PERIOD,
        n_balance_con=23,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    for name, obj in objects.items():
        print(f"{name} CoM = {obj.body.com}")
    IPython.embed()

    r_obs0 = np.array(r_ew_w) + [0, -10, 0]
    v_obs = np.array([0, 0.4, 0])
    obs_radius = 0.1
    obs_collision_uid = pyb.createCollisionShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=obs_radius,
    )
    obs_visual_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=obs_radius,
        rgbaColor=(1, 0, 0, 1),
    )
    obstacle = Obstacle(r_obs0, -1, obs_visual_uid, velocity=v_obs)

    # r_obs5 = obstacle.sample_position(5)

    # initial time, state, and input
    t = 0.0
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    # target_times = [0, 2, 4, 6, 8, 10]
    target_times = [0]

    # setup MPC and initial EE target pose
    mpc = mpc_interface("mpc")
    t_target = scalar_array()
    for target_time in target_times:
        t_target.push_back(target_time)

    input_target = vector_array()
    for _ in target_times:
        input_target.push_back(u)

    state_target = vector_array()
    # r_ew_w_d = np.array(r_ew_w) + [2, 0, 0]
    r_ew_w_d = np.array(r_ew_w) + [2, 0, -0.5]
    # r_ew_w_d = np.array(r_ew_w) + [0, 2, 0.5]
    # r_ew_w_d = np.array(r_ew_w) + [0, -2, 0]
    # r_ew_w_d = np.array([0, -3, 1])
    Qd = Q_we
    # Qd = util.quat_multiply(Q_we, np.array([0, 0, 1, 0]))
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [1, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [2, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [3, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [4, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [4, 0, 0], Qd, r_obs0)))

    target_trajectories = TargetTrajectories(t_target, state_target, input_target)
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

        # evaluate the current MPC policy (returns an entire trajectory of
        # waypoints, starting from the current time)
        # t_result = scalar_array()
        # x_result = vector_array()
        # u_result = vector_array()
        # mpc.getMpcSolution(t_result, x_result, u_result)
        # u = u_result[0]

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
            # recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
            #     "obstacleAvoidance", t, x
            # )
            # recorder.collision_pair_distance[idx, :] = mpc.stateInequalityConstraint(
            #     "selfCollision", t, x
            # )

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

            # r_tw_w, Q_wt = tray.bullet.get_pose()
            # recorder.r_tw_ws[idx, :] = r_tw_w
            # recorder.Q_wts[idx, :] = Q_wt
            #
            # if len(objects) > 1:
            #     r_ow_w, Q_wo = cuboid1.bullet.get_pose()
            #     recorder.r_ow_ws[idx, :] = r_ow_w
            #     recorder.Q_wos[idx, :] = Q_wo

            recorder.cmd_vels[idx, :] = robot.cmd_vel

            if (recorder.ineq_cons[idx, :] < -1).any():
                print("constraint less than -1")
                IPython.embed()
                break

        # print(f"cmd_vel before step = {robot.cmd_vel}")
        sim.step(step_robot=True)
        # _, v_test = robot.joint_states()
        # print(f"v after step = {v_test}")
        t += sim.dt
        if t >= target_times[target_idx] and target_idx < len(target_times) - 1:
            target_idx += 1

        # if t > 9:
        #     IPython.embed()
        #     return
        # if i % 100 == 0:
        #     IPython.embed()

    print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    if "--save" in sys.argv:
        fname = "balance_data_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
