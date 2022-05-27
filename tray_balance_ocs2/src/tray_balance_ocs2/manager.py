import time

import numpy as np

import tray_balance_constraints as core
from tray_balance_ocs2 import bindings
from tray_balance_ocs2.robot import PinocchioRobot
from tray_balance_ocs2.wrappers import TargetTrajectories, ControllerSettings

import IPython


# TODO somewhere we can add a method to convert a plan to JointTrajectory
# TODO does it make sense to include interpolation functionality?
# class TrackingController:
#     # tracks a plan
#     def __init__(self, trajectory):
#         pass
#
# class ClosedLoopController:
#     pass


# TODO: make a jerk input version, too (we can use better integration then)
class JointVelocityController:
    pass


class JointTrajectory:
    pass


class ControllerManager:
    """High-level control management:
    - rollout MPC to generate plans
    - generate low-level controllers to execute in simulation"""

    def __init__(self, config, x0=None, operating_trajectory=None, logger=None):
        self.config = config
        self.settings = ControllerSettings(config, x0, operating_trajectory)
        self.robot = PinocchioRobot(config["robot"])
        self.logger = logger

        # control should be done every ctrl_period timesteps
        self.period = config["control_period"]

        # compute EE pose
        self.robot.forward(self.settings.initial_state)
        r_ew_w, Q_we = self.robot.link_pose()

        # reference pose trajectory
        self.ref = TargetTrajectories.from_config(
            self.config, r_ew_w, Q_we, np.zeros(self.settings.dims.u)
        )

        # MPC
        self.mpc = bindings.ControllerInterface(self.settings)
        self.mpc.reset(self.ref)

        self.last_planning_time = -np.infty
        self.x_opt = np.zeros(self.settings.dims.x)
        self.u_opt = np.zeros(self.settings.dims.u)

        self.replanning_durations = []

    def warmstart(self):
        """Do the first optimize to get things warmed up."""
        x0 = self.settings.initial_state
        u0 = np.zeros(self.settings.dims.u)
        self.mpc.setObservation(0, x0, u0)

        self.mpc.advanceMpc()
        self.last_planning_time = 0

    def step(self, t, x):
        """Evaluate MPC at a single timestep, replanning if needed."""
        self.mpc.setObservation(t, x, self.u_opt)

        # replan if `period` has elapsed since the last time
        if t >= self.last_planning_time + self.period:
            t0 = time.time()
            self.mpc.advanceMpc()
            t1 = time.time()

            self.last_planning_time = t
            self.replanning_durations.append(t1 - t0)

        # evaluate the current solution
        self.mpc.evaluateMpcSolution(t, x, self.x_opt, self.u_opt)

        # TODO check if time to log now
        if self.logger is not None:
            self.robot.forward(x, self.u_opt)

            if self.settings.tray_balance_settings.enabled:
                self.logger.append(
                    "balancing_constraints", self.balancing_constraints(t, x)
                )

            self.logger.append("sa_dists", self.support_area_distances())
            self.logger.append("orn_err", self.angle_between_acc_and_normal())

        return self.x_opt, self.u_opt

    def plan(self, timestep, duration):
        """Construct a new plan by rolling out the MPC.

        Parameters:
            timestep: timestep of the planning loop
            duration: duration of the plan

        Returns: the plan (a full state-input trajectory)
        """
        ts = []
        xs = []
        us = []

        t = 0
        while t <= duration:
            self.mpc.step(t, self.x_opt)
            ts.append(t)
            xs.append(self.x_opt.copy())
            us.append(self.u_opt.copy())
            t += timestep

        return JointTrajectory(ts, xs, us)

        # ts = timestep * np.arange(total_steps)
        # xs = np.zeros((total_steps, self.robot.dims.x))
        # us = np.zeros((total_steps, self.robot.dims.u))
        #
        # t = 0
        # x = x0
        # u = np.zeros(self.robot.dims.u)
        # for i in range(total_steps):
        #     if i % replan_steps == 0:
        #         self.mpc.setObservation(t, x, u)
        #         self.mpc.advanceMpc()
        #     self.mpc.evaluateMpcSolution(t, x, x, u)
        #     xs[i, :] = x
        #     us[i, :] = u
        #     t += timestep
        #
        # return core.trajectory.Trajectory(ts=ts, xs=xs, us=us)

    def balancing_constraints(self, t, x):
        """Evaluate the balancing constraints at time t and state x."""
        if (
            self.settings.tray_balance_settings.constraint_type
            == bindings.ConstraintType.Hard
        ):
            return self.mpc.stateInputInequalityConstraint("trayBalance", t, x, u_opt)
        return self.mpc.softStateInputInequalityConstraint("trayBalance", t, x, u_opt)

    def support_area_distances(self):
        """Compute shortest distance of intersection of gravity vector with
        support plane from support area for each object.

        A negative distance indicates that the intersection is inside the
        support area.

        self.robot.forward(x, u) must have been called first.
        """
        _, Q_we = self.robot.link_pose()
        dists = []
        for obj in self.settings.objects:
            dists.append(core.util.support_area_distance(obj, Q_we))
        return np.array(dists)

    def angle_between_acc_and_normal(self):
        """Compute the angle between the total acceleration vector and EE normal vector.

        self.robot.forward(x, u) must have been called first.
        """

        _, Q_we = self.robot.link_pose()
        _, ω_ew_w = self.robot.link_velocity()
        a_ew_w, α_ew_w = self.robot.link_acceleration()
        C_we = core.util.quaternion_to_matrix(Q_we)

        # find EE normal vector in the world frame
        z_e = np.array([0, 0, 1])
        z_w = C_we @ z_e

        # compute direction (unit vector) of total acceleration (inertial + gravity)
        total_acc = a_ew_w - self.settings.gravity
        total_acc_direction = total_acc / np.linalg.norm(total_acc)

        # compute the angle between the two
        angle = np.arccos(z_w @ total_acc_direction)
        return angle
