import time

import numpy as np

import tray_balance_constraints as core
from tray_balance_ocs2 import bindings
from tray_balance_ocs2.robot import PinocchioRobot
from tray_balance_ocs2.wrappers import TargetTrajectories, ControllerSettings
from tray_balance_ocs2.trajectory import StateInputTrajectory

import IPython


# TODO somewhere we can add a method to convert a plan to JointTrajectory msg


class StateInputMapping:
    def __init__(self, dims):
        self.dims = dims

    def xu2qva(self, x, u=None):
        q = x[: self.dims.q]
        v = x[self.dims.q : self.dims.q + self.dims.v]
        a = x[self.dims.q + self.dims.v : self.dims.q + 2 * self.dims.v]
        return q, v, a

    def qva2xu(self, q, v, a):
        x = np.concatenate((q, v, a))
        return x, None


class QuinticPoint:
    def __init__(self, t, q, v, a):
        self.t = t
        self.q = q
        self.v = v
        self.a = a


class QuinticInterpolator:
    def __init__(self, p1, p2):
        self.t0 = p1.t
        T = p2.t - p1.t
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],
            ]
        )
        b = np.array([p1.q, p2.q, p1.v, p2.v, p1.a, p2.a])
        self.coeffs = np.linalg.solve(A, b)

    def eval(self, t):
        t = np.array(t) - self.t0  # normalize time

        zs = np.zeros_like(t)
        os = np.ones_like(t)

        w = np.array([os, t, t ** 2, t ** 3, t ** 4, t ** 5])
        dw = np.array([zs, os, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4])
        ddw = np.array([zs, zs, 2 * os, 6 * t, 12 * t ** 2, 20 * t ** 3])

        q = w.T @ self.coeffs
        v = dw.T @ self.coeffs
        a = ddw.T @ self.coeffs

        return q, v, a


class JointVelocityController:
    def __init__(self, Kp, mapping, trajectory):
        self.Kp = Kp
        self.mapping = mapping
        self.interpolator = None
        self.update(trajectory)

    def compute_desired(self, t):
        # make sure we are in the correct time range
        if t < self.trajectory.ts[self.traj_idx]:
            raise ValueError(
                f"We are going back in time with t = {t} but our trajectory at t = {self.trajectory.ts[self.traj_idx]}"
            )
        if t > self.trajectory.ts[-1]:
            raise ValueError(
                f"We are at t = {t} but our trajectory only goes to t = {self.trajectory.ts[-1]}"
            )

        # find the new point in the trajectory
        for i in range(self.traj_idx, len(self.trajectory) - 1):
            if self.trajectory.ts[i] <= t <= self.trajectory.ts[i + 1]:
                break

        # TODO could short-circuit if the timestep is small

        # update the interpolator
        if self.traj_idx != i or self.interpolator is None:
            t1, x1, u1 = self.trajectory[i]
            q1, v1, a1 = self.mapping.xu2qva(x1, u1)
            p1 = QuinticPoint(t=t1, q=q1, v=v1, a=a1)

            t2, x2, u2 = self.trajectory[i + 1]
            q2, v2, a2 = self.mapping.xu2qva(x2, u2)
            p2 = QuinticPoint(t=t2, q=q2, v=v2, a=a2)

            self.traj_idx = i
            self.interpolator = QuinticInterpolator(p1, p2)

        # do interpolation
        qd, vd, ad = self.interpolator.eval(t)
        return self.mapping.qva2xu(qd, vd, ad)[0]

    def compute_input(self, t, x):
        xd = self.compute_desired(t)

        qd, vd, _ = self.mapping.xu2qva(xd)
        q, _, _ = self.mapping.xu2qva(x)

        return self.Kp @ (qd - q) + vd

    # update optimal trajectory
    def update(self, trajectory):
        self.trajectory = trajectory
        self.traj_idx = 0


# TODO is this actually useful? why not just use the above?
class DoubleIntegrator:
    def __init__(self):
        pass


class ControllerModel:
    """Contains system model: robot, objects, and other settings."""

    def __init__(self, settings, robot):
        self.settings = settings
        self.robot = robot

    @classmethod
    def from_config(cls, config, x0=None):
        settings = ControllerSettings(config=config, x0=x0)
        robot = PinocchioRobot(config=config["robot"])
        return cls(settings, robot)

    def update(self, x, u=None):
        """Update model with state x and input u. Required before calling other methods."""
        self.robot.forward(x, u)

    def balancing_constraints(self):
        """Evaluate the balancing constraints at time t and state x."""
        _, Q_we = self.robot.link_pose()
        _, ω_ew_w = self.robot.link_velocity()
        a_ew_w, α_ew_w = self.robot.link_acceleration()
        C_we = core.util.quaternion_to_matrix(Q_we)
        return core.bindings.balancing_constraints(
            self.settings.objects, self.settings.gravity, C_we, ω_ew_w, a_ew_w, α_ew_w
        )

    def support_area_distances(self):
        """Compute shortest distance of intersection of gravity vector with
        support plane from support area for each object.

        A negative distance indicates that the intersection is inside the
        support area.

        `update` must have been called first.
        """
        _, Q_we = self.robot.link_pose()
        dists = []
        for obj in self.settings.objects:
            dists.append(core.util.support_area_distance(obj, Q_we))
        return np.array(dists)

    def angle_between_acc_and_normal(self):
        """Compute the angle between the total acceleration vector and EE normal vector.

        `update` must have been called first.
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

    def ddC_we_norm(self):
        _, Q_we = self.robot.link_pose()
        C_we = core.util.quaternion_to_matrix(Q_we)
        _, ω_ew_w = self.robot.link_velocity()
        _, α_ew_w = self.robot.link_acceleration()

        Sα = core.math.skew3(α_ew_w)
        Sω = core.math.skew3(ω_ew_w)
        ddC_we = (Sα + Sω @ Sω) @ C_we
        return np.linalg.norm(ddC_we, ord=2)


class ControllerManager:
    """High-level control management:
    - rollout MPC to generate plans
    - generate low-level controllers to execute in simulation"""

    def __init__(self, model, ref_trajectory, timestep):
        self.model = model
        self.ref = ref_trajectory
        self.timestep = timestep

        # MPC
        self.mpc = bindings.ControllerInterface(self.model.settings)
        self.mpc.reset(self.ref)

        self.last_planning_time = -np.infty
        self.x_opt = np.zeros(self.model.settings.dims.x)
        self.u_opt = np.zeros(self.model.settings.dims.u)

        # TODO can I log this directly?
        self.replanning_durations = []

    @classmethod
    def from_config(cls, config, x0=None):
        model = ControllerModel.from_config(config, x0=x0)

        # control should be done every timestep
        timestep = config["timestep"]

        # compute EE pose
        model.update(x=model.settings.initial_state)
        r_ew_w, Q_we = model.robot.link_pose()

        # reference pose trajectory
        ref_trajectory = TargetTrajectories.from_config(
            config, r_ew_w, Q_we, np.zeros(model.settings.dims.u)
        )
        return cls(model, ref_trajectory, timestep)

    def warmstart(self):
        """Do the first optimize to get things warmed up."""
        x0 = self.model.settings.initial_state
        u0 = np.zeros(self.model.settings.dims.u)
        self.mpc.setObservation(0, x0, u0)

        self.mpc.advanceMpc()
        self.last_planning_time = 0

    def step(self, t, x):
        """Evaluate MPC at a single timestep, replanning if needed."""
        self.mpc.setObservation(t, x, self.u_opt)

        # replan if `timestep` has elapsed since the last time
        if t >= self.last_planning_time + self.timestep:
            t0 = time.time()
            self.mpc.advanceMpc()
            t1 = time.time()

            self.last_planning_time = t
            self.replanning_durations.append(t1 - t0)

        # evaluate the current solution
        try:
            self.mpc.evaluateMpcSolution(t, x, self.x_opt, self.u_opt)
        except:
            IPython.embed()

        return self.x_opt, self.u_opt

    def plan(self, timestep, duration):
        """Construct a new plan by rolling out the MPC.

        Parameters:
            timestep: timestep of the planning loop---not the same as the MPC
                      timestep (the rate at which a new trajectory is optimized)
            duration: duration of the plan

        Returns: the plan (a full state-input trajectory)
        """
        ts = []
        xs = []
        us = []

        t = 0.0
        x = self.model.settings.initial_state
        while t <= duration:
            x, u = self.step(t, x)
            ts.append(t)
            xs.append(x.copy())
            us.append(u.copy())
            t += timestep

        return StateInputTrajectory(ts, xs, us)
