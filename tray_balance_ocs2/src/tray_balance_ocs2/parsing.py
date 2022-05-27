import os

import numpy as np
import rospkg

import tray_balance_constraints as core
from tray_balance_ocs2 import bindings
from tray_balance_ocs2.robot import PinocchioRobot

import IPython


# TODO
# TODO somewhere we can add a method to convert a plan to JointTrajectory
# TODO does it make sense to include interpolation functionality?
# class TrackingController:
#     # tracks a plan
#     def __init__(self, trajectory):
#         pass
#
# class ClosedLoopController:
#     pass


# class Plan:
#     def __init__(self, ts, xs, us):
#         self.ts = ts
#         self.xs = xs
#         self.us = us
#
#     @classmethod
#     def plan(cls, x0, robot, mpc, timestep, total_steps, replan_steps):
#         """Construct a new plan by rolling out the MPC.
#
#         Parameters:
#             x0: initial state
#             robot: the robot model to use
#             mpc: the model predictive controller that generates optimal trajectories
#             timestep: timestep of the planning loop
#             total_steps: total number of timesteps
#             replan_steps: replan every `replan_steps` steps
#
#         Returns: the plan (a full state-input trajectory)
#         """
#         self.ts = timestep * np.arange(total_steps)
#         self.xs = np.zeros((total_steps, robot.nx))
#         self.us = np.zeros((total_steps, robot.nu))
#
#         t = 0
#         x = x0
#         u = np.zeros(robot.nu)
#         for i in range(total_steps):
#             if i % replan_steps == 0:
#                 mpc.setObservation(t, x, u)
#                 mpc.advanceMpc()
#             mpc.evaluateMpcSolution(t, x, x, u)
#             xs[i, :] = x
#             us[i, :] = u
#             t += timestep
#
#         return cls(ts=ts, xs=xs, us=us)


# class PoseTrajectory(core.trajectory.Trajectory):
#     """Reference trajectory of a sequence of waypoints.
#
#     If there is more than one waypoint, then the desired pose is interpolated
#     between them.
#     """
#
#     def __init__(self, ts, xs, us):
#         super().__init__(ts, xs, us)
#
#     @classmethod
#     def from_config(cls, config, r_ew_w, Q_we, u):
#         """Parse the reference trajectory from the config dict."""
#         times = []
#         inputs = []
#         states = []
#         for waypoint in config["waypoints"]:
#             secs = core.parsing.millis_to_secs(waypoint["millis"])
#
#             r_ew_w_d = r_ew_w + waypoint["position"]
#             Q_we_d = core.math.quat_multiply(Q_we, waypoint["orientation"])
#             r_obs = np.zeros(3)  # r_ew_w + np.array([0, -10, 0])
#             state = np.concatenate((r_ew_w_d, Q_we_d, r_obs))
#
#             times.append(secs)
#             inputs.append(np.copy(u))
#             states.append(state)
#         return cls(np.array(times), np.array(states), np.array(inputs))
#
#     @staticmethod
#     def pose(state):
#         """Extract EE position and orientation from a generic reference state."""
#         # this is useful because the state also contains info about a dynamic
#         # obstacle
#         r = state[:3]
#         Q = state[3:7]
#         return r, Q


# wrapper around bound TargetTrajectories
class TargetTrajectories(bindings.TargetTrajectories):
    def __init__(self, ts, xs, us):
        ts_ocs2 = bindings.scalar_array()
        xs_ocs2 = bindings.vector_array()
        us_ocs2 = bindings.vector_array()
        for t, x, u in zip(ts, xs, us):
            ts_ocs2.push_back(t)
            xs_ocs2.push_back(x)
            us_ocs2.push_back(u)
        super().__init__(ts_ocs2, xs_ocs2, us_ocs2)

    @classmethod
    def from_config(cls, config, r_ew_w, Q_we, u):
        ts = []
        xs = []
        us = []
        for waypoint in config["waypoints"]:
            t = core.parsing.millis_to_secs(waypoint["millis"])

            r_ew_w_d = r_ew_w + waypoint["position"]
            Q_we_d = core.math.quat_multiply(Q_we, waypoint["orientation"])
            r_obs = np.zeros(3)
            x = np.concatenate((r_ew_w_d, Q_we_d, r_obs))

            ts.append(t)
            us.append(np.copy(u))
            xs.append(x)
        return cls(ts, xs, us)

    @staticmethod
    def _state_to_pose(x):
        return x[:3], x[3:7]

    def poses(self):
        """Iterate over all poses in the trajectory."""
        for x in self.xs:
            yield self._state_to_pose(x)

    def get_desired_pose(self, t):
        """Extract EE position and orientation from a generic reference state."""
        x = self.get_desired_state(t)
        return self._state_to_pose(x)


# def parse_reference_trajectory(config, r_ew_w, Q_we, u):
#     # TODO, ideally this could just be kept as a TargetTrajectories
#     # object---can I sample it?
#     times = []
#     inputs = []
#     states = []
#     for waypoint in config["waypoints"]:
#         secs = core.parsing.millis_to_secs(waypoint["millis"])
#
#         r_ew_w_d = r_ew_w + waypoint["position"]
#         Q_we_d = core.math.quat_multiply(Q_we, waypoint["orientation"])
#         r_obs = np.zeros(3)  # r_ew_w + np.array([0, -10, 0])
#         state = np.concatenate((r_ew_w_d, Q_we_d, r_obs))
#
#         times.append(secs)
#         inputs.append(np.copy(u))
#         states.append(state)
#     return TargetTrajectories(np.array(times), np.array(states), np.array(inputs))


# def make_target_trajectories(trajectory):
#     """Construct the target trajectories object for OCS2."""
#     ts = bindings.scalar_array()
#     xs = bindings.vector_array()
#     us = bindings.vector_array()
#     for i in range(len(trajectory)):
#         ts.push_back(trajectory.ts[i])
#         xs.push_back(trajectory.xs[i, :])
#         us.push_back(trajectory.us[i, :])
#     return bindings.TargetTrajectories(ts, xs, us)


def parse_control_settings(config, x0=None, operating_trajectory=None):
    settings = bindings.ControllerSettings()

    settings.method = bindings.ControllerSettings.Method.DDP

    settings.end_effector_link_name = config["robot"]["tool_link_name"]
    settings.robot_base_type = bindings.robot_base_type_from_string(
        config["robot"]["base_type"]
    )

    # gravity
    settings.gravity = config["gravity"]

    # dimensions
    settings.dims.q = config["robot"]["dims"]["q"]
    settings.dims.v = config["robot"]["dims"]["v"]
    settings.dims.x = config["robot"]["dims"]["x"]
    settings.dims.u = config["robot"]["dims"]["u"]

    # initial state can be passed in directly (for example to match exactly
    # a simulation) or parsed from the config
    if x0 is None:
        settings.initial_state = core.parsing.parse_array(
            config["robot"]["x0"]
        )
    else:
        settings.initial_state = x0
    assert settings.initial_state.shape == (settings.dims.x,)

    # weights for state, input, and EE pose
    settings.input_weight = core.parsing.parse_diag_matrix_dict(
        config["weights"]["input"]
    )
    settings.state_weight = core.parsing.parse_diag_matrix_dict(
        config["weights"]["state"]
    )
    settings.end_effector_weight = core.parsing.parse_diag_matrix_dict(
        config["weights"]["end_effector"]
    )
    assert settings.input_weight.shape == (settings.dims.u, settings.dims.u)
    assert settings.state_weight.shape == (settings.dims.x, settings.dims.x)
    assert settings.end_effector_weight.shape == (6, 6)

    # input limits
    settings.input_limit_lower = core.parsing.parse_array(
        config["limits"]["input"]["lower"]
    )
    settings.input_limit_upper = core.parsing.parse_array(
        config["limits"]["input"]["upper"]
    )
    assert settings.input_limit_lower.shape == (settings.dims.u,)
    assert settings.input_limit_upper.shape == (settings.dims.u,)

    # state limits
    settings.state_limit_lower = core.parsing.parse_array(
        config["limits"]["state"]["lower"]
    )
    settings.state_limit_upper = core.parsing.parse_array(
        config["limits"]["state"]["upper"]
    )
    assert settings.state_limit_lower.shape == (settings.dims.x,)
    assert settings.state_limit_upper.shape == (settings.dims.x,)

    # URDFs
    settings.robot_urdf_path = core.parsing.parse_ros_path(
        config["robot"]["urdf"]
    )
    settings.obstacle_urdf_path = core.parsing.parse_ros_path(
        config["static_obstacles"]["urdf"]
    )

    # task info file (Boost property tree format)
    settings.ocs2_config_path = core.parsing.parse_ros_path(config["infofile"])
    settings.lib_folder = "/tmp/ocs2"

    # operating points
    if operating_trajectory is not None:
        settings.use_operating_points = True
        for i in range(len(operating_trajectory)):
            settings.operating_times.push_back(operating_trajectory.ts[i])
            settings.operating_states.push_back(operating_trajectory.xs[i, :])
            settings.operating_inputs.push_back(operating_trajectory.us[i, :])

    # tray balance settings
    settings.tray_balance_settings.enabled = config["balancing"]["enabled"]
    settings.tray_balance_settings.constraint_type = bindings.ConstraintType.Soft
    settings.tray_balance_settings.mu = core.parsing.parse_number(
        config["balancing"]["mu"]
    )
    settings.tray_balance_settings.delta = core.parsing.parse_number(
        config["balancing"]["delta"]
    )

    ctrl_objects = core.parsing.parse_control_objects(config)
    settings.tray_balance_settings.objects = ctrl_objects

    settings.tray_balance_settings.constraints_enabled.normal = config[
        "balancing"
    ]["enable_normal_constraint"]
    settings.tray_balance_settings.constraints_enabled.friction = config[
        "balancing"
    ]["enable_friction_constraint"]
    settings.tray_balance_settings.constraints_enabled.zmp = config[
        "balancing"
    ]["enable_zmp_constraint"]

    # alternative inertial alignment objective
    # tries to keep tray/EE normal aligned with the negative acceleration
    # vector
    settings.inertial_alignment_settings.enabled = config[
        "inertial_alignment"
    ]["enabled"]
    settings.inertial_alignment_settings.use_angular_acceleration = config[
        "inertial_alignment"
    ]["use_angular_acceleration"]
    settings.inertial_alignment_settings.weight = config["inertial_alignment"][
        "weight"
    ]
    settings.inertial_alignment_settings.r_oe_e = ctrl_objects[
        -1
    ].body.com_ellipsoid.center()  # TODO could specify index in config

    # collision avoidance settings
    settings.collision_avoidance_settings.enabled = False
    # fmt: off
    for pair in [
        # the base with most everything
        ("base_collision_link_0", "table1_link_0"),
        ("base_collision_link_0", "table2_link_0"),
        ("base_collision_link_0", "table3_link_0"),
        ("base_collision_link_0", "table4_link_0"),
        ("base_collision_link_0", "table5_link_0"),
        ("base_collision_link_0", "chair1_1_link_0"),
        ("base_collision_link_0", "chair1_2_link_0"),
        ("base_collision_link_0", "chair2_1_link_0"),
        ("base_collision_link_0", "chair3_1_link_0"),
        ("base_collision_link_0", "chair3_2_link_0"),
        ("base_collision_link_0", "chair4_1_link_0"),
        ("base_collision_link_0", "chair4_2_link_0"),

        # wrist and tables
        ("wrist_collision_link_0", "table1_link_0"),
        ("wrist_collision_link_0", "table2_link_0"),
        ("wrist_collision_link_0", "table3_link_0"),
        ("wrist_collision_link_0", "table4_link_0"),
        ("wrist_collision_link_0", "table5_link_0"),

        # wrist and shoulder
        ("wrist_collision_link_0", "shoulder_collision_link_0"),

        # elbow and tables
        ("elbow_collision_link_0", "table1_link_0"),
        ("elbow_collision_link_0", "table2_link_0"),
        ("elbow_collision_link_0", "table3_link_0"),
        ("elbow_collision_link_0", "table4_link_0"),
        ("elbow_collision_link_0", "table5_link_0"),

        # elbow and tall chairs
        ("elbow_collision_link_0", "chair3_1_link_0"),
        ("elbow_collision_link_0", "chair4_2_link_0"),
        ("elbow_collision_link_0", "chair2_1_link_0"),
    ]:
        settings.collision_avoidance_settings.collision_link_pairs.push_back(pair)
    # fmt: on
    settings.collision_avoidance_settings.minimum_distance = 0
    settings.collision_avoidance_settings.mu = 1e-2
    settings.collision_avoidance_settings.delta = 1e-3

    # dynamic obstacle settings
    settings.dynamic_obstacle_settings.enabled = False
    settings.dynamic_obstacle_settings.obstacle_radius = 0.1
    settings.dynamic_obstacle_settings.mu = 1e-2
    settings.dynamic_obstacle_settings.delta = 1e-3

    for sphere in [
        bindings.CollisionSphere(
            name="elbow_collision_link",
            parent_frame_name="ur10_arm_forearm_link",
            offset=np.zeros(3),
            radius=0.15,
        ),
        bindings.CollisionSphere(
            name="forearm_collision_sphere_link1",
            parent_frame_name="ur10_arm_forearm_link",
            offset=np.array([0, 0, 0.2]),
            radius=0.15,
        ),
        bindings.CollisionSphere(
            name="forearm_collision_sphere_link2",
            parent_frame_name="ur10_arm_forearm_link",
            offset=np.array([0, 0, 0.4]),
            radius=0.15,
        ),
        bindings.CollisionSphere(
            name="wrist_collision_link",
            parent_frame_name="ur10_arm_wrist_3_link",
            offset=np.zeros(3),
            radius=0.15,
        ),
    ]:
        settings.dynamic_obstacle_settings.collision_spheres.push_back(sphere)

    return settings


class ControllerManager:
    """High-level control management:
        - rollout MPC to generate plans
        - generate low-level controllers to execute in simulation"""
    def __init__(self, config, x0=None, operating_trajectory=None):
        self.config = config
        self.settings = parse_control_settings(config, x0, operating_trajectory)
        self.robot = PinocchioRobot(config["robot"])

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

    def get_num_balance_constraints(self):
        if self.settings.tray_balance_settings.bounded:
            return self.settings.tray_balance_settings.objects.num_constraints()
        return self.settings.tray_balance_settings.config.num_constraints()

    def get_num_collision_avoidance_constraints(self):
        if self.settings.collision_avoidance_settings.enabled:
            return len(self.settings.collision_avoidance_settings.collision_link_pairs)
        return 0

    def get_num_dynamic_obstacle_constraints(self):
        if self.settings.dynamic_obstacle_settings.enabled:
            return len(self.settings.dynamic_obstacle_settings.collision_spheres)
        return 0

    @property
    def objects(self):
        return self.settings.tray_balance_settings.objects

    def plan(self, timestep, total_steps, replan_steps):
        """Construct a new plan by rolling out the MPC.

        Parameters:
            timestep: timestep of the planning loop
            total_steps: total number of timesteps
            replan_steps: replan every `replan_steps` steps

        Returns: the plan (a full state-input trajectory)
        """
        ts = timestep * np.arange(total_steps)
        xs = np.zeros((total_steps, self.robot.dims.x))
        us = np.zeros((total_steps, self.robot.dims.u))

        t = 0
        x = x0
        u = np.zeros(self.robot.dims.u)
        for i in range(total_steps):
            if i % replan_steps == 0:
                self.mpc.setObservation(t, x, u)
                self.mpc.advanceMpc()
            self.mpc.evaluateMpcSolution(t, x, x, u)
            xs[i, :] = x
            us[i, :] = u
            t += timestep

        return core.trajectory.Trajectory(ts=ts, xs=xs, us=us)

    # TODO ideally I would compute r_ew_w and Q_we based on a Pinocchio model
    # of the robot included in this repo
    # def reference_trajectory(self, r_ew_w, Q_we):
    #     return PoseTrajectory.from_config(
    #         self.config, r_ew_w, Q_we, np.zeros(self.settings.dims.u)
    #     )
    #
    # def controller(self, reference_trajectory):
    #     """Build the interface to the controller."""
    #     # TODO take target trajectories out of the settings
    #     controller = bindings.ControllerInterface(self.settings)
    #     controller.reset(make_target_trajectories(reference_trajectory))
    #     return controller
