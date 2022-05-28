import os

import numpy as np

import tray_balance_constraints as core
from tray_balance_ocs2 import bindings

import IPython


class TargetTrajectories(bindings.TargetTrajectories):
    """Wrapper around bound TargetTrajectories."""
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
            t = waypoint["time"]

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


class ControllerSettings(bindings.ControllerSettings):
    def __init__(self, config, x0=None, operating_trajectory=None):
        super().__init__()

        self.method = bindings.ControllerSettings.Method.DDP

        self.end_effector_link_name = config["robot"]["tool_link_name"]
        self.robot_base_type = bindings.robot_base_type_from_string(
            config["robot"]["base_type"]
        )

        # gravity
        self.gravity = config["gravity"]

        # dimensions
        self.dims.q = config["robot"]["dims"]["q"]
        self.dims.v = config["robot"]["dims"]["v"]
        self.dims.x = config["robot"]["dims"]["x"]
        self.dims.u = config["robot"]["dims"]["u"]

        # initial state can be passed in directly (for example to match exactly
        # a simulation) or parsed from the config
        if x0 is None:
            self.initial_state = core.parsing.parse_array(
                config["robot"]["x0"]
            )
        else:
            self.initial_state = x0
        assert self.initial_state.shape == (self.dims.x,)

        # weights for state, input, and EE pose
        self.input_weight = core.parsing.parse_diag_matrix_dict(
            config["weights"]["input"]
        )
        self.state_weight = core.parsing.parse_diag_matrix_dict(
            config["weights"]["state"]
        )
        self.end_effector_weight = core.parsing.parse_diag_matrix_dict(
            config["weights"]["end_effector"]
        )
        assert self.input_weight.shape == (self.dims.u, self.dims.u)
        assert self.state_weight.shape == (self.dims.x, self.dims.x)
        assert self.end_effector_weight.shape == (6, 6)

        # input limits
        self.input_limit_lower = core.parsing.parse_array(
            config["limits"]["input"]["lower"]
        )
        self.input_limit_upper = core.parsing.parse_array(
            config["limits"]["input"]["upper"]
        )
        assert self.input_limit_lower.shape == (self.dims.u,)
        assert self.input_limit_upper.shape == (self.dims.u,)

        # state limits
        self.state_limit_lower = core.parsing.parse_array(
            config["limits"]["state"]["lower"]
        )
        self.state_limit_upper = core.parsing.parse_array(
            config["limits"]["state"]["upper"]
        )
        assert self.state_limit_lower.shape == (self.dims.x,)
        assert self.state_limit_upper.shape == (self.dims.x,)

        # URDFs
        self.robot_urdf_path = core.parsing.parse_ros_path(
            config["robot"]["urdf"]
        )
        self.obstacle_urdf_path = core.parsing.parse_ros_path(
            config["static_obstacles"]["urdf"]
        )

        # task info file (Boost property tree format)
        self.ocs2_config_path = core.parsing.parse_ros_path(config["infofile"])
        self.lib_folder = "/tmp/ocs2"

        # operating points
        if operating_trajectory is not None:
            self.use_operating_points = True
            for i in range(len(operating_trajectory)):
                self.operating_times.push_back(operating_trajectory.ts[i])
                self.operating_states.push_back(operating_trajectory.xs[i, :])
                self.operating_inputs.push_back(operating_trajectory.us[i, :])

        # tray balance settings
        self.tray_balance_settings.enabled = config["balancing"]["enabled"]
        self.tray_balance_settings.constraint_type = bindings.ConstraintType.Soft
        self.tray_balance_settings.mu = core.parsing.parse_number(
            config["balancing"]["mu"]
        )
        self.tray_balance_settings.delta = core.parsing.parse_number(
            config["balancing"]["delta"]
        )

        ctrl_objects = core.parsing.parse_control_objects(config)
        self.tray_balance_settings.objects = ctrl_objects

        self.tray_balance_settings.constraints_enabled.normal = config[
            "balancing"
        ]["enable_normal_constraint"]
        self.tray_balance_settings.constraints_enabled.friction = config[
            "balancing"
        ]["enable_friction_constraint"]
        self.tray_balance_settings.constraints_enabled.zmp = config[
            "balancing"
        ]["enable_zmp_constraint"]

        # alternative inertial alignment objective
        # tries to keep tray/EE normal aligned with the negative acceleration
        # vector
        self.inertial_alignment_settings.enabled = config[
            "inertial_alignment"
        ]["enabled"]
        self.inertial_alignment_settings.use_angular_acceleration = config[
            "inertial_alignment"
        ]["use_angular_acceleration"]
        self.inertial_alignment_settings.weight = config["inertial_alignment"][
            "weight"
        ]
        self.inertial_alignment_settings.r_oe_e = ctrl_objects[
            -1
        ].body.com_ellipsoid.center()  # TODO could specify index in config

        # collision avoidance settings
        self.collision_avoidance_settings.enabled = False
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
            self.collision_avoidance_settings.collision_link_pairs.push_back(pair)
        # fmt: on
        self.collision_avoidance_settings.minimum_distance = 0
        self.collision_avoidance_settings.mu = 1e-2
        self.collision_avoidance_settings.delta = 1e-3

        # dynamic obstacle settings
        self.dynamic_obstacle_settings.enabled = False
        self.dynamic_obstacle_settings.obstacle_radius = 0.1
        self.dynamic_obstacle_settings.mu = 1e-2
        self.dynamic_obstacle_settings.delta = 1e-3

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
            self.dynamic_obstacle_settings.collision_spheres.push_back(sphere)

    def get_num_balance_constraints(self):
        if self.tray_balance_settings.bounded:
            return self.tray_balance_settings.objects.num_constraints()
        return self.tray_balance_settings.config.num_constraints()

    def get_num_collision_avoidance_constraints(self):
        if self.collision_avoidance_settings.enabled:
            return len(self.collision_avoidance_settings.collision_link_pairs)
        return 0

    def get_num_dynamic_obstacle_constraints(self):
        if self.dynamic_obstacle_settings.enabled:
            return len(self.dynamic_obstacle_settings.collision_spheres)
        return 0

    @property
    def objects(self):
        return self.tray_balance_settings.objects
