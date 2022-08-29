import os

import numpy as np

import upright_core as core
from upright_control import bindings
from upright_control.robot import PinocchioRobot
from upright_control.trajectory import StateInputTrajectory

import IPython


class TargetTrajectories(bindings.TargetTrajectories):
    """Wrapper around TargetTrajectories binding."""

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

    @classmethod
    def from_config_file(cls, config_path, x0):
        """Load the trajectory directly from a config file.

        This is convenient for loading from C++, for example in the MRT node.
        """
        config = core.parsing.load_config(config_path)
        ctrl_config = config["controller"]
        robot = PinocchioRobot(config=ctrl_config["robot"])
        robot.forward(x0)
        r_ew_w, Q_we = robot.link_pose()
        u0 = np.zeros(robot.dims.ou)
        return cls.from_config(ctrl_config, r_ew_w, Q_we, u0)

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
    """Wrapper around ControllerSettings binding."""
    def __init__(self, config, x0=None, operating_trajectory=None):
        super().__init__()

        # can be either DDP or SQP
        self.solver_method = bindings.ControllerSettings.solver_method_from_string(
            config["solver_method"].lower()
        )

        self.end_effector_link_name = config["robot"]["tool_link_name"]
        self.robot_base_type = bindings.robot_base_type_from_string(
            config["robot"]["base_type"]
        )

        # gravity
        self.gravity = config["gravity"]

        # dimensions
        # note that dims.f (number of contact forces) is set below
        self.dims.q = config["robot"]["dims"]["q"]
        self.dims.v = config["robot"]["dims"]["v"]
        self.dims.x = config["robot"]["dims"]["x"]
        self.dims.u = config["robot"]["dims"]["u"]

        # initial state can be passed in directly (for example to match exactly
        # a simulation) or parsed from the config
        if x0 is None:
            self.initial_state = core.parsing.parse_array(config["robot"]["x0"])
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
        self.limit_constraint_type = bindings.constraint_type_from_string(
            config["limits"]["constraint_type"]
        )
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

        # tracking gain
        self.Kp = core.parsing.parse_diag_matrix_dict(config["tracking"]["Kp"])
        assert self.Kp.shape == (self.dims.q, self.dims.q)

        # rate for tracking controller
        self.rate = core.parsing.parse_number(config["tracking"]["rate"])

        # URDFs
        self.robot_urdf_path = core.parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        self.obstacle_urdf_path = core.parsing.parse_and_compile_urdf(
            config["static_obstacles"]["urdf"]
        )

        # task info file (Boost property tree format)
        self.ocs2_config_path = core.parsing.parse_ros_path(config["infofile"])
        self.lib_folder = "/tmp/ocs2"

        # operating points
        if config["operating_points"]["enabled"]:
            operating_path = core.parsing.parse_ros_path(config["operating_points"])
            operating_trajectory = StateInputTrajectory.load(operating_path)
            for i in range(len(operating_trajectory)):
                self.operating_times.push_back(operating_trajectory.ts[i])
                self.operating_states.push_back(operating_trajectory.xs[i, :])
                self.operating_inputs.push_back(operating_trajectory.us[i, :])

        # tray balance settings
        self.balancing_settings.enabled = config["balancing"]["enabled"]
        self.balancing_settings.use_force_constraints = config["balancing"]["use_force_constraints"]
        self.balancing_settings.constraint_type = bindings.constraint_type_from_string(
            config["balancing"]["constraint_type"]
        )
        self.balancing_settings.mu = core.parsing.parse_number(
            config["balancing"]["mu"]
        )
        self.balancing_settings.delta = core.parsing.parse_number(
            config["balancing"]["delta"]
        )

        self.balancing_settings.force_weight = config["balancing"]["force_weight"]
        ctrl_objects, contacts = core.parsing.parse_control_objects(config)
        self.balancing_settings.objects = ctrl_objects
        self.balancing_settings.contacts = contacts
        self.dims.f = len(contacts)

        # inputs are augmented with the contact forces
        # if self.balancing_settings.use_force_constraints:
        #     self.dims.u += 3 * self.dims.f

        self.balancing_settings.constraints_enabled.normal = config["balancing"][
            "enable_normal_constraint"
        ]
        self.balancing_settings.constraints_enabled.friction = config["balancing"][
            "enable_friction_constraint"
        ]
        self.balancing_settings.constraints_enabled.zmp = config["balancing"][
            "enable_zmp_constraint"
        ]

        # alternative inertial alignment objective
        # tries to keep tray/EE normal aligned with the negative acceleration
        # vector
        self.inertial_alignment_settings.enabled = config["inertial_alignment"][
            "enabled"
        ]
        if self.inertial_alignment_settings.enabled:
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
        self.static_obstacle_settings.enabled = config["static_obstacles"]["enabled"]
        self.static_obstacle_settings.constraint_type = (
            bindings.constraint_type_from_string(
                config["static_obstacles"]["constraint_type"]
            )
        )
        if config["static_obstacles"]["collision_pairs"] is not None:
            for pair in config["static_obstacles"]["collision_pairs"]:
                self.static_obstacle_settings.collision_link_pairs.push_back(
                    tuple(pair)
                )
        self.static_obstacle_settings.minimum_distance = config["static_obstacles"][
            "minimum_distance"
        ]
        self.static_obstacle_settings.mu = core.parsing.parse_number(
            config["static_obstacles"]["mu"]
        )
        self.static_obstacle_settings.delta = core.parsing.parse_number(
            config["static_obstacles"]["delta"]
        )

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

    @classmethod
    def from_config_file(cls, config_path):
        """Load the settings directly from a config file.

        This is convenient for loading from C++, for example in the MRT node.
        """
        config = core.parsing.load_config(config_path)
        ctrl_config = config["controller"]
        return cls(ctrl_config)

    def get_num_balance_constraints(self):
        if self.balancing_settings.bounded:
            return self.balancing_settings.objects.num_constraints()
        return self.balancing_settings.config.num_constraints()

    def get_num_collision_avoidance_constraints(self):
        if self.static_obstacle_settings.enabled:
            return len(self.static_obstacle_settings.collision_link_pairs)
        return 0

    def get_num_dynamic_obstacle_constraints(self):
        if self.dynamic_obstacle_settings.enabled:
            return len(self.dynamic_obstacle_settings.collision_spheres)
        return 0

    @property
    def objects(self):
        return self.balancing_settings.objects
