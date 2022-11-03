import os

import numpy as np

import upright_core as core
from upright_control import bindings
from upright_control.robot import build_robot_interfaces
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
            x = np.concatenate((r_ew_w_d, Q_we_d))

            ts.append(t)
            us.append(np.copy(u))
            xs.append(x)
        return cls(ts, xs, us)

    @classmethod
    def from_config_file(cls, config_path, x0):
        """Load the trajectory directly from a config file.

        This is convenient for loading from C++, for example in the MRT node.
        """
        config = core.parsing.load_config(config_path)["controller"]
        settings = ControllerSettings(config, x0=x0)

        # update the state of the robot to match the actual (supplied) state;
        # we don't care about the dynamic obstacle state here because we're
        # only after the EE pose
        robot, _ = build_robot_interfaces(settings)
        robot.forward(x0)
        r_ew_w, Q_we = robot.link_pose()
        u0 = np.zeros(settings.dims.robot.u)
        return cls.from_config(config, r_ew_w, Q_we, u0)

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

        # robot dimensions
        self.dims.robot.q = config["robot"]["dims"]["q"]
        self.dims.robot.v = config["robot"]["dims"]["v"]
        self.dims.robot.x = config["robot"]["dims"]["x"]
        self.dims.robot.u = config["robot"]["dims"]["u"]

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
        assert self.input_weight.shape == (self.dims.robot.u, self.dims.robot.u)
        assert self.state_weight.shape == (self.dims.robot.x, self.dims.robot.x)
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
        assert self.input_limit_lower.shape == (self.dims.robot.u,)
        assert self.input_limit_upper.shape == (self.dims.robot.u,)

        # state limits
        self.state_limit_lower = core.parsing.parse_array(
            config["limits"]["state"]["lower"]
        )
        self.state_limit_upper = core.parsing.parse_array(
            config["limits"]["state"]["upper"]
        )
        assert self.state_limit_lower.shape == (self.dims.robot.x,)
        assert self.state_limit_upper.shape == (self.dims.robot.x,)

        # end effector position box constraint
        self.end_effector_box_constraint_enabled = config[
            "end_effector_box_constraint"
        ]["enabled"]
        self.xyz_lower = core.parsing.parse_array(
            config["end_effector_box_constraint"]["xyz_lower"]
        )
        self.xyz_upper = core.parsing.parse_array(
            config["end_effector_box_constraint"]["xyz_upper"]
        )
        assert self.xyz_lower.shape == (3,)
        assert self.xyz_upper.shape == (3,)

        # tracking gain
        self.Kp = core.parsing.parse_diag_matrix_dict(config["tracking"]["Kp"])
        assert self.Kp.shape == (self.dims.robot.q, self.dims.robot.q)

        # rate for tracking controller
        self.rate = core.parsing.parse_number(config["tracking"]["rate"])

        # some joints in the URDF may be locked to constant values, for example
        # to only use part of the robot for an experiment
        if "locked_joints" in config["robot"]:
            for name, value in config["robot"]["locked_joints"].items():
                self.locked_joints[name] = core.parsing.parse_number(value)

        # URDFs
        self.robot_urdf_path = core.parsing.parse_and_compile_urdf(
            config["robot"]["urdf"]
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
        self.balancing_settings.use_force_constraints = config["balancing"][
            "use_force_constraints"
        ]
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
        self.dims.c = len(contacts)

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
        ias = self.inertial_alignment_settings
        iac = config["inertial_alignment"]
        ias.cost_enabled = iac["cost_enabled"]
        ias.constraint_enabled = iac["constraint_enabled"]
        if ias.cost_enabled or ias.constraint_enabled:
            ias.use_angular_acceleration = iac["use_angular_acceleration"]
            ias.cost_weight = iac["cost_weight"]
            normal = np.array(iac["contact_plane_normal"])
            normal = normal / np.linalg.norm(normal)
            ias.contact_plane_normal = normal
            ias.contact_plane_span = core.math.plane_span(normal)
            ias.com = np.array(iac["com"])
            ias.alpha = iac["alpha"]

        # obstacle settings
        x0_obs = []
        self.obstacle_settings.enabled = config["obstacles"]["enabled"]
        if self.obstacle_settings.enabled:
            self.obstacle_settings.constraint_type = (
                bindings.constraint_type_from_string(
                    config["obstacles"]["constraint_type"]
                )
            )
            if config["obstacles"]["collision_pairs"] is not None:
                for pair in config["obstacles"]["collision_pairs"]:
                    self.obstacle_settings.collision_link_pairs.push_back(tuple(pair))
            self.obstacle_settings.minimum_distance = config["obstacles"][
                "minimum_distance"
            ]
            self.obstacle_settings.mu = core.parsing.parse_number(
                config["obstacles"]["mu"]
            )
            self.obstacle_settings.delta = core.parsing.parse_number(
                config["obstacles"]["delta"]
            )

            if "urdf" in config["obstacles"]:
                self.obstacle_settings.obstacle_urdf_path = (
                    core.parsing.parse_and_compile_urdf(config["obstacles"]["urdf"])
                )

            if "dynamic" in config["obstacles"]:
                self.dims.o = len(config["obstacles"]["dynamic"])
                for obs_config in config["obstacles"]["dynamic"]:
                    obs = bindings.DynamicObstacle()
                    obs.name = obs_config["name"]
                    obs.radius = obs_config["radius"]
                    obs.position = np.array(obs_config["position"])
                    obs.velocity = np.array(obs_config["velocity"])
                    obs.acceleration = np.array(obs_config["acceleration"])
                    self.obstacle_settings.dynamic_obstacles.push_back(obs)

                    # the initial state for the obstacles has zero velocity and
                    # acceleration, so they are static. It is expected that the
                    # simulation or real sensors would update this when
                    # appropriate
                    x0_obs.append(np.concatenate((obs.position, np.zeros(6))))

                x0_obs = np.concatenate(x0_obs)

        # initial state can be passed in directly (for example to match exactly
        # a simulation) or parsed from the config
        # we do this at the end to properly account for dynamic obstacles added
        # to the state
        if x0 is None:
            x0_robot = core.parsing.parse_array(config["robot"]["x0"])
            assert x0_robot.shape == (self.dims.robot.x,)
            self.initial_state = np.concatenate((x0_robot, x0_obs))
        else:
            self.initial_state = x0
        assert self.initial_state.shape == (self.dims.x(),)

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
