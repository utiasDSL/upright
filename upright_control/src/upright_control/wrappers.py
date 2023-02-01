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
        robot.forward_xu(x0)
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

        # only SQP supported now
        self.solver_method = bindings.ControllerSettings.solver_method_from_string(
            config["solver_method"].lower()
        )

        # MPC settings
        self.mpc.time_horizon = core.parsing.parse_number(config["mpc"]["time_horizon"])
        self.mpc.debug_print = config["mpc"]["debug_print"]
        self.mpc.cold_start = config["mpc"]["cold_start"]

        # Rollout settings
        self.rollout.abs_tol_ode = core.parsing.parse_number(
            config["rollout"]["abs_tol_ode"]
        )
        self.rollout.rel_tol_ode = core.parsing.parse_number(
            config["rollout"]["rel_tol_ode"]
        )
        self.rollout.timestep = core.parsing.parse_number(config["rollout"]["timestep"])
        self.rollout.max_num_steps_per_second = core.parsing.parse_number(
            config["rollout"]["max_num_steps_per_second"], dtype=int
        )
        self.rollout.check_numerical_stability = config["rollout"][
            "check_numerical_stability"
        ]

        # SQP settings
        self.sqp.dt = core.parsing.parse_number(config["sqp"]["dt"])
        self.sqp.sqp_iteration = config["sqp"]["sqp_iteration"]
        self.sqp.init_sqp_iteration = config["sqp"]["init_sqp_iteration"]
        self.sqp.delta_tol = core.parsing.parse_number(config["sqp"]["delta_tol"])
        self.sqp.cost_tol = core.parsing.parse_number(config["sqp"]["cost_tol"])
        self.sqp.use_feedback_policy = config["sqp"]["use_feedback_policy"]
        self.sqp.project_state_input_equality_constraints = config["sqp"][
            "project_state_input_equality_constraints"
        ]
        self.sqp.print_solver_status = config["sqp"]["print_solver_status"]
        self.sqp.print_solver_statistics = config["sqp"]["print_solver_statistics"]
        self.sqp.print_line_search = config["sqp"]["print_line_search"]

        # HPIPM (QP solver) settings
        self.sqp.hpipm.warm_start = config["sqp"]["hpipm"]["warm_start"]
        self.sqp.hpipm.iter_max = config["sqp"]["hpipm"]["iter_max"]
        self.sqp.hpipm.slacks.enabled = config["sqp"]["hpipm"]["slacks"]["enabled"]
        self.sqp.hpipm.slacks.upper_L2_penalty = config["sqp"]["hpipm"]["slacks"].get("upper_L2_penalty", 100)
        self.sqp.hpipm.slacks.lower_L2_penalty = config["sqp"]["hpipm"]["slacks"].get("lower_L2_penalty", 100)
        self.sqp.hpipm.slacks.upper_L1_penalty = config["sqp"]["hpipm"]["slacks"].get("upper_L1_penalty", 0)
        self.sqp.hpipm.slacks.lower_L1_penalty = config["sqp"]["hpipm"]["slacks"].get("lower_L1_penalty", 0)
        self.sqp.hpipm.slacks.upper_low_bound = config["sqp"]["hpipm"]["slacks"].get("upper_low_bound", 0)
        self.sqp.hpipm.slacks.lower_low_bound = config["sqp"]["hpipm"]["slacks"].get("lower_low_bound", 0)

        self.end_effector_link_name = config["robot"]["tool_link_name"]
        self.robot_base_type = bindings.robot_base_type_from_string(
            config["robot"]["base_type"]
        )

        # Estimation settings
        self.estimation.robot_init_variance = config["estimation"]["robot_init_variance"]
        self.estimation.robot_process_variance = config["estimation"]["robot_process_variance"]
        self.estimation.robot_measurement_variance = config["estimation"]["robot_measurement_variance"]

        # Tracking settings
        self.tracking.rate = config["tracking"]["rate"]
        self.tracking.min_policy_update_time = config["tracking"][
            "min_policy_update_time"
        ]

        self.tracking.kp = config["tracking"]["kp"]
        self.tracking.kv = config["tracking"]["kv"]
        self.tracking.ka = config["tracking"]["ka"]

        self.tracking.enforce_state_limits = config["tracking"]["enforce_state_limits"]
        self.tracking.enforce_input_limits = config["tracking"]["enforce_input_limits"]
        self.tracking.enforce_ee_position_limits = config["tracking"][
            "enforce_ee_position_limits"
        ]
        self.tracking.use_projectile = config["tracking"]["use_projectile"]

        self.tracking.state_violation_margin = config["tracking"][
            "state_violation_margin"
        ]
        self.tracking.input_violation_margin = config["tracking"][
            "input_violation_margin"
        ]
        self.tracking.ee_position_violation_margin = config["tracking"][
            "ee_position_violation_margin"
        ]

        # gravity
        self.gravity = config["gravity"]

        # whether we should recompile the auto-diff libraries even if the
        # library already exists
        self.recompile_libraries = config.get("recompile_libraries", True)

        # debug mode prints/publishes extra information, but may reduce
        # performance
        self.debug = config["debug"]

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

        # some joints in the URDF may be locked to constant values, for example
        # to only use part of the robot for an experiment
        if "locked_joints" in config["robot"]:
            for name, value in config["robot"]["locked_joints"].items():
                self.locked_joints[name] = core.parsing.parse_number(value)

        # (fixed) base pose for when a fixed base is used
        if "base_pose" in config["robot"]:
            base_pose = np.array(config["robot"]["base_pose"])
            assert base_pose.shape == (3,)
            self.base_pose = base_pose
        else:
            self.base_pose = np.zeros(3)

        # URDFs
        self.robot_urdf_path = core.parsing.parse_and_compile_urdf(
            config["robot"]["urdf"]
        )

        # directory for compiled auto-diff libraries
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
        self.balancing_settings.arrangement_name = config["balancing"]["arrangement"]
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

        if self.balancing_settings.enabled:
            self.dims.c = len(contacts)

            # dimension of each force variable: only one if we assume frictionless
            # contacts; three otherwise
            self.dims.nf = 1 if config["balancing"]["frictionless"] else 3
        else:
            # if we aren't balancing then we don't want to augment the problem
            # with extra variables
            self.dims.c = 0
            self.dims.nf = 0

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
            ias.align_with_fixed_vector = iac["align_with_fixed_vector"]
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
                    for mode_config in obs_config["modes"]:
                        mode = bindings.DynamicObstacleMode()
                        mode.time = mode_config["time"]
                        mode.position = np.array(mode_config["position"])
                        mode.velocity = np.array(mode_config["velocity"])
                        mode.acceleration = np.array(mode_config["acceleration"])
                        obs.modes.push_back(mode)
                    self.obstacle_settings.dynamic_obstacles.push_back(obs)

                    # the initial state for the obstacles has zero velocity and
                    # acceleration, so they are static. It is expected that the
                    # simulation or real sensors would update this when
                    # appropriate
                    x0_obs.append(np.concatenate((obs.modes[0].position, np.zeros(6))))

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
