import os

import numpy as np
import rospkg

import tray_balance_constraints as core
from tray_balance_ocs2 import bindings

import IPython

LIBRARY_PATH = "/tmp/ocs2"


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


class ReferenceTrajectory:
    """Reference trajectory of a sequence of waypoints.

    If there is more than one waypoint, then the desired pose is interpolated
    between them.
    """
    def __init__(self, times, states, inputs):
        assert len(times) == len(states)
        assert len(times) == len(inputs)

        self.times = times
        self.states = states
        self.inputs = inputs

    @classmethod
    def from_config(cls, ctrl_config, r_ew_w, Q_we, u):
        """Parse the reference trajectory from the config dict."""
        times = []
        inputs = []
        states = []
        for waypoint in ctrl_config["waypoints"]:
            secs = core.parsing.millis_to_secs(waypoint["millis"])

            r_ew_w_d = r_ew_w + waypoint["position"]
            Q_we_d = core.math.quat_multiply(Q_we, waypoint["orientation"])
            r_obs = np.zeros(3)  #r_ew_w + np.array([0, -10, 0])
            state = np.concatenate((r_ew_w_d, Q_we_d, r_obs))

            times.append(secs)
            inputs.append(np.copy(u))
            states.append(state)
        return cls(times, states, inputs)

    def target_trajectories(self):
        """Construct the target trajectories object for OCS2."""
        target_times_ocs2 = bindings.scalar_array()
        for target_time in self.times:
            target_times_ocs2.push_back(target_time)

        target_states_ocs2 = bindings.vector_array()
        for target_state in self.states:
            target_states_ocs2.push_back(target_state)

        target_inputs_ocs2 = bindings.vector_array()
        for target_input in self.inputs:
            target_inputs_ocs2.push_back(target_input)

        return bindings.TargetTrajectories(
            target_times_ocs2, target_states_ocs2, target_inputs_ocs2
        )

    @staticmethod
    def pose(state):
        """Extract EE position and orientation from a generic reference state."""
        # this is useful because the state also contains info about a dynamic
        # obstacle
        r = state[:3]
        Q = state[3:7]
        return r, Q


class ControllerConfigWrapper:
    def __init__(self, ctrl_config, x0=None):
        self.config = ctrl_config
        settings = bindings.ControllerSettings()

        settings.method = bindings.ControllerSettings.Method.DDP

        # dimensions
        self.q_dim = ctrl_config["robot"]["dims"]["q"]
        self.v_dim = ctrl_config["robot"]["dims"]["v"]
        self.a_dim = ctrl_config["robot"]["dims"]["a"]

        self.x_dim = self.q_dim + self.v_dim
        self.u_dim = self.a_dim

        # initial state can be passed in directly (for example to match exactly
        # a simulation) or parsed from the config
        if x0 is None:
            settings.initial_state = core.parsing.parse_array(
                ctrl_config["robot"]["x0"]
            )
        else:
            settings.initial_state = x0
        assert settings.initial_state.shape == (self.x_dim,)

        # weights for state, input, and EE pose
        settings.input_weight = core.parsing.parse_diag_matrix_dict(
            ctrl_config["weights"]["input"]
        )
        settings.state_weight = core.parsing.parse_diag_matrix_dict(
            ctrl_config["weights"]["state"]
        )
        settings.end_effector_weight = core.parsing.parse_diag_matrix_dict(
            ctrl_config["weights"]["end_effector"]
        )
        assert settings.input_weight.shape == (self.u_dim, self.u_dim)
        assert settings.state_weight.shape == (self.x_dim, self.x_dim)
        assert settings.end_effector_weight.shape == (6, 6)

        # input limits
        settings.input_limit_lower = core.parsing.parse_array(
            ctrl_config["limits"]["input"]["lower"]
        )
        settings.input_limit_upper = core.parsing.parse_array(
            ctrl_config["limits"]["input"]["upper"]
        )
        assert settings.input_limit_lower.shape == (self.u_dim,)
        assert settings.input_limit_upper.shape == (self.u_dim,)

        # state limits
        settings.state_limit_lower = core.parsing.parse_array(
            ctrl_config["limits"]["state"]["lower"]
        )
        settings.state_limit_upper = core.parsing.parse_array(
            ctrl_config["limits"]["state"]["upper"]
        )
        assert settings.state_limit_lower.shape == (self.x_dim,)
        assert settings.state_limit_upper.shape == (self.x_dim,)

        # URDFs
        settings.robot_urdf_path = core.parsing.parse_urdf_path(
            ctrl_config["robot"]["urdf"]
        )
        settings.obstacle_urdf_path = core.parsing.parse_urdf_path(
            ctrl_config["static_obstacles"]["urdf"]
        )

        # task info file (Boost property tree format)
        settings.ocs2_config_path = get_task_info_path()

        # tray balance settings
        settings.tray_balance_settings.enabled = ctrl_config["balancing"]["enabled"]
        settings.tray_balance_settings.bounded = ctrl_config["balancing"][
            "use_bounded_constraints"
        ]
        settings.tray_balance_settings.constraint_type = bindings.ConstraintType.Soft
        settings.tray_balance_settings.mu = core.parsing.parse_number(
            ctrl_config["balancing"]["mu"]
        )
        settings.tray_balance_settings.delta = core.parsing.parse_number(
            ctrl_config["balancing"]["delta"]
        )

        ctrl_objects = core.parsing.parse_control_objects(ctrl_config)
        settings.tray_balance_settings.bounded_config.objects = ctrl_objects

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

        self.settings = settings

    def get_num_balance_constraints(self):
        if self.settings.tray_balance_settings.bounded:
            return self.settings.tray_balance_settings.bounded_config.num_constraints()
        return self.settings.tray_balance_settings.config.num_constraints()

    def get_num_collision_avoidance_constraints(self):
        if self.settings.collision_avoidance_settings.enabled:
            return len(self.settings.collision_avoidance_settings.collision_link_pairs)
        return 0

    def get_num_dynamic_obstacle_constraints(self):
        if self.settings.dynamic_obstacle_settings.enabled:
            return len(self.settings.dynamic_obstacle_settings.collision_spheres)
        return 0

    # TODO ideally I would compute r_ew_w and Q_we based on a Pinocchio model
    # of the robot included in this repo
    def reference_trajectory(self, r_ew_w, Q_we):
        return ReferenceTrajectory.from_config(
            self.config, r_ew_w, Q_we, np.zeros(self.u_dim)
        )

    def controller(self, r_ew_w, Q_we):
        """Build the interface to the controller."""
        target_trajectories = self.reference_trajectory(
            r_ew_w, Q_we
        ).target_trajectories()

        # TODO get rid of dependency on task info file here
        controller = bindings.ControllerInterface(
            get_task_info_path(), LIBRARY_PATH, self.settings
        )
        controller.reset(target_trajectories)
        return controller
