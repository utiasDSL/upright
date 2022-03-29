import os

import numpy as np
import rospkg

from tray_balance_ocs2 import bindings
from tray_balance_constraints import parsing

LIBRARY_PATH = "/tmp/ocs2"


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


class ControllerSettingsWrapper:
    def __init__(self, ctrl_config):
        settings = bindings.ControllerSettings()

        settings.method = bindings.ControllerSettings.Method.DDP

        # TODO hard-coding
        settings.input_weight = 0.1 * np.eye(9)
        settings.state_weight = np.diag(np.concatenate((np.zeros(9), 0.2 * np.ones(9))))
        settings.end_effector_weight = np.diag([1, 1, 1, 0, 0, 1])

        # input limits
        settings.input_limit_lower = np.array([-2, -2, -2, -4, -4, -4, -4, -4, -4])
        settings.input_limit_upper = -settings.input_limit_lower

        # state limits
        q_limits_lower = np.concatenate((-10 * np.ones(3), -2 * np.pi * np.ones(6)))
        q_limits_upper = -q_limits_lower
        v_limits_lower = np.concatenate((-1 * np.ones(3), -2 * np.ones(6)))
        v_limits_upper = -v_limits_lower
        settings.state_limit_lower = np.concatenate((q_limits_lower, v_limits_lower))
        settings.state_limit_upper = np.concatenate((q_limits_upper, v_limits_upper))

        # URDFs
        settings.robot_urdf_path = parsing.parse_urdf_path(ctrl_config["urdf"]["robot"])
        settings.obstacle_urdf_path = parsing.parse_urdf_path(
            ctrl_config["urdf"]["obstacles"]
        )

        settings.ocs2_config_path = get_task_info_path()

        # tray balance settings
        settings.tray_balance_settings.enabled = True
        settings.tray_balance_settings.bounded = True
        settings.tray_balance_settings.constraint_type = bindings.ConstraintType.Soft
        settings.tray_balance_settings.mu = 1e-2
        settings.tray_balance_settings.delta = 5e-4

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
        settings.dynamic_obstacle_settings.mu = 1e-2  # NOTE now matching others
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


def make_target_trajectories(target_times, target_states, target_inputs):
    assert len(target_times) == len(target_states)
    assert len(target_times) == len(target_inputs)

    target_times_ocs2 = bindings.scalar_array()
    for target_time in target_times:
        target_times_ocs2.push_back(target_time)

    target_states_ocs2 = bindings.vector_array()
    for target_state in target_states:
        target_states_ocs2.push_back(target_state)

    target_inputs_ocs2 = bindings.vector_array()
    for target_input in target_inputs:
        target_inputs_ocs2.push_back(target_input)

    return bindings.TargetTrajectories(
        target_times_ocs2, target_states_ocs2, target_inputs_ocs2
    )


def setup_ctrl_interface(settings, target_times, target_states, target_inputs):
    # TODO get rid of dependency on task info file here
    controller = bindings.ControllerInterface(
        get_task_info_path(), LIBRARY_PATH, settings
    )
    target_trajectories = make_target_trajectories(
        target_times, target_states, target_inputs
    )
    controller.reset(target_trajectories)
    return controller
