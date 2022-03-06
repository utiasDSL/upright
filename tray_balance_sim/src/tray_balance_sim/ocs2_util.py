import os

import numpy as np
import rospkg

from tray_balance_sim import geometry
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2

import IPython


LIBRARY_PATH = "/tmp/ocs2"


class TaskSettingsWrapper:
    def __init__(self, composites, x0):
        settings = ocs2.TaskSettings()

        settings.method = ocs2.TaskSettings.Method.DDP
        settings.initial_state = x0

        # tray balance settings
        settings.tray_balance_settings.enabled = True
        settings.tray_balance_settings.robust = False
        settings.tray_balance_settings.constraint_type = ocs2.ConstraintType.Soft
        settings.tray_balance_settings.mu = 1e-2
        settings.tray_balance_settings.delta = 5e-4
        settings.tray_balance_settings.config.objects = composites

        # robust settings
        robust_params = ocs2.RobustParameterSet()
        robust_params.min_support_dist = 0.04
        robust_params.min_mu = 0.2
        robust_params.min_r_tau = geometry.circle_r_tau(robust_params.min_support_dist)
        settings.tray_balance_settings.robust_params = robust_params

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
            ocs2.CollisionSphere(
                name="elbow_collision_link",
                parent_frame_name="ur10_arm_forearm_link",
                offset=np.zeros(3),
                radius=0.15,
            ),
            ocs2.CollisionSphere(
                name="forearm_collision_sphere_link1",
                parent_frame_name="ur10_arm_forearm_link",
                offset=np.array([0, 0, 0.2]),
                radius=0.15,
            ),
            ocs2.CollisionSphere(
                name="forearm_collision_sphere_link2",
                parent_frame_name="ur10_arm_forearm_link",
                offset=np.array([0, 0, 0.4]),
                radius=0.15,
            ),
            ocs2.CollisionSphere(
                name="wrist_collision_link",
                parent_frame_name="ur10_arm_wrist_3_link",
                offset=np.zeros(3),
                radius=0.15,
            ),
            # ocs2.CollisionSphere(
            #     name="wrist_collision_link2",
            #     parent_frame_name="ur10_arm_wrist_3_link",
            #     offset=np.array([0, 0.25, 0]),
            #     radius=0.03,
            # ),
        ]:
            settings.dynamic_obstacle_settings.collision_spheres.push_back(sphere)

        self.settings = settings

    def get_num_balance_constraints(self):
        if self.settings.tray_balance_settings.robust:
            return len(self.settings.tray_balance_settings.robust_params.balls) * 3
        return self.settings.tray_balance_settings.config.num_constraints()

    def get_num_collision_avoidance_constraints(self):
        if self.settings.collision_avoidance_settings.enabled:
            return len(self.settings.collision_avoidance_settings.collision_link_pairs)
        return 0

    def get_num_dynamic_obstacle_constraints(self):
        if self.settings.dynamic_obstacle_settings.enabled:
            return len(self.settings.dynamic_obstacle_settings.collision_spheres)
        return 0


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


def make_target_trajectories(target_times, target_states, target_inputs):
    assert len(target_times) == len(target_states)
    assert len(target_times) == len(target_inputs)

    target_times_ocs2 = ocs2.scalar_array()
    for target_time in target_times:
        target_times_ocs2.push_back(target_time)

    target_states_ocs2 = ocs2.vector_array()
    for target_state in target_states:
        target_states_ocs2.push_back(target_state)

    target_inputs_ocs2 = ocs2.vector_array()
    for target_input in target_inputs:
        target_inputs_ocs2.push_back(target_input)

    return ocs2.TargetTrajectories(
        target_times_ocs2, target_states_ocs2, target_inputs_ocs2
    )


def setup_ocs2_mpc_interface(settings, target_times, target_states, target_inputs):
    mpc = ocs2.mpc_interface(get_task_info_path(), LIBRARY_PATH, settings)
    target_trajectories = make_target_trajectories(
        target_times, target_states, target_inputs
    )
    mpc.reset(target_trajectories)
    return mpc
