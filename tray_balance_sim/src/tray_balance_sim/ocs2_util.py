import os

import rospkg

from tray_balance_sim import geometry
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


LIBRARY_PATH = "/tmp/ocs2"


class TaskSettingsWrapper:
    def __init__(self, composites):
        settings = ocs2.TaskSettings()

        settings.method = ocs2.TaskSettings.Method.DDP

        # collision avoidance settings
        settings.collision_avoidance_settings.enabled = False
        settings.collision_avoidance_settings.collision_link_pairs = [
            ("forearm_collision_link_0", "balanced_object_collision_link_0")
        ]
        settings.collision_avoidance_settings.minimum_distance = 0

        # dynamic obstacle settings
        settings.dynamic_obstacle_settings.enabled = False
        settings.dynamic_obstacle_settings.collision_link_names = [
            "thing_tool",
            "elbow_collision_link",
            "forearm_collision_sphere_link1",
            "forearm_collision_sphere_link2",
            "wrist_collision_link",
        ]
        for r in [0.25, 0.15, 0.15, 0.15, 0.15]:
            settings.dynamic_obstacle_settings.collision_sphere_radii.push_back(r)
        settings.dynamic_obstacle_settings.obstacle_radius = 0.1

        # tray balance settings
        settings.tray_balance_settings.enabled = True
        settings.tray_balance_settings.robust = True
        settings.tray_balance_settings.constraint_type = ocs2.ConstraintType.Soft

        config = ocs2.TrayBalanceConfiguration()
        # config.arrangement = ocs2.TrayBalanceConfiguration.Arrangement.Stacked
        config.objects = composites
        settings.tray_balance_settings.config = config

        # robust settings
        robust_params = ocs2.RobustParameterSet()
        robust_params.min_support_dist = 0.05
        robust_params.min_mu = 0.5
        robust_params.min_r_tau = geometry.circle_r_tau(robust_params.min_support_dist)
        settings.tray_balance_settings.robust_params = robust_params

        self.settings = settings

    def get_num_balance_constraints(self):
        if self.settings.tray_balance_settings.robust:
            return len(self.settings.tray_balance_settings.robust_params.balls) * 3
        return self.settings.tray_balance_settings.config.num_constraints()


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


def setup_ocs2_mpc_interface(settings):
    return ocs2.mpc_interface(get_task_info_path(), LIBRARY_PATH, settings)
