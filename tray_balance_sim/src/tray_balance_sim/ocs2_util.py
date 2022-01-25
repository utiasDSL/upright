import os

import rospkg
import numpy as np

from tray_balance_sim import geometry
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


LIBRARY_PATH = "/tmp/ocs2"


def get_task_settings(composites):
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
    settings.tray_balance_settings.robust = False
    settings.tray_balance_settings.constraint_type = ocs2.ConstraintType.Soft

    config = ocs2.TrayBalanceConfiguration()
    config.arrangement = ocs2.TrayBalanceConfiguration.Arrangement.Stacked
    config.num = 3
    config.objects = composites
    settings.tray_balance_settings.config = config

    # robust settings
    robust_params = ocs2.RobustParameterSet()
    robust_params.min_support_dist = 0.05
    robust_params.min_mu = 0.5
    robust_params.min_r_tau = geometry.circle_r_tau(robust_params.min_support_dist)

    if config.arrangement == ocs2.TrayBalanceConfiguration.Arrangement.Stacked:
        ball1 = ocs2.Ball([0, 0, 0.1], 0.12)
        ball2 = ocs2.Ball([0, 0, 0.3], 0.12)

        robust_params.max_radius = 0.5 * (
            np.linalg.norm(ball2.center - ball2.center) + ball1.radius + ball2.radius
        )
        robust_params.balls = [ball1, ball2]
    else:
        ball = ocs2.Ball([0, 0, 0.02 + 0.02 + 0.075], 0.1)
        robust_params.max_radius = ball.radius
        robust_params.balls = [ball]

    settings.tray_balance_settings.robust_params = robust_params

    return settings


def get_obj_names_from_settings(settings):
    """Parse object names for simulation from the task settings."""
    config = settings.tray_balance_settings.config

    if config.arrangement == ocs2.TrayBalanceConfiguration.Arrangement.Stacked:
        config_prefix = "stacked"
    else:
        config_prefix = "flat"

    obj_root = config_prefix + "_cylinder"
    obj_names = [obj_root + str(i + 1) for i in range(config.num)]

    return ["tray"] + obj_names


def get_num_balance_constraints(settings):
    num_tray_con = 5
    num_cylinder_con = 6
    # num_cylinder_con = 3
    if settings.tray_balance_settings.robust:
        return len(settings.tray_balance_settings.robust_params.balls) * 3
    return num_tray_con + settings.tray_balance_settings.config.num * num_cylinder_con


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


def setup_ocs2_mpc_interface(settings):
    return ocs2.mpc_interface(get_task_info_path(), LIBRARY_PATH, settings)
