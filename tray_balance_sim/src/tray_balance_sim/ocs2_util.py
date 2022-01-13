import os
from enum import Enum

import rospkg
import property_tree as ptree

from tray_balance_ocs2.MobileManipulatorPythonInterface import mpc_interface


def str2bool(s):
    s = s.lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    raise ValueError(
        "String must have value 'true' or 'false' to be converted to bool."
    )


class ConstraintType(Enum):
    SOFT = 0
    HARD = 1


class ConfigType(Enum):
    STACKED = 0
    FLAT = 1


class TrayBalanceSettings:
    def __init__(self, properties):
        key = "trayBalanceConstraints"
        self.enabled = str2bool(properties[key]["enabled"].value)

        s = properties[key]["constraint_type"]
        self.constraint_type = (
            ConstraintType.SOFT if s == "soft" else ConstraintType.HARD
        )

        s = properties[key]["config_type"]
        self.config_type = (
            ConfigType.STACKED if s == "stacked" else ConfigType.FLAT
        )

        self.num_objects = int(properties[key]["num_objects"])

    def obj_names(self):
        config_prefix = "stacked_" if self.config_type == ConfigType.STACKED else "flat_"
        obj_root = config_prefix + "cylinder"
        obj_names = [obj_root + str(i + 1) for i in range(self.num_objects)]
        return ["tray"] + obj_names


class TaskSettings:
    def __init__(self, path):
        properties = ptree.info.load(path)

        # self.tray_balance_settings = TrayBalanceSettings(properties)
        self.dynamic_obstacle_enabled = str2bool(
            properties["dynamicObstacleAvoidance"]["enabled"].value
        )
        self.collision_avoidance_enabled = str2bool(
            properties["collisionAvoidance"]["enabled"].value
        )

        self.num_dynamic_obstacle_pairs = len(
            properties["dynamicObstacleAvoidance"]["collision_link_names"]
        )
        self.num_collision_pairs = len(
            properties["collisionAvoidance"]["collisionLinkPairs"]
        )


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


def load_ocs2_task_settings():
    return TaskSettings(get_task_info_path())


def setup_ocs2_mpc_interface(settings):
    return mpc_interface(get_task_info_path(), "/tmp/ocs2", settings)
