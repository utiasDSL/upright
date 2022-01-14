import os

import rospkg

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


LIBRARY_PATH = "/tmp/ocs2"


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


def get_task_info_path():
    rospack = rospkg.RosPack()
    return os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )


def setup_ocs2_mpc_interface(settings):
    return ocs2.mpc_interface(get_task_info_path(), LIBRARY_PATH, settings)
