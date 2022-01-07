import os
import rospkg
import property_tree as ptree
from tray_balance_ocs2.MobileManipulatorPyBindings import mpc_interface


def str2bool(s):
    s = s.lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    raise ValueError(
        "String must have value 'true' or 'false' to be converted to bool."
    )


class TaskProperties:
    def __init__(self, path):
        properties = ptree.info.load(path)

        self.method = properties["model_settings"]["method"].value
        assert self.method == "SQP" or self.method == "DDP"

        self.tray_balance_enabled = str2bool(
            properties["trayBalanceConstraints"]["enabled"].value
        )
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


def load_ocs2_task_properties():
    return TaskProperties(get_task_info_path())


def setup_ocs2_mpc_interface():
    return mpc_interface(get_task_info_path(), "/tmp/ocs2")
