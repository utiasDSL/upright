import os
import rospkg
from tray_balance_ocs2.MobileManipulatorPyBindings import mpc_interface


def setup_ocs2_mpc_interface():
    rospack = rospkg.RosPack()
    task_info_path = os.path.join(
        rospack.get_path("tray_balance_ocs2"), "config", "mpc", "task.info"
    )
    return mpc_interface(task_info_path, "/tmp/ocs2")
