import time
import shutil
from pathlib import Path

import numpy as np
import upright_core as core


def copy_task_info_file(config, dest_dir):
    infofile_src_path = Path(core.parsing.parse_ros_path(config["controller"]["infofile"]))
    infofile_dest_path = dest_dir / infofile_src_path.name
    shutil.copy(str(infofile_src_path), str(infofile_dest_path))


def support_area_distance(ctrl_object, Q_we):
    """Compute distance outside of SA at current EE orientation Q_we."""
    C_we = core.math.quat_to_rot(Q_we)
    normal = ctrl_object.support_area.normal()

    # position of CoM relative to center of SA
    r_com_e = ctrl_object.com_height * normal
    r_com_w = C_we @ r_com_e

    # solve for the intersection point of r_com_w with the SA (in the
    # SA frame) knowing that:
    # * intersection point in world frame has same (x, y) as CoM
    # * intersection point in EE frame dotted with normal = 0
    A = np.eye(3)
    A[:2, :] = C_we[:2, :]
    A[2, :] = normal
    b = np.zeros(3)
    b[:2] = r_com_w[:2]
    c = np.linalg.solve(A, b)

    d = ctrl_object.support_area.distance(c)
    return d


# see the (more sophisticated) ROS implementation:
# <https://github.com/ros/ros_comm/blob/noetic-devel/clients/rospy/src/rospy/timer.py>
# TODO deprecated
class Rate:
    def __init__(self, timestep_ns, quiet=False):
        """Initialize a Rate based on a timestep in nanoseconds."""
        self.timestep_ns = int(timestep_ns)
        self.timestep_secs = secs_from_ns(timestep_ns)
        self.quiet = quiet

        self._last_time_ns = time.time_ns()

    @classmethod
    def from_timestep_secs(cls, timestep_secs, quiet=False):
        """Construct a Rate based on a timestep in seconds."""
        return cls(secs_to_ns(timestep_secs), quiet=quiet)

    @classmethod
    def from_hz(cls, hz, quiet=False):
        """Construct a Rate based on a frequency in Hertz (1 / seconds)."""
        return cls.from_timestep_secs(1. / hz, quiet=quiet)

    def sleep(self):
        elapsed_ns = time.time_ns() - self._last_time_ns
        duration_ns = self.timestep_ns - elapsed_ns
        if duration_ns > 0:
            time.sleep(secs_from_ns(duration_ns))
        else:
            if not self.quiet:
                print(f"loop is too slow by {-duration_ns} ns")
        self._last_time_ns = time.time_ns()
        # self.last_time += self.secs
