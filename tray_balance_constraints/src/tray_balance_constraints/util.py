import time

import numpy as np
import liegroups


def quaternion_to_matrix(Q, normalize=True):
    """Convert quaternion to rotation matrix."""
    if normalize:
        if np.allclose(Q, 0):
            Q = np.array([0, 0, 0, 1])
        else:
            Q = Q / np.linalg.norm(Q)
    try:
        return liegroups.SO3.from_quaternion(Q, ordering="xyzw").as_matrix()
    except ValueError as e:
        IPython.embed()


def transform_point(r_ba_a, Q_ab, r_cb_b):
    """Transform point r_cb_b to r_ca_a.

    This is equivalent to r_ca_a = T_ab @ r_cb_b, where T_ab is the homogeneous
    transformation matrix from A to B (and I've abused notation for homogeneous
    vs. non-homogeneous points).
    """
    C_ab = quaternion_to_matrix(Q_ab)
    return r_ba_a + C_ab @ r_cb_b


def rotate_point(Q, r):
    """Rotate a point r using quaternion Q."""
    return transform_point(np.zeros(3), Q, r)


def support_area_distance(ctrl_object, Q_we):
    """Compute distance outside of SA at current EE orientation Q_we."""
    C_we = quaternion_to_matrix(Q_we)

    # position of CoM relative to center of SA
    r_com_o = np.array([0, 0, ctrl_object.com_height])
    r_com_w = C_we @ r_com_o

    # solve for the intersection point of r_com_w with the SA (in the
    # SA frame) knowing that:
    # * intersection point in world frame has same (x, y) as CoM
    # * intersection point in object frame has z = 0
    A = np.eye(3)
    A[:2, :] = C_we[:2, :]
    b = np.zeros(3)
    b[:2] = r_com_w[:2]
    c = np.linalg.solve(A, b)

    d = ctrl_object.support_area_min.distance_outside(c[:2])
    return d


def secs_to_ns(secs):
    """Convert seconds to nanoseconds."""
    return 1e9 * secs


def secs_from_ns(ns):
    """Convert nanoseconds to seconds."""
    return 1e-9 * ns

# see the (more sophisticated) ROS implementation:
# <https://github.com/ros/ros_comm/blob/noetic-devel/clients/rospy/src/rospy/timer.py>
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
