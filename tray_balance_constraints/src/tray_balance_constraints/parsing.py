"""Utilities for parsing general configuration dictionaries."""
from collections import deque
from pathlib import Path

import rospkg
import numpy as np

from tray_balance_constraints.bindings import (
    Ellipsoid,
    BoundedRigidBody,
    BoundedBalancedObject,
    PolygonSupportArea,
)
from tray_balance_constraints import math
from tray_balance_constraints.composition import compose_bounded_objects

import IPython


# Naming:
# - config = raw dict
# - config_wrapper = object somehow containing the raw config
# - arrangement = the particular set of objects in use


def parse_number(x):
    """Parse a number from the config.

    If the number can be converted to a float, then it is and is returned.
    Otherwise, check if it ends with "pi" and convert it to a float that is a
    multiple of pi.
    """
    try:
        # this also handles strings like '1e-2'
        return float(x)
    except ValueError:
        # TODO not robust
        return float(x[:-2]) * np.pi


def parse_array_element(x):
    try:
        return [float(x)]
    except ValueError:
        if x.endswith("pi"):
            return [float(x[:-2]) * np.pi]
        if "rep" in x:
            y, n = x.split("rep")
            return float(y) * np.ones(int(n))
        raise ValueError(f"Could not convert {x} to array element.")


def parse_array(a):
    """Parse a one-dimensional iterable into a numpy array."""
    subarrays = []
    for x in a:
        subarrays.append(parse_array_element(x))
    return np.concatenate(subarrays)


def parse_diag_matrix_dict(d):
    """Parse a dict containing a diagonal matrix.

    Key-values are:
      scale: float
      diag:  iterable

    Returns a diagonal numpy array.
    """
    scale = parse_number(d["scale"])
    diag = parse_array(d["diag"])
    base = np.diag(diag)
    return scale * base


def millis_to_secs(ms):
    """Convert milliseconds to seconds."""
    return 0.001 * ms


def parse_ros_path(d):
    """Resolve full path from a dict of containing ROS package and relative path."""
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path(d["package"])) / d["path"]
    return path.as_posix()


def parse_support_offset(d):
    """Parse the x-y offset of an object relative to its support plane.

    The dict d defining the offset can consist of up to four optional
    key-values: x and y define a Cartesian offset, and r and θ define a radial
    offset. If both are included, then the Cartesian offset is applied first
    and the radial offset is added to it.

    Returns: the numpy array [x, y] defining the offset.
    """
    x = d["x"] if "x" in d else 0
    y = d["y"] if "y" in d else 0
    if "r" in d and "θ" in d:
        r = d["r"]
        θ = parse_number(d["θ"])
        x += r * np.cos(θ)
        y += r * np.sin(θ)
    return np.array([x, y])


class BalancedObjectConfigWrapper:
    """Wrapper around the config dict for a balanced object."""

    def __init__(self, config, parent_name):
        self.d = config
        self.parent_name = parent_name
        self.children = []
        self.position = None

    @property
    def height(self):
        return self.d["height"]

    @property
    def offset(self):
        # TODO this needs to be dealt with
        if "offset" in self.d:
            return np.array(self.d["offset"])
        return np.zeros(2)

    def support_area(self):
        config = self.d["support_area"]
        shape = config["shape"]
        if shape == "eq_tri":
            side_length = config["side_length"]
            support_area = PolygonSupportArea.equilateral_triangle(side_length)
            r_tau = math.equilateral_triangle_r_tau(side_length)
        elif shape == "rect":
            lx = config["lx"]
            ly = config["lx"]
            support_area = PolygonSupportArea.axis_aligned_rectangle(lx, ly)
            r_tau = math.rectangle_r_tau(lx, ly)
        elif shape == "circle":
            radius = config["radius"]
            support_area = PolygonSupportArea.circle(radius)
            r_tau = math.circle_r_tau(radius)
        else:
            raise ValueError(f"Unsupported support area shape: {shape}")
        return support_area, r_tau

    def bounded_balanced_object(self):
        """Generate a BoundedBalancedObject for this object."""
        # parse the bounded rigid body
        mass_min = self.d["mass"]["min"]
        mass_max = self.d["mass"]["max"]

        com_center = self.position
        com_half_lengths = np.array(self.d["com"]["half_lengths"])
        com_ellipsoid = Ellipsoid(com_center, com_half_lengths, np.eye(3))

        # parse the radii of gyration
        # this can be specified to be based on the exact inertia matrix for a
        # particular shape
        if "use_exact" in self.d["radii_of_gyration"]:
            shape = self.d["radii_of_gyration"]["use_exact"]["shape"]
            if shape == "cylinder":
                radius = self.d["radii_of_gyration"]["use_exact"]["radius"]
                height = self.d["radii_of_gyration"]["use_exact"]["height"]
                inertia = math.cylinder_inertia_matrix(
                    mass=1, radius=radius, height=height
                )
            elif shape == "cuboid":
                side_lengths = np.array(
                    self.d["radii_of_gyration"]["use_exact"]["side_lengths"]
                )
                inertia = math.cuboid_inertia_matrix(mass=1, side_lengths=side_lengths)
            else:
                raise ValueError(f"Unrecognized shape {shape}.")
            # no need to divide out mass, since we used mass=1 above
            radii_of_gyration = np.sqrt(np.diag(inertia))
            radii_of_gyration_min = radii_of_gyration
            radii_of_gyration_max = radii_of_gyration
        else:
            radii_of_gyration_min = np.array(self.d["radii_of_gyration"]["min"])
            radii_of_gyration_max = np.array(self.d["radii_of_gyration"]["max"])

        body = BoundedRigidBody(
            mass_min=mass_min,
            mass_max=mass_max,
            radii_of_gyration_min=radii_of_gyration_min,
            radii_of_gyration_max=radii_of_gyration_max,
            com_ellipsoid=com_ellipsoid,
        )

        support_area, r_tau = self.support_area()

        com_height = self.d["com"]["height"]
        mu_min = self.d["mu_min"]

        return BoundedBalancedObject(
            body,
            com_height=com_height,
            support_area_min=support_area,
            r_tau_min=r_tau,
            mu_min=mu_min,
        )


def parse_control_objects(ctrl_config):
    arrangement_name = ctrl_config["balancing"]["arrangement"]
    arrangement = ctrl_config["arrangements"][arrangement_name]
    object_configs = ctrl_config["objects"]
    ee = object_configs["ee"]

    wrappers = {}
    for conf in arrangement:
        obj_type = conf["type"]
        parent_name = conf["parent"] if "parent" in conf else None
        object_config = object_configs[obj_type]
        wrapper = BalancedObjectConfigWrapper(object_config, parent_name)

        # compute position of the object
        if wrapper.parent_name is not None:
            parent = wrappers[wrapper.parent_name]
            dz = 0.5 * parent.height + 0.5 * wrapper.height
            wrapper.position = parent.position + [0, 0, dz]
        else:
            dz = 0.5 * ee["height"] + 0.5 * wrapper.height
            wrapper.position = np.array([0, 0, dz])

        # add offset in the x-y (support) plane
        wrapper.position[:2] += wrapper.offset

        obj_name = conf["name"]
        if obj_name in wrappers:
            raise ValueError(f"Multiple control objects named {obj_name}.")
        wrappers[obj_name] = wrapper

    # find the direct children of each object
    for name, wrapper in wrappers.items():
        if wrapper.parent_name is not None:
            wrappers[wrapper.parent_name].children.append(name)

    # convert wrappers to BoundedBalancedObjects as required by the controller
    # and compose them as needed
    composites = []
    for name, wrapper in wrappers.items():
        # all descendants compose the new object
        descendants = []
        queue = deque([wrapper])
        while len(queue) > 0:
            desc_wrapper = queue.popleft()
            descendants.append(desc_wrapper.bounded_balanced_object())
            for child_name in desc_wrapper.children:
                queue.append(wrappers[child_name])

        # descendants have already been converted to C++ objects
        composites.append(compose_bounded_objects(descendants))

    return composites
