"""Utilities for parsing general configuration dictionaries."""
from pathlib import Path

import rospkg
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from xacrodoc import XacroDoc

import mobile_manipulation_central as mm
from upright_core.bindings import RigidBody, ContactPoint
from upright_core import math, polyhedron


# This is from <https://github.com/Maples7/dict-recursive-update/blob/07204cdab891ac4123b19fe3fa148c3dd1c93992/dict_recursive_update/__init__.py>
def recursive_dict_update(default, custom):
    """Return a dict merged from default and custom"""
    if not isinstance(default, dict) or not isinstance(custom, dict):
        raise TypeError("Params of recursive_update should be dicts")

    for key in custom:
        if isinstance(custom[key], dict) and isinstance(default.get(key), dict):
            default[key] = recursive_dict_update(default[key], custom[key])
        else:
            default[key] = custom[key]

    return default


def load_config(path, depth=0, max_depth=5):
    """Load configuration file located at `path`.

    `depth` and `max_depth` arguments are provided to protect against
    unexpectedly deep or infinite recursion through included files.
    """
    if depth > max_depth:
        raise Exception(f"Maximum inclusion depth {max_depth} exceeded.")

    with open(path) as f:
        d = yaml.safe_load(f)

    # get the includes while also removing them from the dict
    includes = d.pop("include", [])

    # construct a dict of everything included
    includes_dict = {}
    for include in includes:
        path = parse_ros_path(include)
        include_dict = load_config(path, depth=depth + 1, max_depth=max_depth)

        # nest the include under `key` if specified
        if "key" in include:
            include_dict = {include["key"]: include_dict}

        # update the includes dict and reassign
        includes_dict = recursive_dict_update(includes_dict, include_dict)

    # now add in the info from this file
    d = recursive_dict_update(includes_dict, d)
    return d


def parse_number(x, dtype=float):
    """Parse a number from the config.

    If the number is a string that ends with pi, then it is parsed as type
    `dtype` that is a multiple of pi. Otherwise, it is convered to type `dtype`.
    """
    if type(x) == str and x.endswith("pi"):
        return dtype(x[:-2]) * np.pi
    return dtype(x)


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


def parse_ros_path(d, as_string=True):
    """Resolve full path from a dict of containing ROS package and relative path."""
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path(d["package"])) / d["path"]
    if as_string:
        path = path.as_posix()
    return path


def parse_and_compile_urdf(d, max_runs=10, compare_existing=True, quiet=False):
    """Parse and compile a URDF from a xacro'd URDF file."""
    includes = d["includes"]
    subargs = d.get("args", None)

    doc = XacroDoc.from_includes(includes, subargs=subargs, max_runs=max_runs)

    # write the final document to a file for later consumption
    output_path = parse_ros_path(d, as_string=False)
    doc.to_urdf_file(output_path, compare_existing=compare_existing, verbose=not quiet)

    return output_path.as_posix()


def parse_support_offset(d):
    """Parse the x-y offset of an object relative to its support plane.

    The dict d defining the offset can consist of up to four optional
    key-values: x and y define a Cartesian offset, and r and θ define a radial
    offset. If both are included, then the Cartesian offset is applied first
    and the radial offset is added to it.

    Returns: the numpy array [x, y] defining the offset.
    """
    x = d.get("x", 0)
    y = d.get("y", 0)
    if "r" in d and "θ" in d:
        r = d["r"]
        θ = parse_number(d["θ"])
        x += r * np.cos(θ)
        y += r * np.sin(θ)
    elif "r" in d or "θ" in d:
        raise ValueError("Radius and angle must *both* be specified in support offset.")
    return np.array([x, y])


class _BalancedObjectWrapper:
    def __init__(self, body, box, parent_name, fixture):
        self.body = body
        self.box = box
        self.parent_name = parent_name
        self.fixture = fixture


def _parse_objects_with_contacts(wrappers, contact_conf, tol=1e-7):
    """
    wrappers is the dict of name: object wrappers
    neighbours is a list of pairs of names specifying objects in contact
    """
    contact_points = []
    for contact in contact_conf:
        name1 = contact["first"]
        name2 = contact["second"]

        mu_margin = contact.get("mu_margin", 0)
        mu = contact["mu"] - mu_margin
        inset = contact.get("support_area_inset", 0)

        box1 = wrappers[name1].box
        box2 = wrappers[name2].box
        points, normal = polyhedron.axis_aligned_contact(box1, box2, tol=1e-7)
        assert points is not None, "No contact points found."

        body1 = wrappers[name1].body
        body2 = wrappers[name2].body

        for i in range(points.shape[0]):
            contact_point = ContactPoint()
            contact_point.object1_name = name1
            contact_point.object2_name = name2
            contact_point.mu = mu
            contact_point.normal = normal

            span = math.plane_span(normal)

            r1 = points[i, :]

            # it doesn't make sense to inset w.r.t. fixtures (EE or otherwise),
            # because we don't have to worry about their dynamics
            if wrappers[name1].fixture:
                r1_inset = r1
            else:
                # project point into tangent plane and inset the tangent part
                # wrt to the box's center
                # NOTE this does not make sense for non-planar contacts
                r1_t = span @ (r1 - box1.position)
                r1_t_inset = math.inset_vertex(r1_t, inset)

                # unproject the inset back into 3D space
                r1_inset = r1 + (r1_t_inset - r1_t) @ span

            r2 = points[i, :]
            r2_t = span @ (r2 - box2.position)
            r2_t_inset = math.inset_vertex(r2_t, inset)
            r2_inset = r2 + (r2_t_inset - r2_t) @ span

            contact_point.span = span
            contact_point.r_co_o1 = r1_inset
            contact_point.r_co_o2 = r2_inset

            contact_points.append(contact_point)

    return contact_points


def parse_local_half_extents(obj_type_conf):
    shape = obj_type_conf["shape"].lower()
    if shape == "cuboid" or shape == "wedge":
        return 0.5 * np.array(obj_type_conf["side_lengths"])
    elif shape == "cylinder":
        r = obj_type_conf["radius"]
        h = obj_type_conf["height"]
        w = np.sqrt(2) * r
        return 0.5 * np.array([w, w, h])
    raise ValueError(f"Unsupported shape type: {shape}")


def parse_box(obj_type_conf, position=None, rotation=None):
    if rotation is None:
        rotation = np.eye(3)

    shape = obj_type_conf["shape"].lower()
    local_half_extents = parse_local_half_extents(obj_type_conf)
    if shape == "wedge":
        box = polyhedron.ConvexPolyhedron.wedge(local_half_extents)
    elif shape == "cuboid":
        box = polyhedron.ConvexPolyhedron.box(local_half_extents)
    elif shape == "cylinder":
        # for the cylinder, we rotate by 45 deg about z so that contacts occur
        # aligned with x-y axes
        rotation = rotation @ math.rotz(np.pi / 4)
        box = polyhedron.ConvexPolyhedron.box(local_half_extents)

    return box.transform(translation=position, rotation=rotation)


def parse_mu_dict(contact_conf, apply_margin):
    """Parse a dictionary of coefficients of friction from the contact configuration.

    Returns a nested dict with object names as keys and mu as the value.
    """
    mus = {}
    for contact in contact_conf:
        parent_name = contact["first"]
        child_name = contact["second"]
        mu = contact["mu"]
        if apply_margin:
            mu -= contact.get("mu_margin", 0)
        if parent_name in mus:
            mus[parent_name][child_name] = mu
        else:
            mus[parent_name] = {child_name: mu}
    return mus


def parse_inset_dict(contact_conf):
    insets = {}
    for contact in contact_conf:
        parent_name = contact["first"]
        child_name = contact["second"]
        inset = contact.get("support_area_inset", 0)
        if parent_name in insets:
            insets[parent_name][child_name] = inset
        else:
            insets[parent_name] = {child_name: inset}
    return insets


def parse_inertia(mass, obj_type_conf):
    shape = obj_type_conf["shape"].lower()
    if shape == "cylinder":
        inertia = math.cylinder_inertia_matrix(
            mass=mass, radius=obj_type_conf["radius"], height=obj_type_conf["height"]
        )
    elif shape == "cuboid":
        inertia = math.cuboid_inertia_matrix(
            mass=mass, side_lengths=obj_type_conf["side_lengths"]
        )
    elif shape == "wedge":
        D, C = math.wedge_inertia_matrix(mass, obj_type_conf["side_lengths"])
        inertia = C @ D @ C.T
    else:
        raise ValueError(f"Unsupported shape type {shape}.")

    return inertia


def _parse_rigid_body_and_box(obj_type_conf, base_position, quat):
    # base_position is directly beneath the reference position for this shape
    mass = obj_type_conf["mass"]
    C = math.quat_to_rot(quat)

    local_com_offset = np.array(obj_type_conf["com_offset"], dtype=np.float64)
    if obj_type_conf["shape"].lower() == "wedge":
        # wedge reference position is the centroid of the box of which it is a
        # half, but that is of course not the CoM: add required offset now
        hx, hy, hz = 0.5 * np.array(obj_type_conf["side_lengths"])
        local_com_offset += np.array([-hx, 0, -hz]) / 3
    com_offset = C @ local_com_offset

    # Inertia can be manually specified; otherwise defaults to assuming uniform
    # density for the given shape. Either the full matrix can be specified or
    # just the diagonal.
    if "inertia" in obj_type_conf:
        local_inertia = np.array(obj_type_conf["inertia"])
        if local_inertia.shape == (3,):
            local_inertia = np.diag(local_inertia)
        elif local_inertia.shape != (3, 3):
            raise ValueError(
                f"Object inertia matrix has wrong shape: {local_inertia.shape}"
            )
    elif "inertia_diag" in obj_type_conf:
        print("Using 'inertia_diag' is deprecated: use 'inertia' instead.")
        local_inertia = np.diag(obj_type_conf["inertia_diag"])
    else:
        local_inertia = parse_inertia(mass, obj_type_conf)
    inertia = C @ local_inertia @ C.T

    z = np.array([0, 0, 1])
    local_box = parse_box(obj_type_conf, rotation=C)
    dz = local_box.distance_from_centroid_to_boundary(-z)

    # position of the shape: this is not necessarily the centroid or CoM
    # this is in the EE frame (not the shape's local frame)
    reference_position = base_position + [0, 0, dz]
    com_position = reference_position + com_offset

    # now we recompute the box with the correct reference position
    box = parse_box(obj_type_conf, reference_position, C)
    body = RigidBody(mass, inertia, com_position)
    return body, box


def parse_control_objects(ctrl_conf):
    """Parse the control objects and contact points from the configuration."""
    arrangement_name = ctrl_conf["balancing"]["arrangement"]
    arrangement = ctrl_conf["arrangements"][arrangement_name]
    obj_type_confs = ctrl_conf["objects"]
    contact_conf = arrangement["contacts"]

    # NOTE: for backward compatibility with previous config format version
    for obj_type_conf in obj_type_confs.values():
        shape = obj_type_conf["shape"]
        if isinstance(shape, dict):
            obj_type_conf["shape"] = shape["type"]
            del shape["type"]
            obj_type_conf.update(shape)

    # placeholder end effector object
    ee_conf = obj_type_confs["ee"]
    ee_box = parse_box(ee_conf, np.array(ee_conf["position"]))
    ee_body = RigidBody(1, np.eye(3), ee_box.position)
    wrappers = {
        "ee": _BalancedObjectWrapper(
            body=ee_body, box=ee_box, parent_name=None, fixture=True
        )
    }

    for obj_instance_conf in arrangement["objects"]:
        obj_name = obj_instance_conf["name"]
        if obj_name in wrappers:
            raise ValueError(f"Multiple control objects named {obj_name}.")

        obj_type = obj_instance_conf["type"]
        obj_type_conf = obj_type_confs[obj_type]

        # object orientation (as a quaternion)
        quat = obj_instance_conf.get("orientation", np.array([0, 0, 0, 1]))
        quat = quat / np.linalg.norm(quat)

        # get parent information and the coefficient of friction between the
        # two objects
        parent_name = obj_instance_conf["parent"]
        parent_box = wrappers[parent_name].box

        # position, accounting for parent box offset and height
        position = parent_box.position.copy()
        if "offset" in obj_instance_conf:
            position[:2] += parse_support_offset(obj_instance_conf["offset"])
        position[2] += parent_box.distance_from_centroid_to_boundary(
            np.array([0, 0, 1])
        )

        fixture = "fixture" in obj_instance_conf and obj_instance_conf["fixture"]

        body, box = _parse_rigid_body_and_box(obj_type_conf, position, quat)
        wrappers[obj_name] = _BalancedObjectWrapper(body, box, parent_name, fixture)

    # fixtured objects includes the EE: we don't want them here, because they
    # need not be constrained
    contact_points = _parse_objects_with_contacts(wrappers, contact_conf)
    bodies = {name: w.body for name, w in wrappers.items() if not w.fixture}
    return bodies, contact_points
