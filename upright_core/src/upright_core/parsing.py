"""Utilities for parsing general configuration dictionaries."""
from collections import deque
from pathlib import Path
import tempfile

import rospkg
import numpy as np
import yaml
import xacro
from scipy.spatial import ConvexHull

from upright_core.bindings import (
    RigidBody,
    BalancedObject,
    PolygonSupportArea,
    ContactPoint,
)
from upright_core import math, geometry, right_triangle

import IPython


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
        include_dict = load_config(path, depth=depth + 1)

        # nest the include under `key` if specified
        if "key" in include:
            include_dict = {include["key"]: include_dict}

        # update the includes dict and reassign
        includes_dict = recursive_dict_update(includes_dict, include_dict)

    # now add in the info from this file
    d = recursive_dict_update(includes_dict, d)
    return d


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


def xacro_include(path):
    return f"""
    <xacro:include filename="{path}" />
    """


def parse_and_compile_urdf(d, max_runs=10):
    """Parse and compile a URDF from a xacro'd URDF file."""
    output_path = parse_ros_path(d)

    s = """
    <?xml version="1.0" ?>
    <robot name="robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
    """.strip()
    for incl in d["includes"]:
        s += xacro_include(incl)
    s += "</robot>"

    doc = xacro.parse(s)
    s1 = doc.toxml()

    # xacro args
    mappings = d["args"] if "args" in d else {}

    # keep processing until a fixed point is reached
    run = 1
    while run < max_runs:
        xacro.process_doc(doc, mappings=mappings)
        s2 = doc.toxml()
        if s1 == s2:
            break
        s1 = s2
        run += 1

    if run == max_runs:
        raise ValueError("URDF file did not converge.")

    # write the final document to a file for later consumption
    with open(output_path, "w") as f:
        f.write(doc.toprettyxml(indent="  "))

    return output_path


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


class _BalancedObjectWrapper:
    def __init__(self, body, box, parent_name):
        self.body = body
        self.box = box
        self.parent_name = parent_name


def _parse_objects_with_contacts(wrappers, contact_conf, inset=0, mu_margin=0):
    """
    wrappers is the dict of name: object wrappers
    neighbours is a list of pairs of names specifying objects in contact
    """
    contact_points = []
    for contact in contact_conf:
        name1 = contact["first"]
        name2 = contact["second"]
        mu = contact["mu"] - mu_margin

        box1 = wrappers[name1].box
        box2 = wrappers[name2].box
        points, normal = geometry.box_box_axis_aligned_contact(box1, box2)

        # DEBUG
        if points is None or points.shape[0] < 4:
            IPython.embed()

        body1 = wrappers[name1].body
        body2 = wrappers[name2].body

        for i in range(points.shape[0]):
            contact_point = ContactPoint()
            contact_point.object1_name = name1
            contact_point.object2_name = name2
            contact_point.mu = mu
            contact_point.normal = normal

            span = math.plane_span(normal)

            r1 = points[i, :] - body1.com

            # project point into tangent plane and inset the tangent part
            r1_t = span @ r1
            r1_t_inset = math.inset_vertex(r1_t, inset)

            # unproject the inset back into 3D space
            r1_inset = r1 + (r1_t_inset - r1_t) @ span

            r2 = points[i, :] - body2.com
            r2_t = span @ r2
            r2_t_inset = math.inset_vertex(r2_t, inset)
            r2_inset = r2 + (r2_t_inset - r2_t) @ span

            contact_point.r_co_o1 = r1_inset
            contact_point.r_co_o2 = r2_inset

            contact_points.append(contact_point)

    # Wrap in balanced objects: not fundamentally necessary for the
    # constraints, but useful for (1) homogeneity of the settings API and (2)
    # allows some additional analysis on e.g. distance from support area
    mus = parse_mu_dict(contact_conf, margin=mu_margin)
    balanced_objects = {}
    for name, wrapper in wrappers.items():
        if name == "ee":
            continue
        body = wrapper.body
        box = wrapper.box
        com_offset = body.com - box.position

        # height of the CoM is in the local object frame
        z = np.array([0, 0, 1])
        com_height = box.distance_from_centroid_to_boundary(-box.rotation @ z, offset=com_offset)

        # find the convex hull of support points to get support area
        support_points = []
        for contact_point in contact_points:
            # we are assuming that the object being supported is always the
            # second one listed in the contact point
            if contact_point.object2_name == name:
                p = contact_point.r_co_o2
            else:
                continue

            # the points that have height equal to the CoM height are in the
            # support plane
            normal = contact_point.normal
            if np.abs(normal @ p - com_height) < 1e-8:
                span = math.plane_span(normal)
                support_points.append(span @ p)

        # take the convex hull of the support points
        support_points = np.array(support_points)
        hull = ConvexHull(support_points)
        support_points = support_points[hull.vertices, :]

        support_area = PolygonSupportArea(list(support_points))

        mu = mus[wrapper.parent_name][name]
        r_tau = 0.1  # placeholder/dummy value

        # contact force constraints don't use r_tau or mu, so we don't care
        # about them
        balanced_objects[name] = BalancedObject(
            body, com_height, support_area, r_tau, mu
        )

    return balanced_objects, contact_points


def _parse_composite_objects(wrappers, contact_conf, inset=0, mu_margin=0):
    # coefficients of friction between contacting objects
    mus = parse_mu_dict(contact_conf, margin=mu_margin)

    # build the balanced objects
    descendants = {}
    for name, wrapper in wrappers.items():
        if name == "ee":
            continue
        body = wrapper.body
        parent_name = wrapper.parent_name
        mu = mus[parent_name][name]

        box = wrapper.box
        com_offset = body.com - box.position
        com_height = box.distance_from_centroid_to_boundary(-box.rotation @ z, offset=com_offset)

        parent_box = wrappers[parent_name].box
        support_area, r_tau = compute_support_area(box, parent_box, com_offset, inset)

        obj = BalancedObject(body, com_height, support_area, r_tau, mu)

        descendants[name] = [(name, obj)]
        if parent_name != "ee":
            descendants[parent_name].append((name, obj))

    # compose objects together
    composites = {}
    for children in descendants.values():
        # composite_name = name + "_composite"
        composite_name = "_".join([name for name, _ in children])
        composite = BalancedObject.compose([obj for _, obj in children])
        composites[composite_name] = composite
    return composites


def parse_local_half_extents(shape_config):
    type_ = shape_config["type"].lower()
    if type_ == "cuboid" or type_ == "right_triangular_prism":
        return 0.5 * np.array(shape_config["side_lengths"])
    elif type_ == "cylinder":
        r = shape_config["radius"]
        h = shape_config["height"]
        w = np.sqrt(2) * r
        return 0.5 * np.array([w, w, h])
    raise ValueError(f"Unsupported shape type: {type_}")


def parse_box(shape_config, position=None, rotation=None):
    if rotation is None:
        rotation = np.eye(3)

    type_ = shape_config["type"].lower()
    local_half_extents = parse_local_half_extents(shape_config)
    if type_ == "right_triangular_prism":
        vertices, normals = right_triangle.right_triangular_prism_vertices_normals(
            local_half_extents
        )
        box = geometry.ConvexPolyhedron(vertices, normals, position, rotation)
    elif type_ == "cuboid":
        box = geometry.Box3d(local_half_extents, position, rotation)
    elif type_ == "cylinder":
        # for the cylinder, we rotate by 45 deg about z so that contacts occur
        # aligned with x-y axes
        rotation = rotation @ math.rotz(np.pi / 4)
        box = geometry.Box3d(local_half_extents, position, rotation)

    return box


def compute_support_area(box, parent_box, com_offset, inset, tol=1e-6):
    """Compute the support area.

    Currently only a rectangular support area is supported, due to the need to
    compute a value for r_tau.

    Parameters:
        box: the collision box of the current object
        parent_box: the collision box of the parent
        inset: positive scalar denoting how much to reduce the support area

    Returns:
        support area, r_tau
    """
    # TODO this approach is not too general for cylinders, where we may want
    # contacts with more than just their boxes
    points, _ = geometry.box_box_axis_aligned_contact(box, parent_box)
    assert inset >= 0, "Support area inset must be non-negative."

    # only support areas in the x-y plane are supported, check all z
    # coordinates are the same:
    z = points[0, 2]
    assert np.all(np.abs(points[:, 2] - z) < tol)

    # r_tau assumes rectangular for now, check:
    n = points.shape[0]
    assert n == 4, f"Support area has {n} points, not 4!"
    lengths = []
    for i in range(-1, 3):
        lengths.append(np.linalg.norm(points[i + 1, :] - points[i, :]))
    assert (
        abs(lengths[0] - lengths[2]) < tol and abs(lengths[1] - lengths[3]) < tol
    ), f"Support area is not rectangular!"

    # translate points to be relative to the box's centroid
    # TODO check this is right
    com_position = box.position + com_offset
    local_points = points - com_position
    local_points_xy = local_points[:, :2]

    # apply inset
    for i in range(local_points_xy.shape[0]):
        local_points_xy[i, :] = math.inset_vertex(local_points_xy[i, :], inset)
        # local_points_xy[i, :] = math.inset_vertex_abs(local_points_xy[i, :], inset)

    support_area = PolygonSupportArea(list(local_points_xy))
    r_tau = math.rectangle_r_tau(lengths[0], lengths[1])
    return support_area, r_tau


def parse_mu_dict(contact_conf, margin=0):
    """Parse a dictionary of coefficients of friction from the contact configuration.

    Returns a nested dict with object names as keys and mu as the value.
    """
    mus = {}
    for contact in contact_conf:
        parent_name = contact["first"]
        child_name = contact["second"]
        mu = contact["mu"] - margin
        if parent_name in mus:
            mus[parent_name][child_name] = mu
        else:
            mus[parent_name] = {child_name: mu}
    return mus


def parse_inertia(mass, shape_config, com_offset):
    type_ = shape_config["type"].lower()
    if type_ == "cylinder":
        inertia = math.cylinder_inertia_matrix(
            mass=mass, radius=shape_config["radius"], height=shape_config["height"]
        )
    elif type_ == "cuboid":
        inertia = math.cuboid_inertia_matrix(
            mass=mass, side_lengths=shape_config["side_lengths"]
        )
    elif type_ == "right_triangular_prism":
        D, C = right_triangle.right_triangular_prism_inertia_normalized(
            0.5 * np.array(shape_config["side_lengths"])
        )
        inertia = C @ D @ C.T
    else:
        raise ValueError(f"Unsupported shape type {type_}.")

    # adjust inertia for an offset CoM using parallel axis theorem
    R = math.skew3(com_offset)
    inertia = inertia - mass * R @ R

    return inertia


def _parse_rigid_body_and_box(obj_type_conf, base_position, quat):
    # base_position is directly beneath the reference position for this shape
    mass = obj_type_conf["mass"]
    shape_config = obj_type_conf["shape"]
    C = math.quat_to_rot(quat)

    local_com_offset = np.array(obj_type_conf["com_offset"], dtype=np.float64)
    if shape_config["type"].lower() == "right_triangular_prism":
        hx, hy, hz = 0.5 * np.array(shape_config["side_lengths"])
        local_com_offset += np.array([-hx, 0, -hz]) / 3
    com_offset = C @ local_com_offset

    local_inertia = parse_inertia(mass, shape_config, local_com_offset)
    inertia = C @ local_inertia @ C.T

    z = np.array([0, 0, 1])
    local_box = parse_box(shape_config, rotation=C)
    dz = local_box.distance_from_centroid_to_boundary(-z)

    # position of the shape: this is not necessarily the centroid or CoM
    # this is in the EE frame (not the shape's local frame)
    reference_position = base_position + [0, 0, dz]
    com_position = reference_position + com_offset

    # TODO we still don't support ZMP on sloped surface (need a normal
    # parameter for this)

    # now we recompute the box with the correct reference position
    box = parse_box(shape_config, reference_position, C)
    body = RigidBody(mass, inertia, com_position)
    return body, box


def parse_control_objects(ctrl_conf):
    """Parse the control objects and contact points from the configuration."""
    arrangement_name = ctrl_conf["balancing"]["arrangement"]
    arrangement = ctrl_conf["arrangements"][arrangement_name]
    obj_type_confs = ctrl_conf["objects"]
    contact_conf = arrangement["contacts"]

    sa_inset = arrangement.get("support_area_inset", 0)
    mu_margin = arrangement.get("mu_margin", 0)

    # placeholder end effector object
    ee_conf = obj_type_confs["ee"]
    ee_box = parse_box(ee_conf["shape"], np.array(ee_conf["position"]))
    ee_body = RigidBody(1, np.eye(3), ee_box.position)
    wrappers = {"ee": _BalancedObjectWrapper(ee_body, ee_box, None)}

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
        position[2] += parent_box.distance_from_centroid_to_boundary(np.array([0, 0, 1]))

        body, box = _parse_rigid_body_and_box(obj_type_conf, position, quat)
        wrappers[obj_name] = _BalancedObjectWrapper(body, box, parent_name)

    if ctrl_conf["balancing"]["use_force_constraints"]:
        return _parse_objects_with_contacts(
            wrappers, contact_conf, inset=sa_inset, mu_margin=mu_margin
        )
    else:
        composites = _parse_composite_objects(
            wrappers, contact_conf, inset=sa_inset, mu_margin=mu_margin
        )
        return composites, []
