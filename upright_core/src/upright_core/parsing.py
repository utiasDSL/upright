"""Utilities for parsing general configuration dictionaries."""
from collections import deque
from pathlib import Path
import tempfile

import rospkg
import numpy as np
import yaml
import xacro

from upright_core.bindings import (
    Ellipsoid,
    BoundedRigidBody,
    BoundedBalancedObject,
    PolygonSupportArea,
    ContactPoint,
)
from upright_core import math, geometry
from upright_core.composition import compose_bounded_objects

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


# # TODO could possibly convert this to inheriting from BoundedBalancedObject?
# class BalancedObjectConfigWrapper:
#     """Wrapper around the config dict for a balanced object."""
#
#     def __init__(self, config, parent_name, orientation=None):
#         self.d = config
#         self.parent_name = parent_name
#         self.children = []
#         self.position = None
#
#         if orientation is None:
#             orientation = np.array([0, 0, 0, 1])
#         self.C0 = math.quat_to_rot(orientation)
#
#         # TODO we could just immediately generate the half extents then rotate
#         # them to generate the support area?
#         # we also need to rotate the inertia, as well as the CoM offset
#         self.mass = config["mass"]
#
#         # generate half extents (not rotated)
#         # NOTE: we are assuming a 90 deg rotation here, otherwise things like
#         # the SA don't work without more sophistication
#         self.local_half_extents = self.compute_half_extents(config["shape"])
#         # self.half_extents = self.C0 @ half_extents
#
#         # generate (normalized) inertia and rotate it into the body frame
#         local_R2 = self.compute_normalized_inertia(
#             config["shape"], config["com_offset"]
#         )
#         self.R2 = self.C0 @ local_R2 @ self.C0.T
#
#         self._bounded_balanced_object = None
#
#     def box(self):
#         return geometry.Box3d(self.local_half_extents, self.position, self.C0)
#
#     @property
#     def height(self):
#         return self.box().height()
#
#     @staticmethod
#     def compute_normalized_inertia(shape_config, com_offset):
#         """Get the inertia matrix divided by mass."""
#         pass
#
#     def compute_support_area(self):
#         # TODO the most general way to do this is to do the intersection with
#         # the parent
#         pass
#
#     @property
#     def offset(self):
#         # TODO this needs to be dealt with
#         if "offset" in self.d:
#             return np.array(self.d["offset"])
#         return np.zeros(2)
#
#     def support_area(self):
#         config = self.d["support_area"]
#         shape = config["shape"]
#         if shape == "eq_tri":
#             side_length = config["side_length"]
#             support_area = PolygonSupportArea.equilateral_triangle(side_length)
#             r_tau = math.equilateral_triangle_r_tau(side_length)
#         elif shape == "rect":
#             lx = config["lx"]
#             ly = config["lx"]
#             support_area = PolygonSupportArea.axis_aligned_rectangle(lx, ly)
#             r_tau = math.rectangle_r_tau(lx, ly)
#         elif shape == "circle":
#             radius = config["radius"]
#             support_area = PolygonSupportArea.circle(radius)
#             r_tau = math.circle_r_tau(radius)
#         else:
#             raise ValueError(f"Unsupported support area shape: {shape}")
#         return support_area, r_tau
#
#     def bounded_balanced_object(self):
#         """Generate a BoundedBalancedObject for this object."""
#         if self._bounded_balanced_object is not None:
#             return self._bounded_balanced_object
#
#         # parse the bounded rigid body
#         mass_min = self.d["mass"]["min"]
#         mass_max = self.d["mass"]["max"]
#
#         com_center = self.position
#         com_half_lengths = np.array(self.d["com"]["half_lengths"])
#         com_ellipsoid = Ellipsoid(com_center, com_half_lengths, np.eye(3))
#
#         # parse the radii of gyration
#         # this can be specified to be based on the exact inertia matrix for a
#         # particular shape
#         if "use_exact" in self.d["radii_of_gyration"]:
#             shape = self.d["radii_of_gyration"]["use_exact"]["shape"]
#             if shape == "cylinder":
#                 radius = self.d["radii_of_gyration"]["use_exact"]["radius"]
#                 height = self.d["radii_of_gyration"]["use_exact"]["height"]
#                 inertia = math.cylinder_inertia_matrix(
#                     mass=1, radius=radius, height=height
#                 )
#             elif shape == "cuboid":
#                 side_lengths = np.array(
#                     self.d["radii_of_gyration"]["use_exact"]["side_lengths"]
#                 )
#                 inertia = math.cuboid_inertia_matrix(mass=1, side_lengths=side_lengths)
#             else:
#                 raise ValueError(f"Unrecognized shape {shape}.")
#             # no need to divide out mass, since we used mass=1 above
#             radii_of_gyration = np.sqrt(np.diag(inertia))
#             radii_of_gyration_min = radii_of_gyration
#             radii_of_gyration_max = radii_of_gyration
#         else:
#             radii_of_gyration_min = np.array(self.d["radii_of_gyration"]["min"])
#             radii_of_gyration_max = np.array(self.d["radii_of_gyration"]["max"])
#
#         body = BoundedRigidBody(
#             mass_min=mass_min,
#             mass_max=mass_max,
#             radii_of_gyration_min=radii_of_gyration_min,
#             radii_of_gyration_max=radii_of_gyration_max,
#             com_ellipsoid=com_ellipsoid,
#         )
#
#         support_area, r_tau = self.support_area()
#
#         com_height = self.d["com"]["height"]
#         mu_min = self.d["mu_min"]
#
#         # cache for later retrieval
#         self._bounded_balanced_object = BoundedBalancedObject(
#             body,
#             com_height=com_height,
#             support_area_min=support_area,
#             r_tau_min=r_tau,
#             mu_min=mu_min,
#         )
#         return self._bounded_balanced_object
#
#     def base_contact_points(self, name):
#         """Generate the contact points at the base of this object."""
#         obj = self.bounded_balanced_object()
#         vertices = obj.support_area_min.vertices
#         h = obj.com_height
#
#         contacts = []
#         for vertex in vertices:
#             contact = ContactPoint()
#             contact.object1_name = name
#             contact.mu = obj.mu_min
#             contact.normal = np.array([0, 0, 1])  # TODO fixed for now
#             contact.r_co_o1 = np.array([vertex[0], vertex[1], -h])
#             contacts.append(contact)
#         return contacts
#
#     def update_parent_contact_points(self, parent, contacts):
#         """Update base contact points of this object with parent's information.
#
#         The base contact points of this object are in contact with the (top of
#         the) parent.
#         """
#         # need diff between my (child) CoM and parent's CoM
#         Δ = self.position - parent.position
#         for contact in contacts:
#             contact.r_co_o2 = contact.r_co_o1 + Δ
#             contact.object2_name = self.parent_name
#         return contacts


# def _parse_objects_with_contacts_old(wrappers):
#     contacts = []
#     for name, wrapper in wrappers.items():
#         # generate contacts for the base of this object
#         base_contacts = wrapper.base_contact_points(name)
#
#         # if the object has a parent, we need to add the parent info to the
#         # contact points as well
#         if wrapper.parent_name is not None:
#             parent = wrappers[wrapper.parent_name]
#             wrapper.update_parent_contact_points(parent, base_contacts)
#
#         contacts.extend(base_contacts)
#
#     unwrapped_objects = {
#         name: wrapper.bounded_balanced_object() for name, wrapper in wrappers.items()
#     }
#     return unwrapped_objects, contacts


def _parse_objects_with_contacts(wrappers, contacts):
    """
    wrappers is the dict of name: object wrappers
    neighbours is a list of pairs of names specifying objects in contact
    """
    contacts = []
    for contact in contacts:
        name1 = contact["first"]
        name2 = contact["second"]
        mu = contact["mu"]

        box1 = wrappers[name1].box
        box2 = wrappers[name2].box

        points, normal = geometry.box_box_axis_aligned_contact(box1, box2)

        for i in range(points.shape[0]):
            contact = ContactPoint()
            contact.object1_name = name1
            contact.object2_name = name2
            contact.mu = mu
            contact.normal = normal
            # TODO this also assumes a zero CoM offset
            contact.r_co_o1 = points[i, :] - box1.position
            contact.r_co_o2 = points[i, :] - box2.position
            contacts.append(contact)

    wrappers.pop("ee")
    unwrapped_objects = {
        name: wrapper.balanced_object for name, wrapper in wrappers.items()
    }
    return unwrapped_objects, contacts


def _parse_composite_objects(wrappers):
    # remove the "fake" ee object
    wrappers.pop("ee")

    # find the direct children of each object
    for name, wrapper in wrappers.items():
        wrapper.children = []
        if wrapper.parent_name != "ee":
            wrappers[wrapper.parent_name].children.append(name)

    # convert wrappers to BoundedBalancedObjects as required by the controller
    # and compose them as needed
    composites = {}
    for name, wrapper in wrappers.items():
        # all descendants compose the new object (descendants include the
        # current object)
        descendants = []
        descendant_names = [name]
        queue = deque([wrapper])
        while len(queue) > 0:
            desc_wrapper = queue.popleft()
            descendants.append(desc_wrapper.balanced_object)
            for child_name in desc_wrapper.children:
                queue.append(wrappers[child_name])
                descendant_names.append(child_name)

        # new name includes names of all component objects
        composite_name = "_".join(descendant_names)

        # descendants have already been converted to C++ objects
        composites[composite_name] = compose_bounded_objects(descendants)
    return composites


def parse_local_half_extents(shape_config):
    type_ = shape_config["type"].lower()
    if type_ == "cuboid":
        return 0.5 * np.array(shape_config["side_lengths"])
    elif type_ == "cylinder":
        r = shape_config["radius"]
        h = shape_config["height"]
        w = np.sqrt(2) * r
        return np.array([w, w, 0.5 * h])
    raise ValueError(f"Unsupported shape type: {type_}")


def compute_box(shape_config, position=None, rotation=None):
    if rotation is None:
        rotation = np.eye(3)
    C = rotation

    type_ = shape_config["type"].lower()
    local_half_extents = parse_local_half_extents(shape_config)

    # for the cylinder, we rotate by 45 deg about z so that contacts occur
    # aligned with x-y axes
    if type_ == "cylinder":
        C = C @ math.rotz(np.pi / 4)

    return geometry.Box3d(local_half_extents, position, C)


def compute_radii_of_gyration(shape_config, com_offset):
    # TODO handle the com_offset: this is best done when we can use inertia
    # matrices straight-up again
    type_ = shape_config["type"].lower()
    if type_ == "cylinder":
        inertia = math.cylinder_inertia_matrix(
            mass=1, radius=shape_config["radius"], height=shape_config["height"]
        )
    elif type_ == "cuboid":
        inertia = math.cuboid_inertia_matrix(mass=1, side_lengths=shape_config["side_lengths"])
    r_gyr = np.sqrt(np.diag(inertia))
    return r_gyr


def compute_support_area(box, parent_box, inset, tol=1e-6):
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
    if inset < 0:
        raise ValueError("Support area inset must be non-negative.")

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
        lengths[0] == lengths[2] and lengths[1] == lengths[3]
    ), f"Support area is not rectangular!"

    # translate points to be relative to the box's centroid
    local_points = points - box.position
    local_points_xy = local_points[:, :2]

    support_area = PolygonSupportArea(list(local_points_xy), inset=inset)
    r_tau = math.rectangle_r_tau(lengths[0], lengths[1])
    return support_area, r_tau


class SimpleWrapper:
    def __init__(self, balanced_object, box):
        self.balanced_object = balanced_object
        self.box = box


def parse_balanced_object(config, offset_xy, orientation, parent_box, mu):
    C0 = math.quat_to_rot(orientation)

    mass = config["mass"]

    # generate (normalized) inertia and rotate it into the body frame
    local_com_offset = np.array(config["com_offset"])
    if np.linalg.norm(local_com_offset) > 1e-8:
        raise ValueError("com_offset not currently supported")
    local_r_gyr = compute_radii_of_gyration(config["shape"], local_com_offset)
    # TODO just make the constraints accept an inertia like normal (or an H
    # matrix)
    R_gyr = C0 @ np.diag(local_r_gyr) @ C0.T
    r_gyr = np.diag(R_gyr)
    assert np.allclose(R_gyr, np.diag(r_gyr)), "Rotated R_gyr is not diagonal!"

    # compute position of the object centroid
    # TODO things would be somewhat less confusing if the balanced objects
    # directly supported a centroid (object frame) and CoM as separate things
    local_box = compute_box(config["shape"], rotation=C0)
    centroid_position = parent_box.position.copy()
    centroid_position[:2] += offset_xy
    centroid_position[2] += 0.5 * (parent_box.height() + local_box.height())

    com_offset = C0 @ local_com_offset  # relative to centroid
    com_position = centroid_position + com_offset
    com_height = 0.5 * local_box.height() + com_offset[2]

    # now we recompute the box with the correct centroid position
    box = compute_box(config["shape"], centroid_position, C0)

    support_area, r_tau = compute_support_area(
        box, parent_box, config["support_area_inset"]
    )
    # TODO is this correct or should it be positive?
    support_area = support_area.offset(-com_offset[:2])

    body = BoundedRigidBody(
        mass_min=mass,
        mass_max=mass,
        radii_of_gyration_min=np.diag(R_gyr),
        radii_of_gyration_max=np.diag(R_gyr),
        com_ellipsoid=Ellipsoid(com_position, np.zeros(3), np.eye(3)),
    )
    balanced_object = BoundedBalancedObject(
        body,
        com_height=com_height,
        support_area_min=support_area,
        r_tau_min=r_tau,
        mu_min=mu,
    )
    return SimpleWrapper(balanced_object, box)


def parse_mu_dict(contact_config):
    mus = {}
    for contact in contact_config:
        parent_name = contact["first"]
        child_name = contact["second"]
        if parent_name in mus:
            mus[parent_name][child_name] = contact["mu"]
        else:
            mus[parent_name] = {child_name: contact["mu"]}
    return mus


def parse_control_objects(ctrl_config):
    """Parse the control objects and contact points from the configuration."""
    arrangement_name = ctrl_config["balancing"]["arrangement"]
    arrangement = ctrl_config["arrangements"][arrangement_name]
    object_configs = ctrl_config["objects"]

    ee_config = object_configs["ee"]
    ee_box = compute_box(ee_config["shape"], ee_config["position"])
    wrappers = {"ee": SimpleWrapper(None, ee_box)}

    # Parse the dict of friction coefficients for each pair of contacting
    # objects
    # TODO would be nice to move this elsewhere
    mus = parse_mu_dict(arrangement["contacts"])

    for conf in arrangement["objects"]:
        obj_name = conf["name"]
        if obj_name in wrappers:
            raise ValueError(f"Multiple control objects named {obj_name}.")

        obj_type = conf["type"]
        object_config = object_configs[obj_type]

        # object orientation (as a quaternion)
        if "orientation" in conf:
            orientation = np.array(conf["orientation"])
            orientation = orientation / np.linalg.norm(orientation)
        else:
            orientation = np.array([0, 0, 0, 1])

        # get parent information and the coefficient of friction between the
        # two objects
        parent_name = conf["parent"]
        parent_box = wrappers[parent_name].box
        mu = mus[parent_name][obj_name]

        # offset in x-y direction w.r.t. to parent
        offset_xy = np.zeros(2)
        if "offset" in conf:
            offset_xy = parse_support_offset(conf["offset"])

        wrapper = parse_balanced_object(object_config, offset_xy, orientation, parent_box, mu)
        wrapper.parent_name = parent_name
        wrappers[obj_name] = wrapper

        # wrapper = BalancedObjectConfigWrapper(object_config, parent, orientation)
        #
        # # compute position of the object
        # if wrapper.parent_name is not None:
        #     parent = wrappers[wrapper.parent_name]
        #     dz = 0.5 * parent.height + 0.5 * wrapper.height
        #     wrapper.position = parent.position + [0, 0, dz]
        # else:
        #     dz = 0.5 * ee["height"] + 0.5 * wrapper.height
        #     wrapper.position = np.array([0, 0, dz])
        #
        # # add offset in the x-y (support) plane
        # wrapper.position[:2] += wrapper.offset
        #
        # obj_name = conf["name"]
        # if obj_name in wrappers:
        #     raise ValueError(f"Multiple control objects named {obj_name}.")
        # wrappers[obj_name] = wrapper

    if ctrl_config["balancing"]["use_force_constraints"]:
        return _parse_objects_with_contacts(wrappers)
    else:
        composites = _parse_composite_objects(wrappers)
        return composites, []
