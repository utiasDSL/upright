import numpy as np

from tray_balance_constraints import Ellipsoid, BoundedRigidBody, BoundedBalancedObject


class ControlObjectConfigWrapper:
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
        if "offset" in self.d:
            return np.array(self.d["offset"])
        return np.zeros(2)

    def support_area(self):
        config = self.d["support_area"]
        shape = config["shape"]
        if shape == "eq_tri":
            side_length = config["side_length"]
            support_area = ocs2.PolygonSupportArea.equilateral_triangle(side_length)
            r_tau = geometry.equilateral_triangle_r_tau(side_length)
        elif shape == "rect":
            lx = config["lx"]
            ly = config["lx"]
            support_area = ocs2.PolygonSupportArea.axis_aligned_rectangle(lx, ly)
            r_tau = geometry.rectangle_r_tau(lx, ly)
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
        radii_of_gyration = np.array(self.d["radii_of_gyration"])
        com_ellipsoid = con.Ellipsoid(com_center, com_half_lengths, np.eye(3))
        body = con.BoundedRigidBody(
            mass_min=mass_min,
            mass_max=mass_max,
            radii_of_gyration=radii_of_gyration,
            com_ellipsoid=com_ellipsoid,
        )

        support_area, r_tau = self.support_area()

        com_height = self.d["com"]["height"]
        mu_min = self.d["mu_min"]

        return con.BoundedBalancedObject(
            body,
            com_height=com_height,
            support_area_min=support_area,
            r_tau_min=r_tau,
            mu_min=mu_min,
        )


def parse_control_objects(r_ew_w, ctrl_config):
    arrangement_name = ctrl_config["balancing"]["arrangement"]
    arrangement = ctrl_config["arrangements"][arrangement_name]
    object_configs = ctrl_config["objects"]

    wrappers = {}
    for conf in arrangement:
        name = conf["name"]
        parent_name = conf["parent"] if "parent" in conf else None
        object_config = object_configs[name]
        wrapper = ControlObjectConfigWrapper(object_config, parent_name)

        # compute position of the object
        if wrapper.parent_name is not None:
            parent = wrappers[wrapper.parent_name]
            dz = 0.5 * parent.height + 0.5 * wrapper.height
            wrapper.position = parent.position + [0, 0, dz]
        else:
            dz = 0.5 * EE_HEIGHT + 0.5 * wrapper.height
            wrapper.position = r_ew_w + [0, 0, dz]

        # add offset in the x-y (support) plane
        wrapper.position[:2] += wrapper.offset

        wrappers[name] = wrapper

    # find the direct children of each object
    for name, wrapper in wrappers.items():
        if wrapper.parent_name is not None:
            wrappers[wrapper.parent_name].children.append(name)

    # convert wrappers to BoundedBalancedObjects as required by the controller
    # and compose them as needed
    composites = []
    for wrapper in wrappers.values():
        # all descendants compose the new object
        descendants = []
        queue = deque([wrapper])
        while len(queue) > 0:
            wrapper = queue.popleft()
            descendants.append(wrapper.bounded_balanced_object())
            for name in wrapper.children:
                queue.append(wrappers[name])

        # descendants have already been converted to C++ objects
        composites.append(con.BoundedBalancedObject.compose(descendants))

    return composites
