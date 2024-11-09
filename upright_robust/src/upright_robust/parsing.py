import upright_control as ctrl
import upright_robust.modelling as mdl

import rigeo as rg


def parse_objects_and_contacts(
    ctrl_config, model=None, approx_inertia=False, mu=None, compute_bounds=True
):
    """Parse the balancing objects and contact points from config dict.

    Parameters
    ----------
    ctrl_config : dict
        The controller configuration dict.
    model : ControllerModel
        If a model has already been parsed for this config, it can be supplied
        rather than re-parsing it here.
    approx_inertia : bool
        Set to true to use approximate bounds, otherwise use fully realizable
        bounds.
    mu : float, non-negative
        A custom friction coefficient; otherwise, the value from the config is
        used.
    compute_bounds : bool
        True to compute the robust bounds, False otherwise.

    Returns
    -------
    :
        A tuple (objects, contacts) of balancing objects and contact points.
    """
    if model is None:
        model = ctrl.manager.ControllerModel.from_config(ctrl_config)

    # friction coefficient must be non-negative
    if mu is not None:
        assert mu >= 0

    # make EE origin the reference point for all contacts
    bodies = model.settings.balancing_settings.bodies
    for c in model.settings.balancing_settings.contacts:
        if c.object1_name != "ee":
            b1 = bodies[c.object1_name]
            c.r_co_o1 = c.r_co_o1 + b1.com
        b2 = bodies[c.object2_name]
        c.r_co_o2 = c.r_co_o2 + b2.com

    # update friction coefficient if one was passed in
    contacts = []
    for c in model.settings.balancing_settings.contacts:
        if mu is not None:
            c.mu = mu
        contacts.append(mdl.RobustContactPoint(c))

    if approx_inertia:
        bounds_name = "approx"
    else:
        bounds_name = "realizable"

    # parse the bounds for each object
    uncertain_objects = {}
    arrangement_name = ctrl_config["balancing"]["arrangement"]
    arrangement_config = ctrl_config["arrangements"][arrangement_name]
    for conf in arrangement_config["objects"]:
        obj_name = conf["name"]
        obj_type = conf["type"]
        body = bodies[obj_name]

        obj_config = ctrl_config["objects"][obj_type]
        center = body.com
        bounding_box = rg.Box.from_side_lengths(
            obj_config["side_lengths"], center=center
        )

        com_box = None
        if compute_bounds:
            bounds_config = obj_config["bounds"][bounds_name]
            com_lower = center + bounds_config["com_lower"]
            com_upper = center + bounds_config["com_upper"]
            com_box = rg.Box.from_two_vertices(com_lower, com_upper)
        uncertain_objects[obj_name] = mdl.UncertainObject(
            body=body, bounding_box=bounding_box, com_box=com_box
        )

    return uncertain_objects, contacts
