import upright_control as ctrl
import upright_robust.modelling as mdl


def parse_objects_and_contacts(
    ctrl_config, model=None, approx_inertia=False, mu=None, compute_bounds=True
):
    if model is None:
        model = ctrl.manager.ControllerModel.from_config(ctrl_config)

    # make EE origin the reference point for all contacts
    bodies = model.settings.balancing_settings.bodies
    for c in model.settings.balancing_settings.contacts:
        if c.object1_name != "ee":
            b1 = bodies[c.object1_name]
            c.r_co_o1 = c.r_co_o1 + b1.com
        b2 = bodies[c.object2_name]
        c.r_co_o2 = c.r_co_o2 + b2.com

    contacts = []
    for c in model.settings.balancing_settings.contacts:
        if mu is not None:
            c.mu = mu
        contacts.append(mdl.RobustContactPoint(c))
    # contacts = [
    #     mdl.RobustContactPoint(c) for c in model.settings.balancing_settings.contacts
    # ]

    if approx_inertia:
        bounds_name = "bounds_approx_inertia"
    else:
        bounds_name = "bounds_realizable"

    # parse the bounds for each object
    uncertain_objects = {}
    arrangement_name = ctrl_config["balancing"]["arrangement"]
    arrangement_config = ctrl_config["arrangements"][arrangement_name]
    for conf in arrangement_config["objects"]:
        name = conf["name"]
        type_ = conf["type"]
        if compute_bounds:
            bounds_config = ctrl_config["objects"][type_].get(bounds_name, {})
            bounds = mdl.ObjectBounds.from_config(
                bounds_config, approx_inertia=approx_inertia
            )
            uncertain_objects[name] = mdl.UncertainObject(
                bodies[name], bounds, compute_bounds=True
            )
        else:
            uncertain_objects[name] = mdl.UncertainObject(
                bodies[name], compute_bounds=False
            )

    return uncertain_objects, contacts
