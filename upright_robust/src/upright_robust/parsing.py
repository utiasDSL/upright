import upright_control as ctrl
import upright_robust.modelling as mdl
import upright_robust.control as robctrl


class RobustControllerModel:
    def __init__(self, ctrl_config, timestep):
        # controller
        model = ctrl.manager.ControllerModel.from_config(ctrl_config)
        self.robot = model.robot

        # make EE origin the reference point for all contacts
        objects = model.settings.balancing_settings.objects
        for c in model.settings.balancing_settings.contacts:
            if c.object1_name != "ee":
                o1 = objects[c.object1_name]
                c.r_co_o1 = c.r_co_o1 + o1.body.com
            o2 = objects[c.object2_name]
            c.r_co_o2 = c.r_co_o2 + o2.body.com

        self.contacts = [
            mdl.RobustContactPoint(c)
            for c in model.settings.balancing_settings.contacts
        ]

        # parse the bounds for each object
        self.uncertain_objects = {}
        arrangement_name = ctrl_config["balancing"]["arrangement"]
        arrangement_config = ctrl_config["arrangements"][arrangement_name]
        for conf in arrangement_config["objects"]:
            name = conf["name"]
            type_ = conf["type"]
            bounds_config = ctrl_config["objects"][type_].get("bounds", {})
            bounds = mdl.ObjectBounds.from_config(bounds_config)
            self.uncertain_objects[name] = mdl.UncertainObject(objects[name], bounds)

        # translational tracking gains
        self.kp = ctrl_config["reactive"]["kp"]
        self.kv = ctrl_config["reactive"]["kv"]

        # balancing controller
        self.controller = parse_controller_from_config(
            ctrl_config, self.robot, self.uncertain_objects, self.contacts, timestep
        )


def parse_controller_from_config(ctrl_config, robot, objects, contacts, timestep):
    """Parse the balancing controller from config."""
    use_balancing_constraints = ctrl_config["balancing"]["enabled"]
    tilting_type = ctrl_config["reactive"]["tilting"]
    use_robust_constraints = ctrl_config["reactive"]["robust"]
    use_face_form = ctrl_config["reactive"]["face_form"]

    # rotational tracking gains
    kθ = ctrl_config["reactive"]["kθ"]
    kω = ctrl_config["reactive"]["kω"]
    use_dvdt_scaling = False

    if tilting_type == "full":
        return robctrl.ReactiveBalancingControllerFullTilting(
            robot,
            objects,
            contacts,
            timestep,
            kθ=kθ,
            kω=kω,
            use_dvdt_scaling=use_dvdt_scaling,
            use_face_form=use_face_form,
            use_robust_constraints=use_robust_constraints,
        )
    elif tilting_type == "tray":
        return robctrl.NominalReactiveBalancingControllerTrayTilting(
            robot,
            objects,
            contacts,
            timestep,
            kθ=kθ,
            kω=kω,
            use_balancing_constraints=use_balancing_constraints,
        )
    elif tilting_type == "flat":
        return robctrl.NominalReactiveBalancingControllerFlat(
            robot,
            objects,
            contacts,
            timestep,
            use_balancing_constraints=use_balancing_constraints,
        )
    else:
        raise ValueError(f"Unknown tilting type {tilting_type}")
