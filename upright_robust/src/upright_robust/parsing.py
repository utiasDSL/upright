import upright_control as ctrl
import upright_robust.modelling as mdl
import upright_robust.control as robctrl


def parse_objects_and_contacts(ctrl_config, model=None, approx_inertia=False):
    if model is None:
        model = ctrl.manager.ControllerModel.from_config(ctrl_config)

    # make EE origin the reference point for all contacts
    objects = model.settings.balancing_settings.objects
    for c in model.settings.balancing_settings.contacts:
        if c.object1_name != "ee":
            o1 = objects[c.object1_name]
            c.r_co_o1 = c.r_co_o1 + o1.body.com
        o2 = objects[c.object2_name]
        c.r_co_o2 = c.r_co_o2 + o2.body.com

    contacts = [
        mdl.RobustContactPoint(c) for c in model.settings.balancing_settings.contacts
    ]

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
        bounds_config = ctrl_config["objects"][type_].get(bounds_name, {})
        bounds = mdl.ObjectBounds.from_config(
            bounds_config, approx_inertia=approx_inertia
        )
        uncertain_objects[name] = mdl.UncertainObject(objects[name], bounds)

    return uncertain_objects, contacts


class RobustControllerModel:
    def __init__(
        self,
        ctrl_config,
        timestep,
        v_joint_max,
        a_joint_max,
        a_cart_max,
        α_cart_max,
        v_cart_max,
        ω_cart_max,
        a_cart_weight,
        α_cart_weight,
        a_joint_weight,
        v_joint_weight,
        j_joint_weight,
    ):
        # controller
        model = ctrl.manager.ControllerModel.from_config(ctrl_config)
        self.robot = model.robot

        self.uncertain_objects, self.contacts = parse_objects_and_contacts(
            ctrl_config, model=model, approx_inertia=True
        )

        # translational tracking gains
        self.kp = ctrl_config["reactive"]["kp"]
        self.kv = ctrl_config["reactive"]["kv"]

        # balancing controller
        self.controller = parse_controller_from_config(
            ctrl_config,
            self.robot,
            self.uncertain_objects,
            self.contacts,
            timestep,
            a_joint_max=a_joint_max,
            v_joint_max=v_joint_max,
            a_cart_max=a_cart_max,
            α_cart_max=α_cart_max,
            v_cart_max=v_cart_max,
            ω_cart_max=ω_cart_max,
            a_cart_weight=a_cart_weight,
            α_cart_weight=α_cart_weight,
            a_joint_weight=a_joint_weight,
            v_joint_weight=v_joint_weight,
            j_joint_weight=j_joint_weight,
        )


def parse_controller_from_config(
    ctrl_config,
    robot,
    objects,
    contacts,
    timestep,
    v_joint_max,
    a_joint_max,
    a_cart_max,
    α_cart_max,
    v_cart_max,
    ω_cart_max,
    a_cart_weight,
    α_cart_weight,
    a_joint_weight,
    v_joint_weight,
    j_joint_weight,
):
    """Parse the balancing controller from config."""
    use_balancing_constraints = ctrl_config["balancing"]["enabled"]
    tilting_type = ctrl_config["reactive"]["tilting"]
    use_robust_constraints = ctrl_config["reactive"]["robust"]
    use_approx_robust_constraints = ctrl_config["reactive"]["approx_robust"]
    use_face_form = ctrl_config["reactive"]["face_form"]
    use_slack = ctrl_config["reactive"]["use_slack"]
    slack_weight = ctrl_config["reactive"]["slack_weight"]

    # rotational tracking gains
    kθ = ctrl_config["reactive"]["kθ"]
    kω = ctrl_config["reactive"]["kω"]

    shared_params = {
        "robot": robot,
        "objects": objects,
        "contacts": contacts,
        "dt": timestep,
        "use_slack": use_slack,
        "slack_weight": slack_weight,
        "a_cart_max": a_cart_max,
        "α_cart_max": α_cart_max,
        "v_cart_max": v_cart_max,
        "ω_cart_max": ω_cart_max,
        "a_joint_max": a_joint_max,
        "v_joint_max": v_joint_max,
        "a_cart_weight": a_cart_weight,
        "α_cart_weight": α_cart_weight,
        "a_joint_weight": a_joint_weight,
        "v_joint_weight": v_joint_weight,
        "j_joint_weight": j_joint_weight,
    }

    if tilting_type == "full":
        return robctrl.ReactiveBalancingControllerFullTilting(
            kθ=kθ,
            kω=kω,
            use_face_form=use_face_form,
            use_robust_constraints=use_robust_constraints,
            use_approx_robust_constraints=use_approx_robust_constraints,
            use_dvdt_scaling=False,
            **shared_params,
        )
    elif tilting_type == "tray":
        return robctrl.NominalReactiveBalancingControllerTrayTilting(
            kθ=kθ,
            kω=kω,
            use_balancing_constraints=use_balancing_constraints,
            **shared_params,
        )
    elif tilting_type == "flat":
        return robctrl.NominalReactiveBalancingControllerFlat(
            use_balancing_constraints=use_balancing_constraints,
            **shared_params,
        )
    else:
        raise ValueError(f"Unknown tilting type {tilting_type}")
