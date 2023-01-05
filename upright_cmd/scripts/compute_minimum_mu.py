import argparse
import numpy as np
from scipy.optimize import minimize
from spatialmath.base import rpy2r

import upright_core as core
import upright_control as ctrl

import IPython


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    args = parser.parse_args()

    config = core.parsing.load_config(args.config)

    settings = ctrl.manager.ControllerSettings(config["controller"])
    dims = settings.dims
    contacts = settings.balancing_settings.contacts
    objects = settings.balancing_settings.objects
    gravity = settings.gravity

    # optimization variables
    # Q = np.array([0, 0, 0, 1])  # orientation
    rpy = np.zeros(3)
    mus = np.zeros(dims.c)  # friction coefficients
    fs = np.zeros(dims.f())  # forces

    # J = np.concatenate((np.zeros_like(Q), np.ones_like(mus), np.zeros_like(fs)))

    # TODO doesn't make sense
    # mu0s = np.zeros_like(mus)

    x0 = np.concatenate((rpy, mus, fs))

    z = np.array([0, 0, 1])

    def unwrap_opt_vars(x):
        rpy = x[:3]
        mus = x[3 : 3 + dims.c]
        fs = x[3 + dims.c :]
        return rpy, mus, fs

    def cost(x):
        _, mus, fs = unwrap_opt_vars(x)
        # return np.sum(mus)  # + 0.01 * np.sum(np.abs(fs))
        return 0.5 * mus @ mus

    def jac(x):
        _, mus, fs = unwrap_opt_vars(x)
        J = np.zeros_like(x)
        J[3 : 3 + dims.c] = mus
        return J

    def eq_constraints(x):
        rpy, mus, fs = unwrap_opt_vars(x)
        C = rpy2r(rpy)

        # (static) EE state
        state = core.bindings.RigidBodyState.Zero()
        state.pose.orientation = C

        # update friction coefficients
        for i in range(dims.c):
            contacts[i].mu = mus[i]

        return core.bindings.compute_object_dynamics_constraints(
            objects, contacts, fs, state, gravity
        )

    def ineq_constraints(x):
        _, mus, fs = unwrap_opt_vars(x)

        # update friction coefficients
        for i in range(dims.c):
            contacts[i].mu = mus[i]

        return core.bindings.compute_contact_force_constraints_linearized(contacts, fs)

    bounds = (
        [(None, None) for _ in range(3)]
        + [(0, None) for _ in range(dims.c)]
        + [(None, None) for _ in range(dims.f())]
    )

    res = minimize(
        cost,
        x0,
        method="slsqp",
        jac=jac,
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": eq_constraints},
            {"type": "ineq", "fun": ineq_constraints},
        ],
    )
    rpy, mus, fs = unwrap_opt_vars(res.x)
    C = rpy2r(rpy)

    IPython.embed()


main()
