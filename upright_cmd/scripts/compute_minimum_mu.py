import argparse
import numpy as np
from scipy.optimize import minimize
from spatialmath.base import rpy2r

import upright_core as core
import upright_control as ctrl

import IPython


def main():
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    args = parser.parse_args()

    config = core.parsing.load_config(args.config)

    settings = ctrl.manager.ControllerSettings(config["controller"])
    dims = settings.dims
    contacts = settings.balancing_settings.contacts
    objects = settings.balancing_settings.objects
    gravity = settings.gravity

    # TODO why doesn't this weighting scheme work?
    mu0s = np.ones(dims.c)
    # mu0s = np.array([0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1])

    # optimization variables
    rpy = np.zeros(3)
    mus = mu0s.copy()
    fs = np.zeros(3 * dims.c)  # forces

    x0 = np.concatenate((rpy, mus, fs))

    def unwrap_opt_vars(x):
        rpy = x[:3]
        mus = x[3 : 3 + dims.c]
        fs = x[3 + dims.c :]
        return rpy, mus, fs

    def cost(x):
        _, mus, fs = unwrap_opt_vars(x)
        y = mus / mu0s
        return np.sum(y)
        # return 0.5 * y @ y  #+ 0.005 * fs @ fs

    def jac(x):
        _, mus, fs = unwrap_opt_vars(x)
        J = np.zeros_like(x)
        # J[3 : 3 + dims.c] = mus / (mu0s ** 2)
        J[3 : 3 + dims.c] = 1 / mu0s
        # J[3+dims.c:] = 0.01 * fs
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

        # TODO it would be nice to only optimize over roll and pitch

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
        + [(0, mu0s[i]) for i in range(dims.c)]
        + [(None, None) for _ in range(3 * dims.c)]
    )

    res = minimize(
        cost,
        x0,
        method="slsqp",
        # jac=jac,
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
