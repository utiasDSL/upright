#!/usr/bin/env python3
"""Compute minimum feasible friction coefficients for a given arrangement."""
import argparse
import numpy as np
from scipy.optimize import minimize
from spatialmath.base import rpy2r

import upright_core as core
import upright_control as ctrl


def update_friction_coefficients(contacts, contact_idx, mus):
    for i in range(len(contacts)):
        contacts[i].mu = mus[contact_idx[i]]


def compute_contact_idx(contacts):
    """Assign indices such that all contact points between the same pair of
    objects has the same value."""
    n = len(contacts)
    contact_idx = np.zeros(n, dtype=int)
    counter = 0
    contact_dict = {}
    for i, contact in enumerate(contacts):
        p = (contact.object1_name, contact.object2_name)
        if p not in contact_dict:
            contact_dict[p] = counter
            counter += 1
        contact_idx[i] = contact_dict[p]
    return contact_idx


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

    # we could optimize each contact mu individually, but for simplicity we
    # assume friction is the same between each pair of objects
    contact_idx = compute_contact_idx(contacts)
    nc = np.max(contact_idx) + 1

    # upper bounds on mu, also used for weighting
    # NOTE: user sets these
    mu0s = np.ones(nc)
    # mu0s = np.array([0.25, 1.0])

    # optimization variables and initial guess
    rpy = np.zeros(3)  # EE orientation
    mus = mu0s.copy()  # friction coefficients
    fs = np.zeros(3 * dims.c)  # contact forces
    x0 = np.concatenate((rpy, mus, fs))

    def unwrap_opt_vars(x):
        rpy = x[:3]
        mus = x[3 : 3 + nc]
        fs = x[3 + nc :]
        return rpy, mus, fs

    def cost(x):
        _, mus, _ = unwrap_opt_vars(x)
        y = mus / np.sqrt(mu0s)
        return 0.5 * y @ y

    def eq_constraints(x):
        rpy, mus, fs = unwrap_opt_vars(x)

        # (static) EE state
        state = core.bindings.RigidBodyState.Zero()
        state.pose.orientation = rpy2r(rpy)

        update_friction_coefficients(contacts, contact_idx, mus)

        return core.bindings.compute_object_dynamics_constraints(
            objects, contacts, fs, state, gravity
        )

    def ineq_constraints(x):
        _, mus, fs = unwrap_opt_vars(x)
        update_friction_coefficients(contacts, contact_idx, mus)
        return core.bindings.compute_contact_force_constraints_linearized(contacts, fs)

    bounds = (
        [(None, None) for _ in range(3)]
        + [(0, mu0s[i]) for i in range(nc)]
        + [(None, None) for _ in range(3 * dims.c)]
    )

    assert len(bounds) == len(x0)

    res = minimize(
        cost,
        x0,
        method="slsqp",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": eq_constraints},
            {"type": "ineq", "fun": ineq_constraints},
        ],
    )

    if not res.success:
        print("Optimization was not successful!")

    rpy, mus, fs = unwrap_opt_vars(res.x)
    print(f"μ = {mus}")


main()
