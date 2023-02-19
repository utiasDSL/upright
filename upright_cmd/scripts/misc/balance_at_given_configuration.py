#!/usr/bin/env python3
"""Given a robot configuration, find velocity and acceleration that satisfy balancing constraints."""
import os
import numpy as np
import sys
import time

import upright_core as core
import upright_control as ctrl

import scipy.linalg
from scipy.optimize import minimize

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    # get config path from command line argument
    config_path = sys.argv[1]
    config = core.parsing.load_config(config_path)["controller"]
    model = ctrl.manager.ControllerModel.from_config(config)
    dims = model.settings.dims

    balancing_constraint_wrapper = ctrl.bindings.BalancingConstraintWrapper(
        model.settings
    )

    # nominal state to stay near
    q_nom = np.array([0.5, -0.25, 0, 0.25, 0.5, -0.583]) * np.pi
    v_nom = np.zeros(dims.robot.v)
    a_nom = np.zeros(dims.robot.v)
    x_nom = np.concatenate((q_nom, v_nom, a_nom))
    u_nom = np.zeros(dims.robot.u)

    Wq = np.eye(dims.robot.q)
    Wv = 0.1 * np.eye(dims.robot.v)
    Wa = 0.01 * np.eye(dims.robot.v)

    def objective(va):
        Δv = v_nom - va[:dims.robot.v]
        Δa = a_nom - va[dims.robot.v:]
        return 0.5 * (Δv @ Wv @ Δv + Δa @ Wa @ Δa)

    def objective_jac(va):
        Δv = v_nom - va[:dims.robot.v]
        Δa = a_nom - va[dims.robot.v:]
        return np.concatenate((Wv @ Δv, Wa @ Δa))

    # def objective_hess(x):
    #     return W
    #
    # def eq_constraint(x):
    #     # keep joint position fixed
    #     return x[:6] - x_nom[:6]
    #
    # def eq_constraint_jac(x):
    #     Z = np.zeros((model.robot.dims.q, model.robot.dims.q))
    #     return np.hstack((np.eye(6), Z, Z))
    #
    def constraint(va):
        x = np.concatenate((q_nom, va))
        approx = balancing_constraint_wrapper.getLinearApproximation(0, x, u_nom)
        return approx.f

    def constraint_jac(va):
        x = np.concatenate((q_nom, va))
        approx = balancing_constraint_wrapper.getLinearApproximation(0, x, u_nom)
        # don't include derivative w.r.t. q
        return approx.dfdx[:, dims.robot.q:]

    # optimize v, a leaving q fixed at the nominal configuration s.t. balancing
    # constraints, where we want v, a to be as close to zero as possible
    cons = [
        {"type": "ineq", "fun": constraint, "jac": constraint_jac},
    ]
    t1 = time.time()
    res = minimize(
        objective,
        x0=np.concatenate((v_nom, a_nom)),
        jac=objective_jac,
        constraints=cons,
        method="slsqp",
    )
    dt = time.time() - t1

    x = np.concatenate((q_nom, res.x))
    cons = balancing_constraint_wrapper.getLinearApproximation(0, x, u_nom).f

    print(f"Solver time = {dt}")
    print(f"Optimal x = {x}")
    print(f"Constraints at optimum = {cons}")

    IPython.embed()


if __name__ == "__main__":
    main()
