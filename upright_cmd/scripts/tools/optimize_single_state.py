#!/usr/bin/env python3
import os
import numpy as np
import sys
import time

import upright_core as core
import upright_control as ctrl

from cyipopt import minimize_ipopt

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    # get config path from command line argument
    config_path = sys.argv[1]
    config = core.parsing.load_config(config_path)["controller"]
    model = ctrl.manager.ControllerModel.from_config(config)
    mapping = ctrl.trajectory.StateInputMapping(model.robot.dims)

    balancing_constraint_wrapper = ctrl.bindings.BalancingConstraintWrapper(
        model.settings
    )

    # nominal state to stay near
    # TODO maybe take from config?
    q_nom = np.array([0.5, -0.25, 0, 0.25, 0.5, -0.583]) * np.pi
    v_nom = np.zeros(model.robot.dims.v)
    a_nom = np.zeros(model.robot.dims.v)
    # a_nom = np.array([0, 5, 6, 0, 0, 0])
    # a_nom = np.random.random(robot.dims.v) - 0.5
    x_nom = np.concatenate((q_nom, v_nom, a_nom))
    u_nom = np.zeros(model.robot.dims.u)

    W = np.diag(np.concatenate((np.ones(6), 0.1 * np.ones(6), 0.01 * np.ones(6))))

    # initial guess
    x0 = x_nom.copy()

    # approx = balancing_constraint_wrapper.getLinearApproximation(0, x_nom, u_nom)
    # model.robot.forward(x_nom)
    # A = np.concatenate(model.robot.link_acceleration())
    # IPython.embed()

    # desired EE state
    # TODO this is not sophisticated enough: we need to just optimize for -9.81
    # in the negative z direction, and not be concerned about other accelerations
    # model.robot.forward(x_nom)
    # Pd = np.concatenate(model.robot.link_pose())
    # Vd = np.zeros(6)
    # Ad = np.array([0, 0, -9.81, 0, 0, 0])

    def objective(va):
        x = np.concatenate((q_nom, va))
        Δx = x_nom - x
        return 0.5 * Δx @ W @ Δx

    def objective_jac(va):
        x = np.concatenate((q_nom, va))
        Δx = x_nom - x
        return W @ Δx

    def objective_hess(x):
        return W

    def eq_constraint(x):
        # keep joint position fixed
        return x[:6] - x_nom[:6]

    def eq_constraint_jac(x):
        Z = np.zeros((model.robot.dims.q, model.robot.dims.q))
        return np.hstack((np.eye(6), Z, Z))

    def constraint(va):
        x = np.concatenate((q_nom, va))
        approx = balancing_constraint_wrapper.getLinearApproximation(0, x, u_nom)
        # only using the first constraint here
        return approx.f

    def constraint_jac(va):
        x = np.concatenate((q_nom, va))
        approx = balancing_constraint_wrapper.getLinearApproximation(0, x, u_nom)
        # Z = np.zeros((model.robot.dims.q, model.robot.dims.q))
        # return np.block([[np.eye(6), Z, Z], [approx.dfdx[0, :]]])
        return approx.dfdx[:, 6:]

        # robot.forward_derivatives(x)
        #
        # dVdq, dVdv = robot.link_velocity_derivatives()
        # dAdq, dAdv, dAda = robot.link_acceleration_derivatives()
        #
        # # TODO I think this is wrong: don't we want to analytic Jacobian here?
        # dPdq = dVdv
        # # dPdq = robot.jacobian(x[:6])
        #
        # Z = np.zeros((robot.dims.q, robot.dims.q))
        #
        # # return np.block([[dPdq, Z, Z], [dVdq, dVdq, Z], [dAdq, dAdv, dAda]])
        # return np.block(
        #     [[np.eye(6), Z, Z], [dVdq, dVdq, Z], [dAdq[2, :], dAdv[2, :], dAda[2, :]]]
        # )

    # TODO we can also have a constraint that tries to keep the inverted
    # orientation
    cons = [
        {"type": "ineq", "fun": constraint, "jac": constraint_jac},
        # {"type": "eq", "fun": eq_constraint, "jac": eq_constraint_jac},
    ]
    t1 = time.time()
    res = minimize_ipopt(
        objective, jac=objective_jac, x0=np.concatenate((v_nom, a_nom)), constraints=cons, options={"disp": 5}
    )
    print(time.time() - t1)

    x = np.concatenate((q_nom, res.x))
    cons = balancing_constraint_wrapper.getLinearApproximation(0, x, u_nom).f

    print(f"Optimal x = {x}")
    print(f"Constraints at optimum = {cons}")

    IPython.embed()


if __name__ == "__main__":
    main()
