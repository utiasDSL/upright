#!/usr/bin/env python3
import os
import numpy as np
import sys

import tray_balance_constraints as core
from tray_balance_ocs2.trajectory import StateInputMapping
from tray_balance_ocs2.robot import PinocchioRobot

from cyipopt import minimize_ipopt

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    # get config path from command line argument
    config_path = sys.argv[1]
    config = core.parsing.load_config(config_path)["controller"]["robot"]
    robot = PinocchioRobot(config)
    mapping = StateInputMapping(robot.dims)

    # nominal state to stay near
    q_nom = np.array([0.5, -0.25, 0, 0.25, 0.5, -0.583]) * np.pi
    v_nom = np.zeros(robot.dims.v)
    a_nom = np.zeros(robot.dims.v)
    # a_nom = np.random.random(robot.dims.v) - 0.5
    x_nom = np.concatenate((q_nom, v_nom, a_nom))

    # initial guess
    x0 = x_nom.copy()

    # desired EE state
    # TODO this is not sophisticated enough: we need to just optimize for -9.81
    # in the negative z direction, and not be concerned about other accelerations
    robot.forward(x_nom)
    Pd = np.concatenate(robot.link_pose())
    Vd = np.zeros(6)
    Ad = np.array([0, 0, -9.81, 0, 0, 0])

    def objective(x):
        Δx = x_nom - x
        return 0.5 * Δx @ Δx

    def objective_jac(x):
        Δx = x_nom - x
        return Δx

    def constraint(x):
        robot.forward(x)

        P = np.concatenate(robot.link_pose())
        V = np.concatenate(robot.link_velocity())
        A = np.concatenate(robot.link_acceleration())

        # return np.concatenate(((P - Pd)[:6], V - Vd, A - Ad))
        return np.concatenate((x[:6] - x_nom[:6], V - Vd, [A[2] - Ad[2]]))

    def constraint_jac(x):
        robot.forward_derivatives(x)

        dVdq, dVdv = robot.link_velocity_derivatives()
        dAdq, dAdv, dAda = robot.link_acceleration_derivatives()

        # TODO I think this is wrong: don't we want to analytic Jacobian here?
        dPdq = dVdv
        # dPdq = robot.jacobian(x[:6])

        Z = np.zeros((robot.dims.q, robot.dims.q))

        # return np.block([[dPdq, Z, Z], [dVdq, dVdq, Z], [dAdq, dAdv, dAda]])
        return np.block([[np.eye(6), Z, Z], [dVdq, dVdq, Z], [dAdq[2, :], dAdv[2, :], dAda[2, :]]])

    cons = [{"type": "eq", "fun": constraint, "jac": constraint_jac}]
    res = minimize_ipopt(
        objective, jac=objective_jac, x0=x0, constraints=cons, options={"disp": 5}
    )
    IPython.embed()


if __name__ == "__main__":
    main()
