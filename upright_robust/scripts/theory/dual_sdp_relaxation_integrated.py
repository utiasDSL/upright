"""Constraint elimination code.

The goal is to prove that certain constraints will never be active over a set
of parameter values and object states (orientation, velocity, acceleration).
"""
import numpy as np
import cvxpy as cp
import time

import upright_core as core
import upright_cmd as cmd
import upright_robust as rob
import inertial_params as ip

import IPython


# gravity constant
g = 9.81


def solve_constraint_elimination_sdp(
    obj,
    f,
    v_max=1,
    ω_max=0.25,
    a_max=1,
    α_max=0.25,
    max_tilt_angle=np.deg2rad(30),
    verbose=False,
):
    """Global convex problem based on the face form of the dual constraint formulation.

    Minimize the constraints for obj1 subject to all constraints on obj2, which
    are typically approximations of those on obj1.
    """
    Z = rob.body_regressor_by_vector_velocity_matrix(f)
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]

    nv = 6
    ng = 3
    nz = Z.shape[0]

    P_tilde = rob.compute_P_tilde_matrix(obj.P, obj.p)
    R = rob.span_to_face_form(P_tilde.T)

    z_normal = np.array([0, 0, 1])

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    Λ = cp.Variable((6, 6), PSD=True)

    values = []
    Vs = []

    for i in range(R.shape[0]):
        objective = cp.Maximize(R[i, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A))
        # fmt: off
        constraints = [
            z == cp.vec(Λ),

            # we constrain z completely through Λ
            cp.diag(Λ[:3, :3]) <= v_max**2,
            cp.diag(Λ[3:, 3:]) <= ω_max**2,

            # gravity constraints
            cp.norm(G) <= g,
            z_normal @ G <= -g * np.cos(max_tilt_angle),

            # aligned approach
            # (A[:3] - G) @ [1, 0, 0] == 0,
            # (A[:3] - G) @ [0, 1, 0] == 0,

            # adaptive tilting
            # A[3:] == 0.1 * core.math.skew3(z_normal) @ (A[:3] - G),

            # acceleration constraints
            A[:3] >= -a_max,
            A[:3] <= a_max,
            A[3:] >= -α_max,
            A[3:] <= α_max,
        ]

        # fmt: on
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        values.append(problem.value)

    values = np.array(values)
    n_neg = np.sum(values < 0)
    if verbose:
        print(
            f"min={np.min(values)}, max={np.max(values)}, ({n_neg} / {len(values)} negative)"
        )
    return np.min(values)


def main():
    np.set_printoptions(precision=5, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()
    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]

    # TODO I should be able to parse just the objects without the controller
    model = rob.RobustControllerModel(
        ctrl_config,
        timestep=0,
        v_joint_max=0,
        a_joint_max=0,
    )
    objects = model.uncertain_objects
    contacts = model.contacts

    names = list(objects.keys())
    name_index = rob.compute_object_name_index(names)
    F = rob.compute_cwc_face_form(name_index, contacts)

    # TODO should be reasonably straightforward to extend to multiple objects
    # just need to construct the correct (P, p) polytope data here
    obj = objects[names[0]]

    # F = rob.cwc(obj.contacts())

    for i in range(F.shape[0]):
        print(i + 1)
        f = F[i, :]
        relaxed = solve_constraint_elimination_sdp(obj, f, verbose=True)
        if relaxed <= 0:
            print(f"Constraint {i + 1} can be eliminated.")


if __name__ == "__main__":
    main()
