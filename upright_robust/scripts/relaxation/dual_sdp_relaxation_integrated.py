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

import IPython


# gravity constant
g = 9.81

V_MAX = 2.0
ω_MAX = 1.0
A_MAX = 1.0
α_MAX = 1.0
TILT_ANGLE_MAX = np.deg2rad(15)


def schur(X, x):
    y = cp.reshape(x, (x.shape[0], 1))
    return cp.bmat([[X, y], [y.T, [[1]]]])


def solve_constraint_elimination_sdp(
    obj,
    f,
    v_max=V_MAX,
    ω_max=ω_MAX,
    a_max=A_MAX,
    α_max=α_MAX,
    tilt_angle_max=TILT_ANGLE_MAX,
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
            z_normal @ G <= -g * np.cos(tilt_angle_max),

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


def solve_approx_inertia_sdp(
    obj1,
    obj2,
    f,
    F2,
    v_max=V_MAX,
    ω_max=ω_MAX,
    a_max=A_MAX,
    α_max=α_MAX,
    tilt_angle_max=TILT_ANGLE_MAX,
    verbose=False,
):
    """Global convex problem based on the face form of the dual constraint formulation.

    Minimize the constraints for obj1 subject to all constraints on obj2, which
    are typically approximations of those on obj1.
    """
    Z = rob.body_regressor_by_vector_velocity_matrix(f)
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]

    # construct for every constraint
    Zs = []
    Ds = []
    Dgs = []
    for i in range(F2.shape[0]):
        Zs.append(rob.body_regressor_by_vector_velocity_matrix(F2[i, :]))
        Ds.append(rob.body_regressor_by_vector_acceleration_matrix(F2[i, :]))
        Dgs.append(-Ds[-1][:3, :])

    nv = 6
    ng = 3
    nz = Z.shape[0]

    P1_tilde = rob.compute_P_tilde_matrix(obj1.P, obj1.p)
    R1 = rob.span_to_face_form(P1_tilde.T)
    # R1 = R1 / np.max(np.abs(R1))

    P2_tilde = rob.compute_P_tilde_matrix(obj2.P, obj2.p)
    R2 = rob.span_to_face_form(P2_tilde.T)
    # R2 = R2 / np.max(np.abs(R2))

    z_normal = np.array([0, 0, 1])

    A = cp.Variable(nv)  # EE acceleration vector
    G = cp.Variable(ng)  # gravity vector
    z = cp.Variable(nz)
    Λ = cp.Variable((6, 6), PSD=True)  # = V @ V.T

    W = Λ[3:, 3:]
    c = obj1.body.com
    vo = A[:3] - core.math.skew3(c) @ A[3:] + (W - cp.trace(W) * np.eye(3)) @ c

    ω = cp.Variable(3)

    values = []
    Vs = []

    for i in range(R1.shape[0]):
        objective = cp.Maximize(R1[i, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A))
        # fmt: off
        constraints = [
            z == cp.vec(Λ),

            # we constrain z completely through Λ
            cp.diag(Λ[:3, :3]) <= v_max**2,
            cp.diag(Λ[3:, 3:]) <= ω_max**2,

            Λ[:3, 3:] <= v_max * ω_max * np.ones((3, 3)),
            # Λ[3:, 3:] <= ω_max**2 * np.ones((3, 3)),

            # gravity constraints
            cp.norm(G) <= g,
            z_normal @ G <= -g * np.cos(tilt_angle_max),

            # aligned approach
            # (A[:3] - G) @ [1, 0, 0] == 0,
            # (A[:3] - G) @ [0, 1, 0] == 0,

            # adaptive tilting
            # A[3:] == -2 * ω + 1 * core.math.skew3(z_normal) @ (vo - G),
            #
            # ω <= ω_max,
            # ω >= -ω_max,
            # schur(W, ω) >> 0,

            # acceleration constraints
            A[:3] >= -a_max,
            A[:3] <= a_max,
            A[3:] >= -α_max,
            A[3:] <= α_max,

            # cp.norm(A[:3]) <= a_max,
            # cp.norm(A[3:]) <= α_max,
        ] + [
            # all of the approximate constraints
            R2[:, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A) <= 0.0
            for (Z, D, Dg) in zip(Zs, Ds, Dgs)
        ]
        # fmt: on
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        values.append(problem.value)
        if objective.value > 0:
            print(f"index = {i}")
            print(f"objective = {objective.value}")
            print(f"Λ eigvals = {np.linalg.eigvals(Λ.value)}")
            print(f"||g|| = {np.linalg.norm(G.value)}")
            print(f"A = {A.value}")

    values = np.array(values)
    n_neg = np.sum(values < 0)
    if verbose:
        print(
            f"min={np.min(values)}, max={np.max(values)}, ({n_neg} / {len(values)} negative)"
        )
    return np.min(values)


def verify_approx_inertia(ctrl_config):
    objects, contacts = rob.parse_objects_and_contacts(ctrl_config)
    objects_approx, contacts_approx = rob.parse_objects_and_contacts(
        ctrl_config, approx_inertia=True
    )

    names = list(objects.keys())
    name_index = rob.compute_object_name_index(names)
    F = rob.compute_cwc_face_form(name_index, contacts)
    F_approx = rob.compute_cwc_face_form(name_index, contacts_approx)

    # TODO should be reasonably straightforward to extend to multiple objects
    # just need to construct the correct (P, p) polytope data here
    obj = objects[names[0]]
    obj_approx = objects_approx[names[0]]

    print("We want all constraints negative.")

    for i in range(F.shape[0]):
        print(i + 1)
        f = F[i, :]
        relaxed = solve_approx_inertia_sdp(obj, obj_approx, f, F_approx, verbose=True)


def check_elimination(ctrl_config):
    objects, contacts = rob.parse_objects_and_contacts(ctrl_config, approx_inertia=True)
    names = list(objects.keys())
    name_index = rob.compute_object_name_index(names)
    F = rob.compute_cwc_face_form(name_index, contacts)

    # TODO should be reasonably straightforward to extend to multiple objects
    # just need to construct the correct (P, p) polytope data here
    obj = objects[names[0]]

    print("Need constraint to be negative for it to be eliminated.")

    for i in range(F.shape[0]):
        print(i + 1)
        f = F[i, :]
        relaxed = solve_constraint_elimination_sdp(obj, f, verbose=True)
        if relaxed <= 0:
            print(f"Constraint {i + 1} can be eliminated.")


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = cmd.cli.sim_arg_parser()
    parser.add_argument(
        "--elimination",
        action="store_true",
        help="Check for constraints that can be eliminated.",
    )
    parser.add_argument(
        "--verify-approx",
        action="store_true",
        help="Verify that inertia approximation holds.",
    )
    args = parser.parse_args()
    config = core.parsing.load_config(args.config)
    ctrl_config = config["controller"]

    if args.verify_approx:
        verify_approx_inertia(ctrl_config)
    if args.elimination:
        check_elimination(ctrl_config)


if __name__ == "__main__":
    main()
