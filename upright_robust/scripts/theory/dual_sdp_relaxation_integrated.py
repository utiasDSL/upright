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


def solve_global_relaxed_dual_approx_inertia(
    obj1,
    obj2,
    F1,
    F2,
    idx,
    ell,
    v_max=1,
    ω_max=0.25,
    a_max=1,
    α_max=0.25,
):
    """Global convex problem based on the face form of the dual constraint formulation.

    Minimize the constraints for obj1 subject to all constraints on obj2, which
    are typically approximations of those on obj1.
    """
    f = F1[idx, :]

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

    # ellipsoid realizability
    Q = ell.Q
    As = rob.pim_sum_vec_matrices()
    νs = np.array([np.trace(Q @ A) for A in As])

    # fmt: off
    P1_tilde = np.block([
        [obj1.P, obj1.p[:, None]],
        [np.zeros((1, obj1.P.shape[1])), np.array([[1]])],
        [-νs[None, :], np.array([[0]])]])
    # P1_tilde = np.block([
    #     [obj1.P, -obj1.p[:, None]],
    #     [np.zeros((1, obj1.P.shape[1])), np.array([[-1]])]])
    # fmt: on
    R1 = rob.span_to_face_form(P1_tilde.T)
    R1 = R1 / np.max(np.abs(R1))

    # fmt: off
    P2_tilde = np.block([
        [obj2.P, obj2.p[:, None]],
        [np.zeros((1, obj2.P.shape[1])), np.array([[1]])]])
    # fmt: on
    R2 = rob.span_to_face_form(P2_tilde.T)
    R2 = R2 / np.max(np.abs(R2))

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    Λ = cp.Variable((6, 6), PSD=True)

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
        ] + [
            # all of the approximate constraints
            R2[:, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A) <= 0.0
            for (Z, D, Dg) in zip(Zs, Ds, Dgs)
        ]
        # fmt: on
        problem = cp.Problem(objective, constraints)
        t0 = time.time()
        problem.solve(solver=cp.MOSEK)
        t1 = time.time()
        # print(f"Δt = {t1 - t0}")
        values.append(problem.value)
        if problem.value > 0:
            print(f"eigvals = {np.linalg.eigvals(Λ.value)}")

        # IPython.embed()
        # raise ValueError()

        # eigs, eigvecs = np.linalg.eig(Λ.value)
        # max_eig_idx = np.argmax(eigs)
        # Vs.append(np.sqrt(eigs[max_eig_idx]) * eigvecs[:, max_eig_idx])

    # min_idx = np.argmin(values)
    # print(f"relaxed value = {values[min_idx]}")
    # print(f"relaxed V = {Vs[min_idx]}")
    n_neg = np.sum(np.array(values) < 0)
    print(
        f"min={np.min(values)}, max={np.max(values)}, ({n_neg} / {len(values)} negative)"
    )
    return np.min(values)


def main_inertia_approx():
    np.set_printoptions(precision=5, suppress=True)

    # TODO parsing
    cli_args = cmd.cli.sim_arg_parser().parse_args()
    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]

    # TODO I should be able to parse just the objects without the controller
    objects = rob.RobustControllerModel(
        ctrl_config,
        timestep=0,
        v_joint_max=0,
        a_joint_max=0,
    ).uncertain_objects

    IPython.embed()
    return

    obj1 = rob.BalancedObject(
        m=1,
        h=0.2,
        δ=0.05,
        μ=0.2,
        h0=0,
        x0=0,
        uncertain_mh=False,
        uncertain_inertia=True,
    )
    obj2 = rob.BalancedObject(
        m=1,
        h=0.2,
        δ=0.04,
        μ=0.2,
        h0=0,
        x0=0,
        uncertain_mh=True,
        uncertain_inertia=False,
        mh_factor=1,
    )
    box = ip.Box(half_extents=[0.05, 0.05, 0.2], center=[0, 0, 0.2])
    # ell = ip.maximum_inscribed_ellipsoid(box.vertices)
    ell = box.minimum_bounding_ellipsoid()

    F1 = rob.cwc(obj1.contacts())
    F2 = rob.cwc(obj2.contacts())
    # norm = np.max(np.abs(F1))
    # F1 /= norm
    # F2 /= norm

    for i in range(F1.shape[0]):
        print(i + 1)
        relaxed = solve_global_relaxed_dual_approx_inertia(obj1, obj2, F1, F2, i, ell)
        # print(f"{i+1}: {relaxed}")


if __name__ == "__main__":
    main_inertia_approx()
