
def optimize_acceleration_robust_sequential(C, V, ad, obj, a_bound=5, α_bound=1):
    """Optimize acceleration with different constraints sequentially,
    constraining the current problem so that the previous one is no worse.

    This is just as slow as the full problem and underestimates the feasible
    set.
    """
    contacts = obj.contacts()
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then one set of duals
    nf = F.shape[0]
    nλ = obj.p.shape[0]
    nz = 10
    nv = 6 + nz
    x0 = np.zeros(nv)

    # bounds on acceleration
    lb = -np.inf * np.ones(nv)
    ub = np.inf * np.ones(nv)
    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    # initial guess
    x0[:3] = ad

    # range and nullspace of P.T
    Pr = np.linalg.pinv(obj.P.T)
    Pn = scipy.linalg.null_space(obj.P.T)

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)
    P_sparse = sparse.csc_matrix(P)

    q = np.zeros(nv)
    q[:3] = -ad

    zs = np.zeros((nf, nz))
    Gs = np.zeros((0, nv))
    hs = []

    # solve sequence of QPs
    solve_time_total = 0
    for i in range(nf):
        d0, D = body_regressor_by_vector_matrix(C, V, F[i, :])

        G1 = np.concatenate((obj.p @ Pr @ D, -obj.p @ Pn))
        G2 = np.hstack((Pr @ D, -Pn))
        Gi = np.vstack((G1, G2))
        h = np.concatenate(([-obj.p @ Pr @ d0], -Pr @ d0))

        G_total = np.vstack((Gs, Gi))
        h_total = np.concatenate((hs, h))
        # print(Gi.shape)

        t0 = time.time()
        x = solve_qp(
            P=P_sparse,
            q=q,
            G=sparse.csc_matrix(G_total),
            h=h_total,
            lb=lb,
            ub=ub,
            initvals=x0,
            solver="osqp",
        )
        t1 = time.time()
        solve_time_total += t1 - t0
        A = x[:6]
        z = x[6:]
        G_save = Gi.copy()
        G_save[:, 6:] = 0
        Gs = np.vstack((Gs, G_save))
        hs = np.concatenate((hs, h - Gi[:, 6:] @ z))
    print(f"solve took {solve_time_total} seconds")
    return A


def optimize_acceleration_robust_reparam(C, V, ad, obj, a_bound=10, α_bound=10):
    """Optimize acceleration with robust constraints reparameterized to
    eliminate equality constraints.

    This however destroys the sparsity of the original problem and is thus slower.
    """
    contacts = obj.contacts()
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then a ton of duals
    # {λ_i}
    nf = F.shape[0]
    nλ = obj.p.shape[0]
    nv = 6 + nf * nλ
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # pre-compute Jacobians
    # equality Jacobian
    n_eq = obj.P.shape[1]  # dimension of one set of equality constraints
    N_eq = obj.P.shape[1] * nf  # dim of all equality constraints
    J_eq = np.zeros((N_eq, nv))
    d_eq = np.zeros(N_eq)
    for i in range(nf):
        d0, D = body_regressor_by_vector_matrix(C, V, F[i, :])
        d_eq[i * n_eq : (i + 1) * n_eq] = d0

        # Jacobian w.r.t. A
        J_eq[i * n_eq : (i + 1) * n_eq, :6] = D

        # Jacobian w.r.t. λi
        J_eq[i * n_eq : (i + 1) * n_eq, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.P.T

    # inequality Jacobian
    J_ineq = np.zeros((nf, nv))
    for i in range(nf):
        J_ineq[i, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.p

    # re-formulation using a decomposition: x = Y @ b + Z @ z, where z is now
    # our optimization variable. This appears to be extremely slow.
    Y = np.linalg.pinv(J_eq)
    Z = scipy.linalg.null_space(J_eq)

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)

    q = np.zeros(nv)
    q[:3] = -ad
    q = Z.T @ q + Z.T @ P @ Y @ -d_eq
    P = Z.T @ P @ Z

    lb = np.zeros(nv)
    ub = np.inf * np.ones(nv)

    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    Gm = -J_ineq @ Z
    h = J_ineq @ Y @ -d_eq

    Ga = np.vstack((Gm, Z, -Z))
    ha = np.concatenate((h, ub + Y @ d_eq, -lb - Y @ d_eq))

    z0, _, _, _ = np.linalg.lstsq(Z, x0 + Y @ d_eq, rcond=None)

    t0 = time.time()
    z = solve_qp(
        P=sparse.csc_matrix(P),
        q=q,
        G=sparse.csc_matrix(Ga),
        h=ha,
        initvals=z0,
        # eps_abs=1e-6,
        # eps_rel=1e-6,
        # max_iter=10000,
        solver="proxqp",
    )
    t1 = time.time()
    print(f"solve took {t1 - t0} seconds")
    x = -Y @ d_eq + Z @ z
    A = x[:6]

    IPython.embed()

    return A

