import numpy as np
from scipy.optimize import minimize

from tray_balance_constraints.bindings import (
    Ellipsoid,
    BoundedRigidBody,
    BoundedBalancedObject,
)
from tray_balance_constraints.math import skew3

import IPython


def compute_com(masses, coms):
    assert masses.shape[1] == 1, "masses.shape[1] != 1"
    return np.sum(masses * coms, axis=0) / np.sum(masses)


class BoundingOptProblem:
    def __init__(self, bodies):
        self.bodies = bodies
        self.n_bodies = len(bodies)

        # mass constraint matrices
        self.Am = np.kron(np.eye(self.n_bodies), np.array([[1], [-1]]))
        self.bm = np.concatenate([(-body.mass_min, body.mass_max) for body in bodies])

        self.n_eq_con = np.sum([3 - body.com_ellipsoid.rank() for body in bodies])

    def _parse_args(self, x):
        masses = x[: self.n_bodies].reshape((self.n_bodies, 1))
        coms = x[self.n_bodies :].reshape((self.n_bodies, 3))
        return masses, coms

    def _ineq_constraints(self, x):
        # inequality constraints are formulated to be non-negative

        # always want to call this class's version of _parse_args (child
        # classes may have overridden it to add new variables)
        masses, coms = BoundingOptProblem._parse_args(self, x)

        # each body's mass is within [mass_min, mass_max]
        mass_constraints = self.Am @ masses.flatten() + self.bm

        # each body's CoM is within its bounding ellipsoid
        com_constraints = np.zeros(self.n_bodies)
        for i in range(self.n_bodies):
            delta = coms[i, :] - self.bodies[i].com_ellipsoid.center()
            E = self.bodies[i].com_ellipsoid.E()
            com_constraints[i] = 1 - delta.T @ E @ delta

        return np.concatenate((mass_constraints, com_constraints))

    def _eq_constraints(self, x):
        # equality constraints are required whenever at least one of the CoM
        # ellipsoids of the composing bodies is not full rank
        _, coms = BoundingOptProblem._parse_args(self, x)

        constraints = []
        for i, body in enumerate(self.bodies):
            for j in range(3 - body.com_ellipsoid.rank()):
                V = body.com_ellipsoid.directions()
                c = body.com_ellipsoid.center()
                v = V[:, -(j + 1)]
                constraints.append(v @ (c - coms[i, :]))
        return np.array(constraints)


class BoundingEllipsoidProblem(BoundingOptProblem):
    def __init__(self, ell, bodies):
        self.ell = ell
        super().__init__(bodies)

    def _cost(self, x):
        masses, coms = self._parse_args(x)
        com = compute_com(masses, coms)
        delta = com - self.ell.center()

        # our goal is maximization so make negative
        # sqrt is required for numerical stability
        return -np.sqrt(delta.T @ self.ell.E() @ delta)

    def solve(self, masses0, coms0, method="SLSQP"):
        """Solve the optimization problem given the initial guess."""
        x0 = np.concatenate((masses0, coms0.flatten()))
        constraints = [{"type": "ineq", "fun": self._ineq_constraints}]
        if self.n_eq_con > 0:
            constraints.append({"type": "eq", "fun": self._eq_constraints})
        res = minimize(
            self._cost,
            x0,
            method=method,
            constraints=constraints,
        )
        try:
            assert res.success, "Optimization problem failed to solve."
        except AssertionError as e:
            print(e)
            IPython.embed()
            raise e
        masses_opt, coms_opt = self._parse_args(res.x)
        return masses_opt, coms_opt


class BoundingRadiiOfGyrationProblem(BoundingOptProblem):
    """Bounding radii of gyration based on diagonal approximation."""

    def __init__(self, bodies):
        super().__init__(bodies)

    def _parse_args(self, x):
        s = x[0]
        masses, coms = super()._parse_args(x[1:])
        return s, masses, coms

    def _cost(self, x):
        s, _, _ = self._parse_args(x)

        # our goal is maximization so make negative
        return -s

    def _ineq_constraints(self, x, *args):
        # inequality constraints are formulated to be non-negative
        j = args[0]  # index
        s, masses, coms = self._parse_args(x)
        com = compute_com(masses, coms)

        Ajj = 0
        for i in range(self.n_bodies):
            p_i = com - coms[i, :]
            r_gyr = self.bodies[i].radii_of_gyration_max[j]
            Ajj += masses[i, 0] * (r_gyr ** 2 + p_i @ p_i)
        Ajj /= np.sum(masses)

        psd_constraint = Ajj - s

        mass_com_constraints = super()._ineq_constraints(x[1:])

        return np.concatenate(([psd_constraint], mass_com_constraints))

    def _eq_constraints(self, x):
        return super()._eq_constraints(x[1:])

    def _solve_one(self, x0, j, method="SLSQP"):
        """Solve the optimization problem given the initial guess."""
        constraints = [{"type": "ineq", "fun": self._ineq_constraints, "args": (j,)}]
        if self.n_eq_con > 0:
            constraints.append({"type": "eq", "fun": self._eq_constraints})
        res = minimize(
            self._cost,
            x0,
            method=method,
            constraints=constraints,
        )
        assert res.success, f"Optimization problem failed to solve for index {j}."
        s_opt, masses_opt, coms_opt = self._parse_args(res.x)
        return s_opt, masses_opt, coms_opt

    def solve(self, masses0, coms0, method="SLSQP"):
        """Solve the optimization problem given the initial guess."""
        x0 = np.concatenate(([0], masses0, coms0.flatten()))
        radii_squared_opt = np.zeros(3)
        for j in range(3):
            s_opt, masses_opt, coms_opt = self._solve_one(x0, j, method=method)
            radii_squared_opt[j] = s_opt
        return radii_squared_opt


class BoundingRadiiOfGyrationProblem2(BoundingOptProblem):
    """Alternative bounding radii implementation based on max eigenvalue."""

    def __init__(self, bodies):
        super().__init__(bodies)

    def _parse_args(self, x):
        s = x[0]
        u = x[1:4]
        masses, coms = super()._parse_args(x[4:])
        return s, u, masses, coms

    def _cost(self, x):
        s, _, _, _ = self._parse_args(x)

        # our goal is maximization so make negative
        return -s

    def _ineq_constraints(self, x):
        # inequality constraints are formulated to be non-negative
        s, u, masses, coms = self._parse_args(x)
        com = compute_com(masses, coms)

        A = np.zeros((3, 3))
        for i in range(self.n_bodies):
            P_i = skew3(com - coms[i, :])
            R_i = np.diag(self.bodies[i].radii_of_gyration_max)
            A += masses[i, 0] * (R_i @ R_i + P_i.T @ P_i)
        A /= np.sum(masses)

        ev_constraint = u.T @ A @ u - s
        mass_com_constraints = super()._ineq_constraints(x[4:])
        constraints = np.concatenate(([ev_constraint], mass_com_constraints))
        return constraints

    def _eq_constraints(self, x):
        _, u, _, _ = self._parse_args(x)
        u_norm_constraint = 1 - u @ u
        default_cons = super()._eq_constraints(x[4:])
        return np.concatenate(([u_norm_constraint], default_cons))

    def solve(self, masses0, coms0, method="SLSQP"):
        """Solve the optimization problem given the initial guess."""
        x0 = np.concatenate(([0, 0, 0, 1], masses0, coms0.flatten()))
        constraints = [{"type": "ineq", "fun": self._ineq_constraints}]
        if self.n_eq_con > 0:
            constraints.append({"type": "eq", "fun": self._eq_constraints})
        res = minimize(
            self._cost,
            x0,
            method=method,
            constraints=constraints,
        )
        assert res.success, "Optimization problem failed to solve."
        s_opt, u_opt, masses_opt, coms_opt = self._parse_args(res.x)
        return s_opt * np.ones(3)


def sample_com(bodies, boundary=True):
    """Sample a single possible center of mass for the composite."""
    n = len(bodies)
    masses = np.zeros((n, 1))
    coms = np.zeros((n, 3))
    for i in range(n):
        masses[i], coms[i, :] = bodies[i].sample(boundary=boundary)
    return compute_com(masses, coms)


def compose_com_ellipsoid(bodies, N=100, eps=0.01):
    """Compute an ellipsoid that contains all possible centers of mass of the
    body composed from bodies."""

    # we get a candidate ellipsoid based on random sampling
    sample_coms = np.zeros((N, 3))
    for i in range(N):
        sample_coms[i, :] = sample_com(bodies, boundary=True)
    assert np.isfinite(sample_coms).all(), "Sampled CoMs not finite."

    ell = Ellipsoid.bounding(sample_coms, eps)

    # now we scale that ellipsoid to actually fit all possible CoMs
    problem = BoundingEllipsoidProblem(ell, bodies)
    masses_guess = np.array([0.5 * (body.mass_min + body.mass_max) for body in bodies])
    coms_guess = np.array([body.com_ellipsoid.center() for body in bodies])
    try:
        masses_opt, coms_opt = problem.solve(masses_guess, coms_guess)
    except AssertionError as e:
        IPython.embed()
        raise e

    com = compute_com(masses_opt, coms_opt)
    delta = com - ell.center()
    scale = np.sqrt(delta.T @ ell.E() @ delta)
    scaled_half_lengths = scale * ell.half_lengths()

    return Ellipsoid(ell.center(), scaled_half_lengths, ell.directions())


def compose_radii_of_gyration(bodies):
    """Compute upper bound on possible radii of gyration."""
    # Here we assume radii are along Cartesian directions
    masses_guess = np.array([0.5 * (body.mass_min + body.mass_max) for body in bodies])
    coms_guess = np.array([body.com_ellipsoid.center() for body in bodies])

    problem = BoundingRadiiOfGyrationProblem(bodies)
    radii_squared_opt = problem.solve(masses_guess, coms_guess)
    return np.sqrt(radii_squared_opt)


def compose_bounded_bodies(bodies):
    """Compose a single bounded body out of multiple."""
    mass_min = np.sum([body.mass_min for body in bodies])
    mass_max = np.sum([body.mass_max for body in bodies])
    com_ellipsoid = compose_com_ellipsoid(bodies)
    radii_of_gyration_max = compose_radii_of_gyration(bodies)
    radii_of_gyration_min = np.zeros(3)  # TODO?
    return BoundedRigidBody(
        mass_min=mass_min,
        mass_max=mass_max,
        radii_of_gyration_min=radii_of_gyration_min,
        radii_of_gyration_max=radii_of_gyration_max,
        com_ellipsoid=com_ellipsoid,
    )


def compose_bounded_objects(objects):
    """Compose a single bounded object out of multiple."""
    assert len(objects) > 0, "List of objects is empty."

    # if there is only a single object, then we needn't do anything
    if len(objects) == 1:
        return objects[0]

    body = compose_bounded_bodies([obj.body for obj in objects])

    base = objects[0]
    com_height = (
        base.com_height
        + body.com_ellipsoid.center()[2]
        - base.body.com_ellipsoid.center()[2]
    )

    # TODO need to add an offset to the support area vertices of
    base.body.com_ellipsoid.center() - body.com_ellipsoid.center()

    # support area, mu, r_tau directly inherited from the base object
    sa_min = base.support_area_min
    mu_min = base.mu_min
    r_tau_min = base.r_tau_min

    return BoundedBalancedObject(
        body=body,
        com_height=com_height,
        support_area_min=sa_min,
        r_tau_min=r_tau_min,
        mu_min=mu_min,
    )
