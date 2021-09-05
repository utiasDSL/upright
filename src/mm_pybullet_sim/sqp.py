from functools import partial
import time

import numpy as np
import qpoases
import osqp
from scipy import sparse
import scipy.optimize

import IPython

# import jax
# import jax.numpy as jnp


# @partial(jax.jit, static_argnums=(0, 4, 5))
def backtrack_line_search(f, df, x, dx, alpha, beta):
    """Backtracking line search.

    Find step size t that minimizes f(x) along descent direction dx

    Parameters:
        f:  The function to evaluate
        df: Derivative of f at x
        x:  Input at which to evaluate f
        dx: Descent direction
        alpha: Tuning parameter in (0, 0.5)
        beta:  Tuning parameter in (0, 1)

    Typical values for alpha are between 0.01 and 0.3; typical values for beta
    are between 0.1 and 0.8.

    See Convex Optimization by Boyd & Vandenberghe, pg. 464

    Returns:
        The step size in (0, 1]
    """
    t = 1
    fx = f(x)
    a = alpha * df @ dx

    while f(x + t * dx) > fx + t * a:
        t = beta * t  # reduce step size

    # def cond_func(t):
    #     return f(x + t * dx) > fx + t * a
    #
    # def body_func(t):
    #     return beta * t  # reduce step size

    # return jax.lax.while_loop(cond_func, body_func, t)
    return t


class Objective:
    """Objective function fun(x) with Jacobian jac(x) and Hessian hess(x)."""

    def __init__(self, fun, jac, hess):
        self.fun = fun
        self.jac = jac
        self.hess = hess

    def value(self, *args, dtype=float):
        return np.array(self.fun(*args), dtype=dtype)

    def jacobian(self, *args, dtype=float):
        return np.array(self.jac(*args), dtype=dtype)

    def hessian(self, *args, dtype=float):
        return np.array(self.hess(*args), dtype=dtype)


# class MultiObjective:
#     def __init__(self, objectives):
#         self.objectives = objectives
#
#     def value(self, *args, dtype=float):
#         return sum([obj.value(*args, dtype=dtype) for obj in self.objectives])
#
#     def jacobian(self, *args, dtype=float):
#         return sum([obj.jacobian(*args, dtype=dtype) for obj in self.objectives])
#
#     def hessian(self, *args, dtype=float):
#         return sum([obj.hessian(*args, dtype=dtype) for obj in self.objectives])


class SparseObjective:
    def __init__(self, obj_jac_hess_func, hess_nz_idx, hess_shape):
        # there are all evaluated in a single function because there is
        # substantial overlap in the terms needed to compute each
        self.obj_jac_hess_func = obj_jac_hess_func

        # build initial sparse Hessian matrix, filled with zeros
        self.hess_nz_idx = hess_nz_idx
        self.hessian = sparse.csc_matrix(
            (np.zeros(hess_shape)[hess_nz_idx], hess_nz_idx), shape=hess_shape
        )

    def evaluate(self, *args):
        self.value, g, H = self.obj_jac_hess_func(*args)

        # explicit array conversion to get rid of jax types
        self.jacobian = np.array(g)

        # update Hessian data rather than creating a new sparse matrix
        # there are some savings here, but it would be better it we could keep
        # H sparse from the start
        # NOTE: conversion from DeviceArray to normal numpy array
        # self.hessian.data = np.array(H)[self.hess_nz_idx]

        # TODO: I'd like to do an update as above, but this doesn't seem to
        # work
        self.hessian = sparse.csc_matrix(
            (np.array(H)[self.hess_nz_idx], self.hess_nz_idx), shape=H.shape
        )


class SparseConstraints:
    def __init__(self, func, jac_func, lb, ub, jac_nz_idx, jac_shape):
        self.func = func
        self.jac_func = jac_func
        self.lb = lb
        self.ub = ub

        self.jac_nz_idx = jac_nz_idx
        self.jac_mat = sparse.csc_matrix(
            (np.zeros(jac_shape)[jac_nz_idx], jac_nz_idx), shape=jac_shape
        )

    def evaluate(self, *args):
        return self.func(*args)

    def jacobian(self, *args):
        J = self.jac_func(*args)
        # self.jac_mat.data = np.array(J)[self.jac_nz_idx]

        jac_mat = sparse.csc_matrix(
            (np.array(J)[self.jac_nz_idx], self.jac_nz_idx), shape=J.shape
        )
        return jac_mat

    def linearized_bounds(self, *args):
        a = self.func(*args)
        lbA = np.array(self.lb - a)
        ubA = np.array(self.ub - a)
        return lbA, ubA


class Constraints:
    """Constraints of the form lb <= fun(x) <= ub.

    jac is the Jacobian of fun w.r.t. x
    nz_idx is the (row, column) indices for elements of the linearized
    constraint matrix that are in general non-zero. This is used for approaches
    that represent matrices sparsely, such as OSQP.
    """

    def __init__(self, fun, jac, lb, ub, nz_idx=None):
        self.fun = fun
        self.jac = jac
        self.lb = lb
        self.ub = ub
        self.nz_idx = nz_idx


class Bounds:
    """Simple bounds of the form lb <= x <= ub."""

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


class Benchmark:
    def __init__(self):
        self.count = 0
        self.time_taken = 0

    def start(self):
        self._time = time.time()

    def end(self):
        _time = time.time()
        dt = _time - self._time
        self._time = _time

        self.count += 1
        self.time_taken += dt

    def print_stats(self):
        print(
            f"count = {self.count}\ntotal time = {self.time_taken}\ntime / call = {self.time_taken / self.count}"
        )


def SQP(*args, solver="qpoases", **kwargs):
    if solver == "qpoases":
        return SQP_qpOASES(*args, **kwargs)
    elif solver == "osqp":
        return SQP_OSQP(*args, **kwargs)
    elif solver == "scipy":
        return SQP_scipy(*args, **kwargs)
    else:
        raise Exception(f"Unknown solver {solver}")


class SQP_scipy:
    def __init__(
        self, nv, obj_fun, obj_jac, ineq_cons, eq_cons, bounds, num_iter=3, verbose=False, var0=None
    ):
        self.obj_fun = obj_fun
        self.obj_jac = obj_jac
        self.ineq_cons = ineq_cons
        self.eq_cons = eq_cons
        self.bounds = scipy.optimize.Bounds(bounds.lb, bounds.ub)

        # keep track of the optimal solution for nominal warm-starting
        if var0 is None:
            self.var = np.zeros(nv)
        else:
            self.var = np.copy(var0)

        self.benchmark = Benchmark()
        self.verbose = verbose

    def fun(self, var, x0, Pd, Vd):
        return self.obj_fun(x0, Pd, Vd, var)

    def jac(self, var, x0, Pd, Vd):
        return self.obj_jac(x0, Pd, Vd, var)

    def solve(self, x0, Pd, Vd):
        """Solve the MPC problem at current state x0 given desired trajectory
        xd."""

        constraints = []
        if self.ineq_cons is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda var, x0, Pd, Vd: self.ineq_cons.fun(x0, Pd, Vd, var),
                "jac": lambda var, x0, Pd, Vd: self.ineq_cons.jac(x0, Pd, Vd, var),
                "args": (x0, Pd, Vd),
            })
        if self.eq_cons is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda var, x0, Pd, Vd: self.eq_cons.fun(x0, Pd, Vd, var),
                "jac": lambda var, x0, Pd, Vd: self.eq_cons.jac(x0, Pd, Vd, var),
                "args": (x0, Pd, Vd),
            })

        self.benchmark.start()
        res = scipy.optimize.minimize(
            self.fun,
            self.var,
            args=(x0, Pd, Vd),
            method="slsqp",
            jac=self.jac,
            bounds=self.bounds,
            constraints=constraints,
            # options={
            #     "maxiter": 10,
            # },
        )
        self.benchmark.end()

        if self.verbose and not res.success:
            print(f"Solver was not successful: {res.message}")

        self.var = res.x

        # return first optimal input
        return self.var


class SQP_qpOASES(object):
    """Generic sequential quadratic program based on qpOASES solver."""

    def __init__(
        self,
        nv,
        nc,
        obj_func,
        constraints,
        bounds,
        num_iter=3,
        num_wsr=100,
        verbose=False,
    ):
        """Initialize the SQP."""
        self.nv = nv
        self.nc = nc
        self.num_iter = num_iter
        self.num_wsr = num_wsr

        self.obj_func = obj_func
        self.constraints = constraints
        self.bounds = bounds

        self.verbose = verbose

        self.qp = qpoases.PySQProblem(nv, nc)
        options = qpoases.PyOptions()
        options.setToMPC()
        # options.setToReliable()
        if verbose:
            options.printLevel = qpoases.PyPrintLevel.MEDIUM
        else:
            options.printLevel = qpoases.PyPrintLevel.LOW
        self.qp.setOptions(options)

        self.qp_initialized = False

        self.benchmark = Benchmark()

    def _lookahead(self, x0, Pd, Vd, var):
        """Generate lifted matrices proprogating the state N timesteps into the
        future."""
        f, g, H = self.obj_func(x0, Pd, Vd, var)
        H = np.array(H, dtype=np.float64)
        g = np.array(g, dtype=np.float64)

        A = np.array(self.constraints.jac(x0, Pd, Vd, var), dtype=np.float64)
        a = np.array(self.constraints.fun(x0, Pd, Vd, var), dtype=np.float64)
        lbA = np.array(self.constraints.lb - a, dtype=np.float64)
        ubA = np.array(self.constraints.ub - a, dtype=np.float64)

        lb = np.array(self.bounds.lb - var, dtype=np.float64)
        ub = np.array(self.bounds.ub - var, dtype=np.float64)

        return H, g, A, lbA, ubA, lb, ub

    def _step(self, x0, Pd, Vd, var, direction):
        """Take a step in the direction."""
        # _, df, _ = self.obj_func(x0, Pd, Vd, var)
        #
        # def merit_func(var):
        #     f, _, _ = self.obj_func(x0, Pd, Vd, var)
        #     a = self.constraints.fun(x0, Pd, Vd, var)
        #     return f - np.sum(np.minimum(0, a))
        #
        # # df = -np.sum(self.constraints.jac(x0, Pd, Vd, var), axis=0)
        #
        # # TODO should I be using the derivative of the merit function?
        #
        # t = backtrack_line_search(merit_func, df, var, direction, alpha=0.25, beta=0.75)
        # return var + t * direction
        return var + direction

    def _iterate(self, x0, Pd, Vd, var):
        delta = np.zeros(self.nv)

        # Initial opt problem.
        H, g, A, lbA, ubA, lb, ub = self._lookahead(x0, Pd, Vd, var)
        if not self.qp_initialized:
            self.qp.init(H, g, A, lb, ub, lbA, ubA, np.array([self.num_wsr]))
            self.qp_initialized = True
        else:
            self.benchmark.start()
            self.qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([self.num_wsr]))
            self.benchmark.end()
        self.qp.getPrimalSolution(delta)

        var = self._step(x0, Pd, Vd, var, delta)

        # Remaining sequence is hotstarted from the first.
        for i in range(self.num_iter - 1):
            H, g, A, lbA, ubA, lb, ub = self._lookahead(x0, Pd, Vd, var)
            self.benchmark.start()
            self.qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([self.num_wsr]))
            self.benchmark.end()
            self.qp.getPrimalSolution(delta)
            var = self._step(x0, Pd, Vd, var, delta)

            if i == self.num_iter - 2:
                v = A @ delta
                mask = np.logical_or(v <= lbA, v >= ubA)
                print(f"number of constraint violations = {np.sum(mask)}")
                print(f"norm of delta = {np.linalg.norm(delta)}")

            # IPython.embed()

        return var

    def solve(self, x0, Pd, Vd):
        """Solve the MPC problem at current state x0 given desired trajectory
        xd."""
        # initialize decision variables
        var = np.zeros(self.nv)

        # iterate to final solution
        var = self._iterate(x0, Pd, Vd, var)

        # return first optimal input
        return var


class SQP_OSQP(object):
    """Generic sequential quadratic program based on OSQP solver."""

    def __init__(
        self,
        nv,
        nc,
        objective,
        constraints,
        bounds,
        num_iter=3,
        verbose=False,
        var0=None,
    ):
        """Initialize the SQP."""
        self.nv = nv
        self.nc = nc
        self.num_iter = num_iter

        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds

        self.qp = osqp.OSQP()
        self.verbose = verbose
        self.qp_initialized = False

        # initial guess for the very first solve
        # subsequent solves reuse the previous optimal solution
        if var0 is None:
            self.var = np.zeros(nv)
        else:
            self.var = np.copy(var0)

        self.benchmark = Benchmark()

    def _lookahead(self, x0, Pd, Vd, var):
        """Generate lifted matrices proprogating the state N timesteps into the
        future."""

        # TODO we can normalize var here

        self.objective.evaluate(x0, Pd, Vd, var)
        g = self.objective.jacobian
        H = self.objective.hessian

        # since OSQP does not use separate bounds, we concatenate onto the
        # constraints
        A1 = self.constraints.jacobian(x0, Pd, Vd, var)
        A2 = sparse.eye(self.nv)  # bounds
        A = sparse.vstack((A1, A2), format="csc")

        lbA, ubA = self.constraints.linearized_bounds(x0, Pd, Vd, var)

        # TODO possibly problematic for SO(3) - should be fine as long as avoid
        # actually constraining elements of SO(3)
        lb = np.array(self.bounds.lb - var)
        ub = np.array(self.bounds.ub - var)

        lower = np.concatenate((lbA, lb))
        upper = np.concatenate((ubA, ub))

        # print(f"nnz(H) = {H.nnz}")
        # print(f"nnz(A) = {A.nnz}")

        return H, g, A, lower, upper

    def _iterate(self, x0, Pd, Vd, var):
        # Initial opt problem.
        H, g, A, lower, upper = self._lookahead(x0, Pd, Vd, var)
        if not self.qp_initialized:
            self.qp.setup(
                P=H,
                q=g,
                A=A,
                l=lower,
                u=upper,
                verbose=self.verbose,
                # linsys_solver="mkl pardiso",
                adaptive_rho=True,
                polish=False,
            )
            self.qp_initialized = True
            results = self.qp.solve()
        else:
            self.benchmark.start()
            self.qp.update(Px=H.data, q=g, Ax=A.data, l=lower, u=upper)
            results = self.qp.solve()
            self.benchmark.end()

        if results.x[0] is None or np.isnan(results.info.obj_val):
            print("Failed to solve")
            IPython.embed()

        var = var + results.x

        # Remaining sequence is hotstarted from the first.
        for i in range(self.num_iter - 1):
            H, g, A, lower, upper = self._lookahead(x0, Pd, Vd, var)
            self.benchmark.start()
            self.qp.update(Px=H.data, q=g, Ax=A.data, l=lower, u=upper)
            results = self.qp.solve()
            self.benchmark.end()

            if results.x[0] is None or np.isnan(results.info.obj_val):
                print("Failed to solve")
                IPython.embed()

            var = var + results.x

        return var

    def solve(self, x0, Pd, Vd):
        """Solve the MPC problem at current state x0 given desired trajectory
        xd."""
        # iterate to final solution
        self.var = self._iterate(x0, Pd, Vd, self.var)

        # return first optimal input
        return self.var