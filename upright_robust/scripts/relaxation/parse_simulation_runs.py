#!/usr/bin/env python3
"""Parse simulated trajectory data from an npz file to determine appropriate
bounds for relaxed problem."""
import argparse
import copy
from pathlib import Path
import glob
import os
import warnings
import yaml

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
import rigeo as rg

import upright_control as ctrl
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob

import IPython

# True to use the face form of the CWC to check the constraints, otherwise use
# the fixed form in a feasible problem
USE_FACE_FORM_CONSTRAINTS = True

# We do *not* compute the minimum friction coefficient required to satisfy the
# constraints at all times. Instead, we specify a (maximum) friction
# coefficient a priori and then *verify* that the constraints are satisfied
# given this much friction.
MU_BOUND = 0

FAILURE_DIST_THRESHOLD = 0.5  # meters

OBJECT_NAME = "nominal_sim_block"
ARRANGEMENT_NAME = "nominal_sim"
MU = 0.2


class RunResults:
    """Summary of maximum values from the run(s)."""

    def __init__(
        self,
        max_linear_acc=0,
        max_angular_acc=0,
        max_linear_vel=0,
        max_angular_vel=0,
        max_tilt_angle=0,
        max_obj_dist=0,
    ):
        self.max_linear_acc = max_linear_acc
        self.max_angular_acc = max_angular_acc
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.max_tilt_angle = max_tilt_angle
        self.max_obj_dist = max_obj_dist
        self.mu_bound = MU_BOUND

        self.max_cwc_violation = None

    def as_dict(self):
        return {
            "max_linear_acc": float(self.max_linear_acc),
            "max_angular_acc": float(self.max_angular_acc),
            "max_linear_vel": float(self.max_linear_vel),
            "max_angular_vel": float(self.max_angular_vel),
            "max_tilt_angle": float(self.max_tilt_angle),
            "max_obj_dist": float(self.max_obj_dist),
            "mu_bound": float(self.mu_bound),
            # "max_cwc_violation": self.max_cwc_violation.tolist(),
        }

    def update(
        self,
        linear_acc=0,
        angular_acc=0,
        linear_vel=0,
        angular_vel=0,
        tilt_angle=0,
        obj_dist=0,
        cwc_violation=None,
    ):
        # we can update directly or from another RunResults object
        if isinstance(linear_acc, RunResults):
            other = linear_acc

            linear_acc = other.max_linear_acc
            angular_acc = other.max_angular_acc
            linear_vel = other.max_linear_vel
            angular_vel = other.max_angular_vel
            tilt_angle = other.max_tilt_angle
            obj_dist = other.max_obj_dist
            cwc_violation = other.max_cwc_violation

        self.max_linear_acc = max(self.max_linear_acc, linear_acc)
        self.max_angular_acc = max(self.max_angular_acc, angular_acc)
        self.max_linear_vel = max(self.max_linear_vel, linear_vel)
        self.max_angular_vel = max(self.max_angular_vel, angular_vel)
        self.max_tilt_angle = max(self.max_tilt_angle, tilt_angle)
        self.max_obj_dist = max(self.max_obj_dist, obj_dist)

        if cwc_violation is not None:
            if self.max_cwc_violation is None:
                self.max_cwc_violation = np.zeros_like(cwc_violation)
            self.max_cwc_violation = np.maximum(self.max_cwc_violation, cwc_violation)


def parse_run_dir(directory):
    """Parse npz and config path from a data directory of a single run.

    Returns (config_path, npz_path), as strings."""
    dir_path = Path(directory)

    config_paths = glob.glob(dir_path.as_posix() + "/*.yaml")
    assert len(config_paths) == 1, f"Found {len(config_paths)} config files."
    config_path = config_paths[0]

    npz_paths = glob.glob(dir_path.as_posix() + "/*.npz")
    assert len(npz_paths) == 1, f"Found {len(npz_paths)} npz files."
    npz_path = npz_paths[0]

    return config_path, npz_path


def compute_run_bounds(directory, check_constraints=True):
    """Compute the bounds for a single run."""
    config_path, npz_path = parse_run_dir(directory)
    config = core.parsing.load_config(config_path)

    ctrl_config = config["controller"]
    ctrl_config2 = copy.deepcopy(ctrl_config)
    ctrl_config["balancing"]["arrangement"] = ARRANGEMENT_NAME

    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot

    # TODO note this is using the mu values from the controller
    # no approx_inertia because we want the actual realizable bounds
    objects, contacts = rob.parse_objects_and_contacts(
        ctrl_config, model=model, compute_bounds=True, approx_inertia=False
    )

    objects2, contacts2 = rob.parse_objects_and_contacts(
        ctrl_config2, compute_bounds=True, approx_inertia=False
    )

    # TODO how do I deal with the large set of wrenches? I am just optimizing
    # θ, so Yθ is a single wrench value... but what about H? recall H does not
    # depend on θ, since we are using the EE origin as the reference point
    # does this mean I need only to parse a single object?
    # TODO should pull out the info like I do in the sim
    obj0 = list(objects.values())[0]
    mass = obj0.body.mass
    com_box = obj0.com_box
    bounding_box = obj0.bounding_box

    names = list(objects.keys())
    name_index = rob.compute_object_name_index(names)
    H = rob.compute_cwc_face_form(name_index, contacts)

    G = rob.compute_contact_force_to_wrench_map(name_index, contacts)
    F = block_diag(*[c.F for c in contacts])

    # if USE_FACE_FORM_CONSTRAINTS:
    #     # face form of the CWC: {w | H @ w <= 0}
    #     H = rob.compute_cwc_face_form(name_index, contacts)
    #
    #     Δw = cp.Variable(len(objects) * 6)
    #     b = cp.Variable(len(objects) * 6)
    #     objective = cp.Minimize(cp.norm(b, 1))
    # else:
    #     # mixed form of the CWC: {w = G @ f | F @ f >= 0}
    #     G = rob.compute_contact_force_to_wrench_map(name_index, contacts)
    #     nf = 3 * len(contacts)
    #     F = block_diag(*[c.F for c in contacts])
    #     f = cp.Variable(nf)
    #     objective = cp.Minimize([0])

    data = np.load(npz_path)
    ts = data["ts"]
    xs = data["xs"]
    fs = data["contact_forces"]
    odcs = data["object_dynamics_constraints"]
    xds = data["xds"]

    r_ew_ws = data["r_ew_ws"]
    Q_wes = data["Q_wes"]
    r_ow_ws = data["r_ow_ws"]
    r_oe_es = []

    z = np.array([0, 0, 1])

    results = RunResults()

    for i in range(ts.shape[0]):
        print(f"i = {i}")
        x = xs[i, :]
        robot.forward_xu(x=x)

        C_we = robot.link_pose(rotation_matrix=True)[1]
        V_e = np.concatenate(robot.link_velocity(frame="local"))
        G_e = rob.body_gravity6(C_ew=C_we.T)
        A_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

        # when there is essentially no tilting, the value can be just over 1.0
        # due to numerical errors, which arccos does not like
        # zCz = z @ C_we @ z
        # if zCz > 1.0 and np.isclose(zCz, 1.0):
        #     zCz = 1.0

        # body wrench about the EE origin
        # w = np.concatenate([obj.wrench(A=A_e - G_e, V=V_e) for obj in objects.values()])
        #
        # results.update(
        #     linear_acc=np.linalg.norm(A_e[:3]),
        #     angular_acc=np.linalg.norm(A_e[3:]),
        #     linear_vel=np.linalg.norm(V_e[:3]),
        #     angular_vel=np.linalg.norm(V_e[3:]),
        #     tilt_angle=np.arccos(zCz),
        #     cwc_violation=H @ w,
        # )

        if check_constraints:
            Y = rg.RigidBody.regressor(V=V_e, A=A_e - G_e)
            θ = cp.Variable(10)
            J = rg.pim_must_equal_vec(θ)
            c = J[:3, 3] / mass  # CoM
            m = J[3, 3]  # mass
            for h in H:
                objective = cp.Maximize(h @ Y @ θ)
                constraints = (
                    rg.pim_psd(J)
                    + com_box.must_contain(c)
                    + bounding_box.must_realize(J)
                    + [m == mass]
                )
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.MOSEK)

                if problem.status != "optimal":
                    print("failed to solve the problem!")
                    IPython.embed()

                if problem.value > 0:
                    print("optimal value is positive!")
                    print("this means constraint could be violated by some I!")
                    params = rg.InertialParameters.from_pim(J.value)

                    w = Y @ θ.value
                    f = cp.Variable(3 * len(contacts))
                    objective2 = cp.Minimize(0)
                    constraints2 = [w == G @ f, F @ f >= 0]
                    problem2 = cp.Problem(objective2, constraints2)
                    problem2.solve(solver=cp.MOSEK)

                    IPython.embed()

        # if USE_FACE_FORM_CONSTRAINTS:
        #     # constraints = H @ w
        #     # if check_constraints and np.max(constraints) > 0:
        #     #     print(f"face violation = {np.max(constraints)}")
        #     #     IPython.embed()
        #     #     raise ValueError(f"μ = {MU_BOUND} not large enough!")
        #
        #     # solve for minimal extra wrench needed to satisfy the constraints
        #     constraints = [H @ (w + Δw) <= 0, -b <= Δw, Δw <= b]
        #     problem = cp.Problem(objective, constraints)
        #     problem.solve(solver=cp.MOSEK)
        #     if problem.status != "optimal":
        #         print("failed to solve!")
        # else:
        #     # actual value of the contact forces
        #     f_act = np.concatenate([fi * c.contact.normal for fi, c in zip(fs[i, :], contacts)])
        #
        #     # alternatively we can solve a feasibility problem
        #     constraints = [w == G @ f, F @ f >= 0]
        #     problem = cp.Problem(objective, constraints)
        #     problem.solve(solver=cp.MOSEK)
        #     if problem.status == "optimal":
        #         # print(f"span violation = {np.min(A @ f.value)}")
        #         pass
        #     else:
        #         print(problem.status.upper())
        #         IPython.embed()
        #         raise ValueError(f"μ = {MU_BOUND} not large enough!")

        # compute object position w.r.t. EE
        C_we = core.math.quat_to_rot(Q_wes[i, :])
        r_oe_e = C_we.T @ (r_ow_ws[i, 0, :] - r_ew_ws[i, :])
        r_oe_es.append(r_oe_e)

    # compute maximum *change* from initial object position w.r.t. to EE
    r_oe_es = np.array(r_oe_es)
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    distances = np.linalg.norm(r_oe_e_err, axis=1)
    results.update(obj_dist=np.max(distances))

    return results


def sort_dir_key(d):
    name = Path(d).name
    return int(name.split("_")[1])


def main():
    # this catches some numerical issues which are just issued as warnings by default
    warnings.filterwarnings("error")

    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing config and npz file.")
    parser.add_argument(
        "--check-constraints",
        action="store_true",
        help="Fail if the constraints are violated with the bounded friction coefficient.",
    )
    args = parser.parse_args()

    # iterate through all directories
    dirs = glob.glob(args.directory + "/*/")
    dirs.sort(key=sort_dir_key)

    results = RunResults()
    num_failures = 0
    for d in dirs[:3]:
        print(Path(d).name)
        run_results = compute_run_bounds(d, check_constraints=args.check_constraints)
        if run_results.max_obj_dist >= FAILURE_DIST_THRESHOLD:
            print(f"{Path(d).name} failed!")
            num_failures += 1
        results.update(run_results)

    outfile = Path(args.directory) / "results.yaml"
    d = results.as_dict()
    d["num_failures"] = num_failures
    with open(outfile, "w") as f:
        yaml.dump(data=d, stream=f)
    print(f"Dumped results to {outfile}")


if __name__ == "__main__":
    main()
