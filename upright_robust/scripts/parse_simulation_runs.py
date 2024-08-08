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
import tqdm

import upright_control as ctrl
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob

import IPython

FAILURE_DIST_THRESHOLD = 0.5  # meters

OBJECT_NAME = "block1"
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

    def as_dict(self):
        return {
            "max_linear_acc": float(self.max_linear_acc),
            "max_angular_acc": float(self.max_angular_acc),
            "max_linear_vel": float(self.max_linear_vel),
            "max_angular_vel": float(self.max_angular_vel),
            "max_tilt_angle": float(self.max_tilt_angle),
            "max_obj_dist": float(self.max_obj_dist),
        }

    def update(
        self,
        linear_acc=0,
        angular_acc=0,
        linear_vel=0,
        angular_vel=0,
        tilt_angle=0,
        obj_dist=0,
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

        self.max_linear_acc = max(self.max_linear_acc, linear_acc)
        self.max_angular_acc = max(self.max_angular_acc, angular_acc)
        self.max_linear_vel = max(self.max_linear_vel, linear_vel)
        self.max_angular_vel = max(self.max_angular_vel, angular_vel)
        self.max_tilt_angle = max(self.max_tilt_angle, tilt_angle)
        self.max_obj_dist = max(self.max_obj_dist, obj_dist)


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


def compute_run_bounds(directory, check_constraints=True, exact_com=False):
    """Compute the bounds for a single run."""
    config_path, npz_path = parse_run_dir(directory)
    config = core.parsing.load_config(config_path)

    ctrl_config = config["controller"]
    # ctrl_config2 = copy.deepcopy(ctrl_config)
    ctrl_config["balancing"]["arrangement"] = ARRANGEMENT_NAME

    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot

    # TODO note this is using the mu values from the controller
    # no approx_inertia because we want the actual realizable bounds
    objects, contacts = rob.parse_objects_and_contacts(
        ctrl_config, model=model, compute_bounds=True, approx_inertia=False
    )

    # objects2, contacts2 = rob.parse_objects_and_contacts(
    #     ctrl_config2, compute_bounds=True, approx_inertia=False
    # )

    if check_constraints:
        obj0 = list(objects.values())[0]
        mass = obj0.body.mass
        com_box = obj0.com_box
        bounding_box = obj0.bounding_box

        com = obj0.body.com

        params0 = rg.InertialParameters(
            mass=mass, com=com, I=obj0.body.inertia, translate_from_com=True
        )

        names = list(objects.keys())
        name_index = rob.compute_object_name_index(names)
        H = rob.compute_cwc_face_form(name_index, contacts)
        G = rob.compute_contact_force_to_wrench_map(name_index, contacts)
        F = block_diag(*[c.F for c in contacts])

    data = np.load(npz_path)
    ts = data["ts"]
    xs = data["xs"]
    fs = data["contact_forces"]
    # odcs = data["object_dynamics_constraints"]
    xds = data["xds"]

    r_ew_ws = data["r_ew_ws"]
    Q_wes = data["Q_wes"]
    r_ow_ws = data["r_ow_ws"]
    r_oe_es = []

    z = np.array([0, 0, 1])

    # compute desired goal position
    robot.forward_xu(x=model.settings.initial_state)
    r0 = robot.link_pose()[0]
    waypoint = np.array(ctrl_config["waypoints"][0]["position"])
    assert len(ctrl_config["waypoints"]) == 1
    rd = r0 + waypoint

    results = RunResults()
    goal_dists = []

    for i in tqdm.trange(ts.shape[0]):
        x = xs[i, :]
        robot.forward_xu(x=x)

        r_ew_w, C_we = robot.link_pose(rotation_matrix=True)
        V_e = np.concatenate(robot.link_velocity(frame="local"))
        G_e = rob.body_gravity6(C_ew=C_we.T)
        A_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

        # compute distance to goal
        goal_dists.append(np.linalg.norm(rd - r_ew_w))

        if check_constraints:
            Y = rg.RigidBody.regressor(V=V_e, A=A_e - G_e)
            θ = cp.Variable(10)
            J = rg.pim_must_equal_vec(θ)
            c = J[:3, 3] / mass  # CoM
            m = J[3, 3]  # mass
            for h in H:
                objective = cp.Maximize(h @ Y @ θ)
                constraints = rg.pim_psd(J) + bounding_box.must_realize(J) + [m == mass]
                if exact_com:
                    constraints.append(c == com)
                else:
                    constraints.extend(com_box.must_contain(c))
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

        # compute object position w.r.t. EE
        C_we = core.math.quat_to_rot(Q_wes[i, :])
        r_oe_e = C_we.T @ (r_ow_ws[i, 0, :] - r_ew_ws[i, :])
        r_oe_es.append(r_oe_e)

    # compute maximum *change* from initial object position w.r.t. to EE
    r_oe_es = np.array(r_oe_es)
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    distances = np.linalg.norm(r_oe_e_err, axis=1)
    results.update(obj_dist=np.max(distances))

    return results, ts, goal_dists


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
    parser.add_argument(
        "--exact-com",
        action="store_true",
        help="Assume no uncertainty in the CoM.",
    )
    args = parser.parse_args()

    # iterate through all directories
    dirs = glob.glob(args.directory + "/*/")
    dirs.sort(key=sort_dir_key)

    results = RunResults()
    num_failures = 0
    all_goal_dists = []
    all_times = []
    for d in dirs:
        print(Path(d).name)
        run_results, times, goal_dists = compute_run_bounds(
            d, check_constraints=args.check_constraints, exact_com=args.exact_com
        )
        all_times.append(times)
        all_goal_dists.append(goal_dists)
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

    outfile = Path(args.directory) / "data.npz"
    np.savez(outfile, times=np.array(all_times), goal_dists=np.array(all_goal_dists))
    print(f"Dumped data to {outfile}")


if __name__ == "__main__":
    main()
