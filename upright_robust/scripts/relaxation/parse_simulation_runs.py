#!/usr/bin/env python3
"""Parse simulated trajectory data from an npz file to determine appropriate
bounds for relaxed problem."""
import argparse
from pathlib import Path
import glob
import os

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

import upright_control as ctrl
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob

import IPython


# We do *not* compute the minimum friction coefficient required to satisfy the
# constraints at all times. Instead, we specify a (maximum) friction
# coefficient a priori and then *verify* that the constraints are satisfied
# given this much friction.
MU_BOUND = 0.05


class Results:
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

    def as_dict(self):
        return {
            "max_linear_acc": self.max_linear_acc,
            "max_angular_acc": self.max_angular_acc,
            "max_linear_vel": self.max_linear_vel,
            "max_angular_vel": self.max_angular_vel,
            "max_tilt_angle": self.max_tilt_angle,
            "max_obj_dist": self.max_obj_dist,
            "mu_bound": self.mu_bound,
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
        # we can update directly or from another Results object
        if isinstance(linear_acc, Results):
            other = linear_acc
            self.max_linear_acc = max(self.max_linear_acc, other.max_linear_acc)
            self.max_angular_acc = max(self.max_angular_acc, other.max_angular_acc)
            self.max_linear_vel = max(self.max_linear_vel, other.max_linear_vel)
            self.max_angular_vel = max(self.max_angular_vel, other.max_angular_vel)
            self.max_tilt_angle = max(self.max_tilt_angle, other.max_tilt_angle)
            self.max_obj_dist = max(self.max_tilt_angle, other.max_obj_dist)
        else:
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


def compute_run_bounds(directory):
    """Compute the bounds for a single run."""
    config_path, npz_path = parse_run_dir(directory)
    config = core.parsing.load_config(config_path)
    ctrl_config = config["controller"]
    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot
    objects, contacts = rob.parse_objects_and_contacts(
        ctrl_config, model=model, mu=MU_BOUND, compute_bounds=False
    )

    names = list(objects.keys())
    name_index = rob.compute_object_name_index(names)

    # face form of the CWC: {w | F @ w <= 0}
    # TODO may be intractable for the robust case
    F = rob.compute_cwc_face_form(name_index, contacts)

    # mixed form of the CWC: {w = W @ f | A @ f >= 0}
    # W = rob.compute_contact_force_to_wrench_map(name_index, contacts)
    # nf = 3 * len(contacts)
    # A = block_diag(*[c.F for c in contacts])
    # f = cp.Variable(nf)

    # objective = cp.Minimize([0])

    data = np.load(npz_path)
    ts = data["ts"]
    xs = data["xs"]
    xds = data["xds"]

    r_ew_ws = data["r_ew_ws"]
    Q_wes = data["Q_wes"]
    r_ow_ws = data["r_ow_ws"]
    r_oe_es = []

    z = np.array([0, 0, 1])

    max_constraint_value = -np.infty
    results = Results()

    for i in range(ts.shape[0]):
        x = xs[i, :]
        robot.forward_xu(x=x)

        C_we = robot.link_pose(rotation_matrix=True)[1]
        V_e = np.concatenate(robot.link_velocity(frame="local"))
        G_e = rob.body_gravity6(C_ew=C_we.T)
        A_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

        results.update(
            linear_acc=np.linalg.norm(A_e[:3]),
            angular_acc=np.linalg.norm(A_e[3:]),
            linear_vel=np.linalg.norm(V_e[:3]),
            angular_vel=np.linalg.norm(V_e[3:]),
            tilt_angle=np.arccos(z @ C_we @ z),
        )

        # compute object position w.r.t. EE
        C_we = core.math.quat_to_rot(Q_wes[i, :])
        r_oe_e = C_we.T @ (r_ow_ws[i, 0, :] - r_ew_ws[i, :])
        r_oe_es.append(r_oe_e)

        # body wrench about the EE origin
        w = np.concatenate([obj.wrench(A=A_e - G_e, V=V_e) for obj in objects.values()])

        constraints = F @ w
        # max_constraint_value = max(max_constraint_value, np.max(constraints))
        if np.max(constraints) > 0:
            print(f"face violation = {np.max(constraints)}")
            raise ValueError(f"μ = {MU_BOUND} not large enough!")

        # NOTE: alternatively we can solve a feasibility problem
        # TODO this may be necessary for more complex CWCs
        # constraints = [w == W @ f, A @ f >= 0]
        # problem = cp.Problem(objective, constraints)
        # problem.solve(solver=cp.MOSEK)
        # if problem.status == "optimal":
        #     # print(f"span violation = {np.min(A @ f.value)}")
        #     pass
        # else:
        #     print(problem.status.upper())

    # compute maximum *change* from initial object position w.r.t. to EE
    r_oe_es = np.array(r_oe_es)
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    distances = np.linalg.norm(r_oe_e_err, axis=1)
    results.update(obj_dist=np.max(distances))

    return results


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing config and npz file.")
    args = parser.parse_args()

    # iterate through all directories
    dirs = glob.glob(args.directory + "/*/")
    results = Results()
    for d in dirs:
        run_results = compute_run_bounds(d)
        results.update(run_results)

    outfile = "results.yaml"
    with open(outfile, "w") as f:
        yaml.dump(f, results.as_dict())
    print(f"Dumped results to {outfile}")


if __name__ == "__main__":
    main()
