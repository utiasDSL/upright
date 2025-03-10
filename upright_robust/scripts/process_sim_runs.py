#!/usr/bin/env python3
"""Parse simulated trajectory data from an npz file to determine appropriate
bounds for relaxed problem."""
import argparse
import datetime
import glob
from pathlib import Path
import time
import yaml

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
import rigeo as rg
import tqdm

import upright_control as ctrl
import upright_core as core
import upright_robust as rob

import IPython

FAILURE_DIST_THRESHOLD = 0.5  # meters

# set true to also check the ellipsoid verification problem
SOLVE_ELLIPSOID_VERIFICATION = False


class RunData:
    """Summary of a single run."""

    def __init__(self):
        # timesteps
        self.times = []

        # distances to goal
        self.dists_to_goal = []

        # times to solve the planning problem
        self.solve_times = []

        # times to solve a single verification problem
        self.verify_times = []

        # maximum displacement of the object relative to its initial position
        # with respect to the EE across the run
        self.max_obj_err = 0

        self.obj_errs = []

        # constraint violations using the polyhedral moment conditions
        # necessary
        self.constraint_violations_poly = []

        # the inertial parameters corresponding to the worst-case constraint
        # violation
        self.worst_case_params = []

        # height of the CoM (relative to centroid) for the run; this is useful
        # for tracking how height impacts run success
        self.z_offset = 0


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


def max_or_init(field, value):
    if field is None:
        return value
    return max(field, value)


def compute_run_data(directory, check_constraints=True, exact_params=False, mu=None):
    """Compute the bounds for a single run."""
    config_path, npz_path = parse_run_dir(directory)
    config = core.parsing.load_config(config_path)

    # use the nominal configuration to compute the bounds
    ctrl_config = config["controller"]
    ctrl_config["balancing"]["arrangement"] = "nominal"
    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot

    if check_constraints:
        # no approx_inertia because we want the actual realizable bounds
        objects, contacts = rob.parse_objects_and_contacts(
            ctrl_config,
            model=model,
            compute_bounds=True,
            approx_inertia=False,
            mu=mu,
        )

        obj0 = list(objects.values())[0]
        mass = obj0.body.mass
        com_box = obj0.com_box
        bounding_box = obj0.bounding_box
        bounding_ell = bounding_box.mbe()

        # nominal parameters
        params0 = rg.InertialParameters(
            mass=mass, com=obj0.body.com, I=obj0.body.inertia, translate_from_com=True
        )
        θ0 = params0.vec
        # IPython.embed()
        # raise ValueError("stop here")

        name = list(objects.keys())[0]
        contacts0 = [c for c in contacts if c.contact.object2_name == name]
        name_index = rob.compute_object_name_index([name])
        H = rob.compute_cwc_face_form(name_index, contacts0)
        G = rob.compute_grasp_matrix(name_index, contacts0)

        # TODO swapped angular and linear components
        H2 = H.copy()
        H[:, :3] = H2[:, 3:]
        H[:, 3:] = H2[:, :3]

        # setup the constant parts of the optimization problems
        # polyhedron problem
        θ_poly = cp.Variable(10)
        J_poly = rg.pim_must_equal_vec(θ_poly)
        c_poly = J_poly[:3, 3] / mass  # CoM
        m_poly = J_poly[3, 3]  # mass

        constraints_poly = (
            rg.pim_psd(J_poly)
            + bounding_box.moment_sdp_constraints(J_poly)
            + com_box.must_contain(c_poly)
            + [m_poly == mass]
        )
        hY_poly = cp.Parameter(10)

        objective_poly = cp.Maximize(hY_poly @ θ_poly)
        problem_poly = cp.Problem(objective_poly, constraints_poly)

        # ellipsoid problem
        θ_ell = cp.Variable(10)
        J_ell = rg.pim_must_equal_vec(θ_ell)
        c_ell = J_ell[:3, 3] / mass  # CoM
        m_ell = J_ell[3, 3]  # mass

        constraints_ell = (
            rg.pim_psd(J_ell)
            + bounding_ell.moment_constraints(
                J_ell
            )  # only difference from poly problem
            + com_box.must_contain(c_ell)
            + [m_ell == mass]
        )
        hY_ell = cp.Parameter(10)

        objective_ell = cp.Maximize(hY_ell @ θ_ell)
        problem_ell = cp.Problem(objective_ell, constraints_ell)

    data = np.load(npz_path)
    ts = data["ts"]
    xs = data["xs"]
    fs = data["contact_forces"]
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

    run_data = RunData()

    run_data.z_offset = config["simulation"]["objects"]["sim_block"]["com_offset"][2]

    for i in tqdm.trange(ts.shape[0]):
        x = xs[i, :]
        robot.forward_xu(x=x)
        r_ew_w, C_we = robot.link_pose(rotation_matrix=True)

        # compute distance to goal
        run_data.dists_to_goal.append(np.linalg.norm(rd - r_ew_w))

        # compute object position w.r.t. EE
        C_we = core.math.quat_to_rot(Q_wes[i, :])
        r_oe_e = C_we.T @ (r_ow_ws[i, 0, :] - r_ew_ws[i, :])
        r_oe_es.append(r_oe_e)

        # no need to check the planned states after the time horizon
        if check_constraints and ts[i] <= model.settings.mpc.time_horizon:
            # check the *planned* state, rather than the actual one
            xd = xds[i, :]
            robot.forward_xu(x=xd)

            V_e = np.concatenate(robot.link_velocity(frame="local"))
            G_e = rob.body_gravity6(C_ew=C_we.T)
            A_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

            V = rg.SV(linear=V_e[:3], angular=V_e[3:])
            A = rg.SV(linear=A_e[:3] - G_e[:3], angular=A_e[3:] - G_e[3:])
            Y = rg.RigidBody.regressor(V=V, A=A)

            if exact_params:
                max_violation = None
                for h in H:
                    violation_poly = h @ Y @ θ0
                    max_violation = max_or_init(max_violation, violation_poly)
                run_data.constraint_violations_poly.append(max_violation)
            else:
                max_violation = None
                # worst_case_params = None
                t0 = time.time()
                for h in H:
                    hY_poly.value = h @ Y
                    problem_poly.solve(solver=cp.MOSEK)
                    assert problem_poly.status == "optimal"

                    # if max_violation is None or objective_poly.value > max_violation:
                    #     worst_case_params = θ_poly.value
                    max_violation = max_or_init(max_violation, objective_poly.value)

                    # if objective_poly.value > 7:
                    #     print("big violation!")
                    #     IPython.embed()

                run_data.constraint_violations_poly.append(max_violation)
                t1 = time.time()
                # run_data.worst_case_params.append(worst_case_params)
                run_data.verify_times.append(t1 - t0)

                # we can also check the ellipsoid constraints, if desired
                if SOLVE_ELLIPSOID_VERIFICATION:
                    for h in H:
                        hY_ell.value = h @ Y
                        problem_ell.solve(solver=cp.MOSEK)
                        assert problem_ell.status == "optimal"

                        run_data.max_constraint_violation_ell = max_or_init(
                            run_data.max_constraint_violation_ell, objective_ell.value
                        )

    # compute maximum *change* from initial object position w.r.t. to EE
    r_oe_es = np.array(r_oe_es)
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    obj_errs = np.linalg.norm(r_oe_e_err, axis=1)

    run_data.max_obj_err = np.max(obj_errs)
    run_data.obj_errs = obj_errs
    run_data.times = ts
    run_data.solve_times = data["solve_times"]  # these are in millseconds

    return run_data


def sort_dir_key(d):
    """Sort run data directories.

    Their names are of the form ``run_[number]_[other stuff]``. Note that the
    number if not padded with zeros to make a fixed width string.

    Directories are sorted in increasing value of ``[number]``.
    """
    name = Path(d).name
    return int(name.split("_")[1])


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing config and npz file.")
    parser.add_argument(
        "--check-constraints",
        default=0,
        type=int,
        help="Check the constraints for the first n runs. Fail if the constraints are violated.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Assume no uncertainty in the inertial parameters.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        help="Friction coefficient to compute violation.",
    )
    args = parser.parse_args()

    # iterate through all directories
    dirs = glob.glob(args.directory + "/*/")
    dirs.sort(key=sort_dir_key)

    results_file_name = f"results_mu{args.mu}.yaml" if args.mu else "results.yaml"
    data_file_name = f"data_mu{args.mu}.npz" if args.mu else "data.npz"

    total_runs = 0
    num_failures = 0

    times = []
    dists_to_goal = []
    solve_times = []
    verify_times = []
    constraint_violations_poly = []

    # maximum object displacement error for each run
    max_obj_errs = []

    obj_errs = []

    # height of the CoM during failed runs
    failure_z_offsets = {}
    failure_waypoints = [0, 0, 0]

    for i, d in enumerate(dirs):
        print(Path(d).name)

        check = i < args.check_constraints
        run_data = compute_run_data(
            d, check_constraints=check, exact_params=args.exact, mu=args.mu
        )

        times.append(run_data.times)
        dists_to_goal.append(run_data.dists_to_goal)
        solve_times.append(run_data.solve_times)

        # keep track of the maximum object displacement for each run
        max_obj_errs.append(run_data.max_obj_err)
        obj_errs.append(run_data.obj_errs)

        # record verification results and time to solve verification problem
        # per trajectory (if we are checking)
        if check:
            constraint_violations_poly.append(run_data.constraint_violations_poly)
            if not args.exact:
                verify_times.append(run_data.verify_times)

        total_runs += 1
        if run_data.max_obj_err >= FAILURE_DIST_THRESHOLD:
            if check:
                print(
                    f">>> constraint violation = {np.max(run_data.constraint_violations_poly)}"
                )
            print(f"{Path(d).name} failed!")
            num_failures += 1

            # keep track of which waypoints are failing
            failure_waypoints[i % 3] += 1

            # keep track of what the height of the CoM was
            if run_data.z_offset in failure_z_offsets:
                failure_z_offsets[run_data.z_offset] += 1
            else:
                failure_z_offsets[run_data.z_offset] = 1

    # worst-case position error at the end of any run
    dists_to_goal = np.array(dists_to_goal)
    final_dists_to_goal = dists_to_goal[:, -1]
    # TODO argmax?
    max_final_dist_to_goal = float(np.max(final_dists_to_goal))

    # maximum constraint violation across all runs
    constraint_violations_poly = np.array(constraint_violations_poly)
    if constraint_violations_poly.size > 0:
        max_constraint_violation_poly = float(np.max(constraint_violations_poly))
    else:
        max_constraint_violation_poly = None

    # average verification times
    # we also compute the average without the first run, since the first run
    # may take longer to do the first problem compilation step
    verify_times = np.array(verify_times)
    if verify_times.size > 0:
        verify_time_avg = np.mean(verify_times, axis=1)
        verify_time_avg_no_first = np.mean(verify_times[:, 1:], axis=1)

        # convert to regular list of floats for yaml export
        verify_time_avg = verify_time_avg.tolist()
        verify_time_avg_no_first = verify_time_avg_no_first.tolist()
    else:
        verify_time_avg = None
        verify_time_avg_no_first = None

    # highest object displacement error across all runs
    max_obj_err_idx = int(np.argmax(max_obj_errs))
    max_obj_err = float(max_obj_errs[max_obj_err_idx])

    # summary of results
    outfile = Path(args.directory) / results_file_name
    with open(outfile, "w") as f:
        yaml.dump(
            data={
                "timestamp": datetime.datetime.now().isoformat(),
                "total_runs": total_runs,
                "num_failures": num_failures,
                "max_obj_err": max_obj_err,
                "max_constraint_violation_poly": max_constraint_violation_poly,
                "max_final_dist_to_goal": max_final_dist_to_goal,
                "max_obj_err_idx": max_obj_err_idx,
                "failure_z_offsets": failure_z_offsets,
                "failure_waypoints": failure_waypoints,
                "verify_time_avg": verify_time_avg,
                "verify_time_avg_no_first": verify_time_avg_no_first,
            },
            stream=f,
        )
    print(f"Wrote results to {outfile}")

    # more detailed data dump
    outfile = Path(args.directory) / data_file_name
    np.savez(
        outfile,
        times=np.array(times),
        dists_to_goal=np.array(dists_to_goal),
        solve_times=np.array(solve_times),
        verify_times=np.array(verify_times),
        max_obj_errs=np.array(max_obj_errs),
        obj_errs=np.array(obj_errs),
        constraint_violations_poly=np.array(constraint_violations_poly),
    )
    print(f"Dumped data to {outfile}")


if __name__ == "__main__":
    main()
