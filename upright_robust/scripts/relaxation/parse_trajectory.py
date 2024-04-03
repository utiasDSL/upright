"""Parse simulated trajectory data from an npz file to determine appropriate
bounds for relaxed problem."""
import argparse
from pathlib import Path
import glob

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

import upright_control as ctrl
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob

import IPython


MU_BOUND = 0.01


def parse_npz_dir(directory):
    """Parse npz and config path from a data directory.

    Returns (config_path, npz_path), as strings."""
    dir_path = Path(directory)

    config_paths = glob.glob(dir_path.as_posix() + "/*.yaml")
    assert len(config_paths) == 1, f"Found {len(config_paths)} config files."
    config_path = config_paths[0]

    npz_paths = glob.glob(dir_path.as_posix() + "/*.npz")
    assert len(npz_paths) == 1, f"Found {len(npz_paths)} npz files."
    npz_path = npz_paths[0]

    return config_path, npz_path


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing config and npz file.")
    args = parser.parse_args()

    config_path, npz_path = parse_npz_dir(args.directory)
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
    F = rob.compute_cwc_face_form(name_index, contacts)

    # mixed form of the CWC: {w = W @ f | A @ f >= 0}
    W = rob.compute_contact_force_to_wrench_map(name_index, contacts)
    nf = 3 * len(contacts)
    A = block_diag(*[c.F for c in contacts])
    f = cp.Variable(nf)

    objective = cp.Minimize([0])

    data = np.load(npz_path)
    ts = data["ts"]
    xs = data["xs"]
    xds = data["xds"]

    z = np.array([0, 0, 1])

    max_constraint_value = -np.infty
    max_linear_acc = -np.infty
    max_angular_acc = -np.infty
    max_linear_vel = -np.infty
    max_angular_vel = -np.infty
    max_tilt_angle = -np.infty

    for t, x in zip(ts, xs):
        # print(f"t = {t}")
        robot.forward_xu(x=x)

        C_we = robot.link_pose(rotation_matrix=True)[1]
        V_e = np.concatenate(robot.link_velocity(frame="local"))
        G_e = rob.body_gravity6(C_ew=C_we.T)
        A_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

        tilt_angle = np.arccos(z @ C_we @ z)
        max_tilt_angle = max(max_tilt_angle, tilt_angle)

        max_linear_acc = max(max_linear_acc, np.linalg.norm(A_e[:3]))
        max_angular_acc = max(max_angular_acc, np.linalg.norm(A_e[3:]))
        max_linear_vel = max(max_linear_vel, np.linalg.norm(V_e[:3]))
        max_angular_vel = max(max_angular_vel, np.linalg.norm(V_e[3:]))

        # V_ew_w = np.concatenate(robot.link_velocity(frame="local_world_aligned"))
        # g_w = np.array([0, 0, -9.81])
        # A_ew_w = np.concatenate(
        #     robot.link_classical_acceleration(frame="local_world_aligned")
        # )
        # A_ew_w[:3] -= g_w
        # # inertial wrench about the CoM
        # # result has already been rotated into the EE frame
        # w_in = np.concatenate(
        #     [
        #         obj.inertial_com_wrench(C_we=C_we, A_ew_w=A_ew_w, V_ew_w=V_ew_w)
        #         for obj in objects.values()
        #     ]
        # )

        # body wrench about the EE origin
        w = np.concatenate(
            [obj.wrench(A=A_e - G_e, V=V_e) for obj in objects.values()]
        )

        constraints = F @ w
        max_constraint_value = max(max_constraint_value, np.max(constraints))
        if np.max(constraints) > 0:
            print(f"face violation = {np.max(F @ w)}")

        # NOTE: alternatively we can solve a feasibility problem
        # constraints = [w == W @ f, A @ f >= 0]
        # problem = cp.Problem(objective, constraints)
        # problem.solve(solver=cp.MOSEK)
        # if problem.status == "optimal":
        #     # print(f"span violation = {np.min(A @ f.value)}")
        #     pass
        # else:
        #     print(problem.status.upper())

    print(f"Max face violation = {max_constraint_value}")
    if max_constraint_value <= 0:
        print(f"We used less than μ = {MU_BOUND}")

    print(f"max linear acc  = {max_linear_acc}")
    print(f"max angular acc = {max_angular_acc}")
    print(f"max linear vel  = {max_linear_vel}")
    print(f"max angular vel = {max_angular_vel}")
    print(f"max tilt angle  = {max_tilt_angle} rad ({np.rad2deg(max_tilt_angle)} deg)")


if __name__ == "__main__":
    main()
