"""SDP relaxations for robust balancing constraints.

We want to find determine the worst-case values of the balancing constraints
over a given set of operating conditions (limits on acceleration, velocity,
etc.) and a set of alternative constraints. These alternative constraints may
be approximate or may be a subset of the true constraints. If the original
constraint is never active, it can be removed from the control problem.
"""
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


# gravity constant
g = 9.81

V_MAX = 2.0
ω_MAX = 1.0
A_MAX = 2.0
α_MAX = 1.0
TILT_ANGLE_MAX = np.deg2rad(30)

# can set to None to use the actual config values
MU_REAL = 0.3
MU_APPROX = 0.03


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
        ctrl_config, model=model, mu=0.01, compute_bounds=False
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

    max_constraint_value = -np.infty

    for t, x in zip(ts, xs):
        print(f"t = {t}")
        robot.forward_xu(x=x)

        C_we = robot.link_pose(rotation_matrix=True)[1]
        V_ew_e = np.concatenate(robot.link_velocity(frame="local"))
        G_e = rob.body_gravity6(C_ew=C_we.T)
        A_ew_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

        V_ew_w = np.concatenate(robot.link_velocity(frame="local_world_aligned"))
        g_w = np.array([0, 0, -9.81])
        A_ew_w = np.concatenate(
            robot.link_classical_acceleration(frame="local_world_aligned")
        )
        A_ew_w[:3] -= g_w

        #
        # # compute about CoM in inertial frame then rotate into body frame
        # obj1 = objects["box1"]
        # m = obj1.body.mass
        # Sc = core.math.skew3(obj1.body.com)
        # I = obj1.body.inertia
        # α_ew_e = C_we.T @ A_ew_w[3:]
        # ω_ew_e = C_we.T @ V_ew_w[3:]
        # ddC_we = (core.math.skew3(A_ew_w[3:]) + core.math.skew3(V_ew_w[3:]) @ core.math.skew3(V_ew_w[3:])) @ C_we
        # f2 = obj1.body.mass * C_we.T @ (A_ew_w[:3] + ddC_we @ obj1.body.com - g_w)
        # τ2 = I @ α_ew_e + np.cross(ω_ew_e, I @ ω_ew_e)
        # w2 = np.concatenate((f2, τ2))
        #
        # # compute about CoM in body frame directly
        # # TODO this is also wrong, because it does not account for the CoM
        # # offset!
        # M = block_diag(m * np.eye(3), I)
        # w3 = M @ (A_ew_e - G_e) + rob.skew6(V_ew_e) @ M @ V_ew_e
        # if t > 0.5:
        #     IPython.embed()
        #     return

        # body wrench about the EE origin
        w = np.concatenate(
            [obj.wrench(A=A_ew_e - G_e, V=V_ew_e) for obj in objects.values()]
        )

        # inertial wrench about the CoM
        # result has already been rotated into the EE frame
        w_in = np.concatenate(
            [
                obj.inertial_com_wrench(C_we=C_we, A_ew_w=A_ew_w, V_ew_w=V_ew_w)
                for obj in objects.values()
            ]
        )

        constraints = F @ w
        max_constraint_value = max(max_constraint_value, np.max(constraints))
        if np.max(constraints) > 0:
            print(f"face violation = {np.max(F @ w)}")

        # constraints = [w_in == W @ f, A @ f >= 0]
        # problem = cp.Problem(objective, constraints)
        # problem.solve(solver=cp.MOSEK)
        # if problem.status == "optimal":
        #     # print(f"span violation = {np.min(A @ f.value)}")
        #     pass
        # else:
        #     print(problem.status.upper())

    print(f"Max face violation = {max_constraint_value}")

    IPython.embed()


if __name__ == "__main__":
    main()
