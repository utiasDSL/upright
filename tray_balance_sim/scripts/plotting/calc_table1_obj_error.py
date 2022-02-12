import numpy as np
import matplotlib.pyplot as plt

from tray_balance_sim.recording import DATA_DRIVE_PATH
import tray_balance_sim.util as util
from liegroups import SO3

import IPython

DATA_PATHS_HEIGHT100 = [
    DATA_DRIVE_PATH / "single-object/height-1.00" / path
    for path in [
        "h1.0_goal1_all_2021-09-08_23-33-47.npz",
        "h1.0_goal1_nn_2021-09-08_23-45-41.npz",
        "h1.0_goal1_nf_2021-09-08_23-58-39.npz",
        "h1.0_goal1_no_zmp_2021-09-09_00-05-31.npz",
        "h1.0_goal2_all_2021-09-08_23-35-55.npz",
        "h1.0_goal2_nn_2021-09-08_23-47-21.npz",
        "h1.0_goal2_nf_2021-09-09_00-00-08.npz",
        "h1.0_goal2_no_zmp_2021-09-09_00-06-37.npz",
        "h1.0_goal3_all_2021-09-08_23-42-04.npz",
        "h1.0_goal3_nn_2021-09-08_23-49-46.npz",
        "h1.0_goal3_nf_2021-09-09_00-02-48.npz",
        "h1.0_goal3_no_zmp_2021-09-09_00-07-47.npz",
    ]
]

DATA_PATHS_HEIGHT2 = [
    DATA_DRIVE_PATH / "single-object/height-0.02" / path
    for path in [
        "h0.02_goal1_all_2021-09-08_22-25-44.npz",
        "h0.02_goal1_nn_2021-09-08_22-05-01.npz",
        "h0.02_goal1_nf_2021-09-08_21-57-06.npz",
        "h0.02_goal1_no_zmp_2021-09-08_22-08-43.npz",
        "h0.02_goal2_all_2021-09-08_22-27-05.npz",
        "h0.02_goal2_nn_2021-09-08_22-33-48.npz",
        "h0.02_goal2_nf_2021-09-08_22-39-25.npz",
        "h0.02_goal2_no_zmp_2021-09-08_22-11-16.npz",
        "h0.02_goal3_all_2021-09-08_22-28-36.npz",
        "h0.02_goal3_nn_2021-09-08_22-35-32.npz",
        "h0.02_goal3_nf_2021-09-08_22-40-42.npz",
        "h0.02_goal3_no_zmp_2021-09-08_22-13-48.npz",
    ]
]


def compute_object_error(ts, r_ew_ws, Q_wes, r_ow_ws, Q_wos, tf=None):
    """Compute error of an object.

    ts:      Array of times.
    r_ew_ws: Array of end effector positions wrt the world.
    Q_wes:   Array of end effector orientations (as quaternions).
    r_ow_ws: Array of object positions wrt the world.
    Q_wos:   Array of object positions wrt the world.
    tf:      Final time: do not use any data after this time.
    """
    # truncate length of data to a given final time
    data_length = ts.shape[0]
    if tf is not None:
        for i in range(ts.shape[0]):
            if ts[i] > tf:
                data_length = i
                break

    # compute position errors at each time step
    r_oe_es = np.zeros_like(r_ow_ws)
    for i in range(r_oe_es.shape[0]):
        C_ew = SO3.from_quaternion(Q_wes[i, :], ordering="xyzw").inv()
        r_oe_es[i, :] = C_ew.dot(r_ow_ws[i, :] - r_ew_ws[i, :])
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    r_err_norm = np.linalg.norm(r_oe_e_err, axis=1)
    print(f"max position error = {np.max(r_err_norm[:data_length]) * 1000} mm")

    # the rotation between EE and tray should be constant throughout the
    # tracjectory, so there error is the deviation from the starting
    # orientation
    Q_oe0 = util.quat_multiply(util.quat_inverse(Q_wos[0, :]), Q_wes[0, :])
    Q_eo_err = np.zeros(Q_wos.shape[0])
    for i in range(Q_eo_err.shape[0]):
        Q_eo = util.quat_multiply(util.quat_inverse(Q_wes[i, :]), Q_wos[i, :])
        Q_eo_err[i] = util.quat_error(util.quat_multiply(Q_oe0, Q_eo))
    # print(f"max angle error = {np.max(Q_eo_err[:data_length])} rad")


def main():
    for path in DATA_PATHS_HEIGHT2:
        with np.load(path) as data:
            ts = data["ts"]
            r_ew_ws = data["r_ew_ws"]
            Q_wes = data["Q_wes"]
            r_ow_ws = data["r_ow_ws"]
            Q_wos = data["Q_wos"]

            print(f"{path.name}")
            compute_object_error(
                ts, r_ew_ws, Q_wes, r_ow_ws[0, :, :], Q_wos[0, :, :], tf=4
            )
            print("")


if __name__ == "__main__":
    main()
