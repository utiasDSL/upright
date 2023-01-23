import numpy as np

import upright_core as core
import upright_control as ctrl
from mobile_manipulation_central import ros_utils


def parse_mpc_observation_msgs(msgs, normalize_time=True):
    ts = []
    xs = []
    us = []

    for msg in msgs:
        ts.append(msg.time)
        xs.append(msg.state.value)
        us.append(msg.input.value)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(xs), np.array(us)


def parse_object_error(bag, tray_vicon_name, object_vicon_name, return_times=False):
    """Parse error of object over time.

    Error is the distance of the object from its initial position w.r.t. the tray.

    If return_times is True, then the messages times are also returned.
    Otherwise just the error (in meters) is returned.
    """
    tray_topic = ros_utils.vicon_topic_name(tray_vicon_name)
    tray_msgs = [msg for _, msg, _ in bag.read_messages(tray_topic)]

    obj_topic = ros_utils.vicon_topic_name(object_vicon_name)
    obj_msgs = [msg for _, msg, _ in bag.read_messages(obj_topic)]

    # parse and align messages
    ts, tray_poses = ros_utils.parse_transform_stamped_msgs(
        tray_msgs, normalize_time=False
    )
    ts_obj, obj_poses = ros_utils.parse_transform_stamped_msgs(
        obj_msgs, normalize_time=False
    )
    r_ow_ws = np.array(ros_utils.interpolate_list(ts, ts_obj, obj_poses[:, :3]))
    t0 = ts[0]
    ts -= t0

    n = len(ts)
    r_ot_ts = []
    for i in range(n):
        r_tw_w, Q_wt = tray_poses[i, :3], tray_poses[i, 3:]
        r_ow_w = r_ow_ws[i, :]

        # tray rotation w.r.t. world
        C_wt = core.math.quat_to_rot(Q_wt)
        C_tw = C_wt.T

        # compute offset of object in tray's frame
        r_ot_w = r_ow_w - r_tw_w
        r_ot_t = C_tw @ r_ot_w
        r_ot_ts.append(r_ot_t)

    r_ot_ts = np.array(r_ot_ts)

    # compute distance w.r.t. the initial position
    r_ot_t_err = r_ot_ts - r_ot_ts[0, :]
    print(f"Initial offset of object w.r.t. tray = {r_ot_ts[0, :]} (distance = {np.linalg.norm(r_ot_ts[0, :])})")
    distances = np.linalg.norm(r_ot_t_err, axis=1)
    if return_times:
        return distances, ts
    return distances


def parse_mpc_solve_times(
    bag, topic_name="/mobile_manipulator_mpc_policy", max_time=None, return_times=False
):
    """Parse times to solve for a new MPC policy from a bag file.

    If max_time is supplied, only the solve_times for messages that were
    received within max_time seconds of the first messages are included.
    Otherwise, all solve times are returned.

    Returns the array of times, in milliseconds.
    """
    policy_msgs = [msg for _, msg, _ in bag.read_messages(topic_name)]
    solve_times = np.array([msg.solveTime for msg in policy_msgs])

    policy_times = np.array([t.to_sec() for _, _, t in bag.read_messages(topic_name)])
    policy_times -= policy_times[0]

    if max_time is not None:
        assert max_time > 0

        # trim off any solve times from times > max_time, relative to the time
        # that the first message was received
        if max_time > policy_times[-1]:
            max_idx = len(policy_times)
        else:
            max_idx = np.argmax(policy_times > max_time)
        solve_times = solve_times[:max_idx]
        policy_times = policy_times[:max_idx]

    if return_times:
        return solve_times, policy_times
    return solve_times


def parse_config_and_control_model(config_path):
    """Load the config and the associated control model from a yaml file path."""
    config = core.parsing.load_config(config_path)
    ctrl_config = config["controller"]
    return config, ctrl.manager.ControllerModel.from_config(ctrl_config)
