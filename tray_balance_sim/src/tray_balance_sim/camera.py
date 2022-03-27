import numpy as np
from pyb_utils.camera import Camera


def camera_from_dict(d, r_ew_w=None):
    """Construct a camera from the configuration dict."""
    if r_ew_w is None:
        r_ew_w = np.zeros(3)

    # target position is always required
    # it can be relative to the EE or absolute
    if "relative_target" in d:
        target = r_ew_w + d["relative_target"]
    elif "target" in d:
        target = np.array(d["target"])
    else:
        raise ValueError("Camera target position is required.")

    # camera can be defined by either its position or distance and roll-pitch-yaw
    from_position = True
    if "relative_position" in d:
        position = r_ew_w + d["relative_position"]
    elif "position" in d:
        position = np.array(d["position"])
    elif "distance" in d:
        from_position = False
        distance = d["distance"]
        roll = d["roll"]
        pitch = d["pitch"]
        yaw = d["yaw"]
    else:
        raise ValueError(
            "Camera must be defined by either its position or distance and roll pitch yaw."
        )

    if from_position:
        return Camera.from_camera_position(
            target_position=target, camera_position=position
        )
    else:
        return Camera.from_distance_rpy(
            target_position=target, distance=distance, roll=roll, pitch=pitch, yaw=yaw
        )
