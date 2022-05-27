from pathlib import Path

import numpy as np
from pyb_utils.camera import Camera, VideoRecorder


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


class VideoManager:
    def __init__(
        self, root_dir, timestamp, video_name, save_frames, timestep, views, ext="avi"
    ):
        self.save = video_name is not None

        # if not saving anything, no need to record
        if not self.save:
            return

        self.save_frames = save_frames
        prefix = video_name + "_" if len(video_name) > 0 else ""
        dir_name = prefix + timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        self.path = root_dir / dir_name
        self.path.mkdir()

        # timestep is in milliseconds
        self.timestep = timestep
        fps = 1000.0 / timestep

        self.recorders = []
        self.frames_dirs = []
        for name, camera in views:
            filename = ".".join([name, ext])
            video_path = self.path / filename
            recorder = VideoRecorder(video_path, camera, fps)
            self.recorders.append(recorder)

            frames_dirname = name + "_frames"
            frames_dir = self.path / frames_dirname
            self.frames_dirs.append(frames_dir)

            # only actually make the directory if we're going to use it
            if save_frames:
                frames_dir.mkdir()

        self.frame_count = 0
        self.last_record_time = -np.infty

    @classmethod
    def from_config_dict(cls, video_name, config, timestamp, r_ew_w=None):
        """Parse the video recording settings from the config.

        Multiple viewpoints can be recorded at the same time.
        """
        root_dir = Path(config["video"]["dir"])
        save_frames = config["video"]["save_frames"]
        timestep = config["video"]["timestep"]  # ms

        views = []
        for view in config["video"]["views"]:
            camera_name = view["camera"]
            camera = camera_from_dict(config["cameras"][camera_name], r_ew_w=r_ew_w)
            views.append((view["name"], camera))

        return cls(
            root_dir=root_dir,
            timestamp=timestamp,
            video_name=video_name,
            save_frames=save_frames,
            timestep=timestep,
            views=views,
        )

    def record(self, t):
        """Record frame at current timestep t.

        The frame is only saved if this timestep aligns with the recording's
        timestep.
        """
        # if not recording, do nothing
        if not self.save:
            return

        if t + self.timestep < self.last_record_time:
            return

        for frames_dir, recorder in zip(self.frames_dirs, self.recorders):
            rgba, _, _ = recorder.camera.get_frame()
            recorder.capture_frame(rgba=rgba)

            if self.save_frames:
                path = frames_dir / f"frame_{self.frame_count}.png"
                recorder.camera.save_frame(path, rgba=rgba)

        self.frame_count += 1
        self.last_record_time = t
