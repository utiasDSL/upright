from tray_balance_sim.camera import Camera
from tray_balance_sim.recording import VideoRecorder


class BalancedObjectCamera(Camera):
    def __init__(self, robot):
        r_ew_w, _ = robot.link_pose()
        super().__init__(
            target_position=r_ew_w + [0, 0, 0.1],
            camera_position=r_ew_w + [0.4, -1, 0.6],
        )

    def save_frame(self):
        super().save_frame("balanced_objects.png")


class RobotCamera(Camera):
    def __init__(self, robot):
        r_ew_w, _ = robot.link_pose()
        super().__init__(
            target_position=r_ew_w + [-0.4, 0, -0.3],
            camera_position=r_ew_w + [0.5, -2, 0.5],
        )

    def save_frame(self):
        super().save_frame("robot.png")


class DynamicObstacleCamera(Camera):
    def __init__(self):
        super().__init__(
            distance=1.8,
            roll=0,
            pitch=-29,
            yaw=147.6,
            target_position=[1.28, 0.045, 0.647],
        )


# dynamic obstacle course POV #1
class DynamicObstacleVideoRecorder1(VideoRecorder):
    def __init__(self, path):
        super().__init__(
            path=path,
            distance=1.8,
            roll=0,
            pitch=-29,
            yaw=147.6,
            target_position=[1.28, 0.045, 0.647],
        )


# dynamic obstacle course POV #2
class DynamicObstacleVideoRecorder2(VideoRecorder):
    def __init__(self, path):
        super().__init__(
            path=path,
            distance=2.6,
            roll=0,
            pitch=-20.6,
            yaw=-3.2,
            target_position=[1.28, 0.045, 0.647],
        )


# static obstacle course POV #1
class StaticObstacleVideoRecorder1(VideoRecorder):
    def __init__(self, path):
        super().__init__(
            path=path,
            distance=4.8,
            roll=0,
            pitch=-13.4,
            yaw=87.6,
            target_position=[2.77, 0.043, 0.142],
        )


# static obstacle course POV #2
class StaticObstacleVideoRecorder2(VideoRecorder):
    def __init__(self, path):
        super().__init__(
            path=path,
            distance=3.4,
            roll=0,
            pitch=-23.4,
            yaw=10.0,
            target_position=[2.77, 0.043, 0.142],
        )


# static obstacle course POV #3
class StaticObstacleVideoRecorder3(VideoRecorder):
    def __init__(self, path):
        super().__init__(
            path=path,
            distance=3.6,
            roll=0,
            pitch=-38.2,
            yaw=-39.6,
            target_position=[1.66, -0.31, 0.03],
        )


class PoseToPoseVideoRecorder1(VideoRecorder):
    def __init__(self, directory, name):
        super().__init__(
            directory=directory,
            name=name,
            distance=4,
            roll=0,
            pitch=-35.8,
            yaw=42,
            target_position=[1.28, 0.045, 0.647],
        )


class PoseToPoseVideoRecorder2(VideoRecorder):
    def __init__(self, directory, name):
        super().__init__(
            directory=directory,
            name=name,
            distance=1.4,
            roll=0,
            pitch=-32.2,
            yaw=91.6,
            target_position=[1.375, -1.097, 1.039],
        )
