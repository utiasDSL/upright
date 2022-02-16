
from tray_balance_sim.camera import Camera


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
