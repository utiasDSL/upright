import numpy as np
import pybullet as pyb
from PIL import Image


class Camera:
    def __init__(
        self,
        target_position,
        distance=None,
        roll=None,
        pitch=None,
        yaw=None,
        camera_position=None,
        near=0.1,
        far=1000.0,
        fov=60.0,
        width=1280,
        height=720,
    ):
        self.width = width
        self.height = height
        self.far = far
        self.near = near

        self.position = camera_position
        self.target = target_position

        if camera_position is not None:
            self.view_matrix = pyb.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=target_position,
                cameraUpVector=[0, 0, 1],
            )
        else:
            self.view_matrix = pyb.computeViewMatrixFromYawPitchRoll(
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                cameraTargetPosition=target_position,
                upAxisIndex=2,
            )
        self.proj_matrix = pyb.computeProjectionMatrixFOV(
            fov=fov, aspect=width / height, nearVal=near, farVal=far
        )

    def get_frame(self):
        w, h, rgb, dep, seg = pyb.getCameraImage(
            width=self.width,
            height=self.height,
            shadow=1,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            # flags=pyb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pyb.ER_BULLET_HARDWARE_OPENGL,
        )
        return w, h, rgb, dep, seg

    def save_frame(self, filename):
        w, h, rgb, dep, seg = self.get_frame()
        img = Image.fromarray(np.reshape(rgb, (h, w, 4)), "RGBA")
        img.save(filename)

    def linearize_depth(self, dep):
        """Convert depth map to actual distance from camera.

        See <https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer>.
        """
        dep = 2 * dep - 1
        dep_linear = (
            2.0
            * self.near
            * self.far
            / (self.far + self.near - dep * (self.far - self.near))
        )
        return dep_linear

    def set_camera_pose(self, position, target):
        self.position = position
        self.target = target
        self.view_matrix = pyb.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
        )

    def get_point_cloud(self, dep):
        """Convert depth buffer to 3D point cloud in world coordinates.

        See <https://stackoverflow.com/a/62247245> for the main source of this
        code.
        """

        # view matrix maps world coordinates to camera coordinates (extrinsics)
        V = np.array(self.view_matrix).reshape((4, 4), order="F")

        # camera projection matrix: map camera coordinates to clip coordinates
        # (intrinsics)
        P = np.array(self.proj_matrix).reshape((4, 4), order="F")

        PV_inv = np.linalg.inv(P @ V)

        # dep is stored (height * width) (i.e., transpose of what one might
        # expect on the numpy side)
        points = np.zeros((self.width, self.height, 3))
        for h in range(self.height):
            for w in range(self.width):
                # convert to normalized device coordinates
                # notice that the y-transform is negative---we actually have a
                # left-handed coordinate frame here (x = right, y = down, z =
                # out of the screen)
                x = (2 * w - self.width) / self.width
                y = -(2 * h - self.height) / self.height

                # depth buffer is already in range [0, 1]
                z = 2 * dep[h, w] - 1

                # back out to world coordinates by applying inverted projection
                # and view matrices
                r_ndc = np.array([x, y, z, 1])
                r_world_unnormalized = PV_inv @ r_ndc

                # normalize homogenous coordinates to get rid of perspective
                # divide
                points[w, h, :] = r_world_unnormalized[:3] / r_world_unnormalized[3]
        return points
