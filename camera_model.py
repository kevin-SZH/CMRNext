"""Camera intrinsics and projection with a positive camera Z axis.
"""

import numpy as np
import torch


class CameraModel:

    """Represent camera intrinsics and project positive-Z camera-frame points.
    """
    def __init__(self, focal_length=None, principal_point=None, skew=0.0):
        """Initialize CameraModel from the supplied settings.

        Args:
            focal_length: Focal length.
            principal_point: Principal point.
            skew: Skew.
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.skew = skew

    def set_intrinsics(self, parameters):
        """Set the focal lengths, principal point, and optional skew.

        Args:
            parameters: Parameters.
        """
        self.focal_length = parameters[:2]
        self.principal_point = parameters[2:4]
        self.skew = parameters[4] if len(parameters) > 4 else 0.0
        return self

    def project_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None, change_frame=False):
        """Project camera-frame points to pixel coordinates using the configured intrinsics.

        Args:
            xyz: Xyz: torch.Tensor.
            image_size: Image size.
            reflectance: Per-point intensity values, if enabled.
            change_frame: Whether to apply the coordinate-frame conversion.

        Geometry:
            Extrinsics map LiDAR column vectors to the camera frame, with translation
            in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
        """
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
        if change_frame:
            order = [1, 2, 0, 3]
            xyzw = xyz[order, :]
        else:
            order = [0, 1, 2, 3]
            xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = (self.focal_length[0] * xyzw[0, :] / xyzw[2, :]
                    + self.skew * xyzw[1, :] / xyzw[2, :]
                    + self.principal_point[0])
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0,:] < image_size[1])
        indexes = indexes & (uv[1,:] < image_size[0])
        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes]

        return uv

    def get_matrix(self):
        """Return the camera intrinsic matrix including skew.
        """
        matrix = np.zeros([3, 3])
        matrix[0, 0] = self.focal_length[0]
        matrix[0, 1] = self.skew
        matrix[1, 1] = self.focal_length[1]
        matrix[0, 2] = self.principal_point[0]
        matrix[1, 2] = self.principal_point[1]
        matrix[2, 2] = 1.0
        return matrix
