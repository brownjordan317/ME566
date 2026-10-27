# takes an image and camera calibration parameters and returns the undistorted image
import cv2
import numpy as np


class CameraUndistorter:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def undistort(self, image):
        if image is None:
            raise ValueError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.ndim not in (2, 3):
            raise ValueError("Input image must be 2D (grayscale) or 3D (color).")
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return undistorted_image