import numpy as np
import cv2
import os

from chessboard_grid_size_detector import GridDetector
from chessboard_detector import ChessboardDetector
from image_reader import ImageReader


class CameraCalibrator:
    def __init__(
        self,
        image_dir,
        chessboard_size=None,
        square_size=None,
    ):
        self.images = ImageReader(image_dir).images

        if not self.images:
            raise ValueError("No calibration images found.")

        if square_size is None or square_size <= 0:
            raise ValueError("square_size must be greater than 0.")

        self.square_size = square_size
        self.chessboard_size = (
            chessboard_size
            if chessboard_size is not None
            else GridDetector(self.images[0]).dimensions
        )

        self.objpoints = []
        self.imgpoints = []

        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = None

    def calibrate(self):
        width, height = self.chessboard_size
        image_size = self.images[0].shape[1::-1]

        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * self.square_size

        detector = ChessboardDetector(self.chessboard_size)

        self.objpoints.clear()
        self.imgpoints.clear()

        for image in self.images:
            if image.shape[1::-1] != image_size:
                raise ValueError(
                    "All calibration images must have the same resolution."
                )

            found, corners = detector.detect_chessboard(image)

            if found:
                self.objpoints.append(objp.copy())
                self.imgpoints.append(corners)

        if len(self.objpoints) < 3:
            raise RuntimeError(
                f"Only {len(self.objpoints)} usable calibration images found."
            )

        (
            self.rms_error,
            self.camera_matrix,
            self.dist_coeffs,
            self.rvecs,
            self.tvecs,
        ) = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            image_size,
            None,
            None,
        )

        return {
            "rms_error": self.rms_error,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "rvecs": self.rvecs,
            "tvecs": self.tvecs,
            "successful_images": len(self.objpoints),
        }

    def save_calibration(self, filename):
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Camera has not been calibrated yet.")
        
        if not filename.endswith(".npz"):
            raise ValueError("Filename must have a .npz extension.")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        np.savez(
            filename,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            rvecs=self.rvecs,
            tvecs=self.tvecs,
            rms_error=self.rms_error,
        )

    def reprojection_error(self):
        if self.camera_matrix is None:
            raise RuntimeError("Camera has not been calibrated yet.")

        errors = []

        for objpoints, imgpoints, rvec, tvec in zip(
            self.objpoints,
            self.imgpoints,
            self.rvecs,
            self.tvecs,
        ):
            projected, _ = cv2.projectPoints(
                objpoints,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )

            error = cv2.norm(
                imgpoints,
                projected,
                cv2.NORM_L2,
            ) / len(projected)

            errors.append(error)

        return {
            "mean": float(np.mean(errors)),
            "per_image": errors,
        }
