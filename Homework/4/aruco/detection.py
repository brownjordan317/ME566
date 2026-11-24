"""Detect square markers and estimate marker-to-camera transforms."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MarkerPose:
    marker_id: int
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    transform: np.ndarray
    reprojection_rmse: float


class MarkerDetector:
    def __init__(self, matrix, distortion, marker_length,
                 dictionary='DICT_4X4_50', max_error=3.0):
        if not np.isfinite(marker_length) or marker_length <= 0:
            raise ValueError('Marker length must be positive and finite.')
        if not np.isfinite(max_error) or max_error <= 0:
            raise ValueError('Maximum reprojection error must be positive and finite.')
        if not dictionary.startswith('DICT_') or not hasattr(cv2.aruco, dictionary):
            raise ValueError(f'Unknown ArUco dictionary: {dictionary}')
        self.matrix, self.distortion = matrix, distortion
        self.marker_length, self.max_error = marker_length, max_error
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)
        half = marker_length / 2
        # IPPE_SQUARE order: top left, top right, bottom right, bottom left.
        self.object_points = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0]], dtype=np.float64)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = {}
        if ids is None:
            return corners, ids, poses
        unique, counts = np.unique(ids, return_counts=True)
        duplicates = set(unique[counts > 1].tolist())
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id in duplicates:
                continue  # Repeated IDs cannot identify a stationary reference.
            points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            result = cv2.solvePnPGeneric(self.object_points, points, self.matrix,
                                        self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            solutions = list(zip(result[1], result[2]))
            # IPPE can become numerically singular for a front-facing square.
            # Include an iterative solution and apply the same depth/error checks.
            success, rvec, tvec = cv2.solvePnP(
                self.object_points, points, self.matrix, self.distortion,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                solutions.append((rvec, tvec))
            candidates = []
            for rvec, tvec in solutions:
                if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                    continue
                rotation = cv2.Rodrigues(rvec)[0]
                camera_points = (rotation @ self.object_points.T).T + tvec.reshape(3)
                if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
                    continue
                projected = cv2.projectPoints(self.object_points, rvec, tvec,
                                             self.matrix, self.distortion)[0].reshape(4, 2)
                error = float(np.sqrt(np.mean(np.sum((projected - points)**2, axis=1))))
                if not np.isfinite(error) or error > self.max_error:
                    continue
                transform = np.eye(4)
                transform[:3, :3], transform[:3, 3] = rotation, tvec.reshape(3)
                candidates.append(MarkerPose(marker_id, points, rvec, tvec, transform, error))
            if candidates:
                poses[marker_id] = min(candidates, key=lambda pose: pose.reprojection_rmse)
        return corners, ids, poses
