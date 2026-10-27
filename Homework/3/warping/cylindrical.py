"""Cylindrical projection and rotation/translation/scale alignment."""
import cv2
import numpy as np
from warping.warp_to_match import WarpToMatch


class CylindricalProjector:
    def __init__(self, camera_matrix):
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError('Expected a valid camera matrix with positive focal lengths.')

    def project(self, image, mask=None):
        h, w = image.shape[:2]
        fx, cx, cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
        y, x = np.indices((h, w), dtype=np.float32)
        theta = (x - cx) / fx
        map_x = (fx * np.tan(theta) + cx).astype(np.float32)
        map_y = ((y - cy) / np.cos(theta) + cy).astype(np.float32)
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8)
        valid = cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
        projected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return projected, valid.astype(bool)


class CylindricalWarper(WarpToMatch):
    """Similarity alignment avoids introducing projective stretching again."""
    def estimate_homography(self, keypoints1, keypoints2, matches):
        # Reuse input validation and reject degenerate correspondences.
        super().estimate_homography(keypoints1, keypoints2, matches)
        destination = np.asarray(keypoints1)[matches[:, 0]]
        source = np.asarray(keypoints2)[matches[:, 1]]
        affine, mask = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        if affine is None or mask is None or mask.sum() < 4 or not np.isfinite(affine).all():
            raise ValueError('Not enough consistent cylindrical matches.')
        self.homography = np.vstack([affine, [0., 0., 1.]])
        self.inlier_mask = mask.ravel().astype(bool)
        return self.homography
