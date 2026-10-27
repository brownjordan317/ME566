"""Custom two-image alignment and narrow-seam blending; no OpenCV Stitcher.

Image 2 is mapped into image 1's coordinate system. A single homography is
appropriate for planar scenes or camera rotation, not large scene parallax.
"""

import cv2
import numpy as np
from warping.geometry import canvas_bounds, image_corners, validate_warp


def mask_weights(mask):
    padded = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]


def estimate_homography(keypoints1, keypoints2, matches, ransac_threshold=3.0):
    """Return (H mapping image 2 to image 1, RANSAC inlier mask)."""
    points = [np.asarray(k, dtype=np.float64) for k in (keypoints1, keypoints2)]
    for k in points:
        if k.ndim != 2 or k.shape[1] != 2 or not np.isfinite(k).all():
            raise ValueError('Keypoints must be finite (N, 2) arrays.')
    matches = np.asarray(matches)
    if (matches.ndim != 2 or matches.shape[1] != 2
            or not np.issubdtype(matches.dtype, np.integer)):
        raise ValueError('Matches must be an integer (M, 2) array.')
    if len(matches) < 4:
        raise ValueError('At least four matches are required for a homography.')
    for column, k in enumerate(points):
        if np.any(matches[:, column] < 0) or np.any(matches[:, column] >= len(k)):
            raise ValueError('Match index outside its keypoint array.')
    destination, source = [k[matches[:, i]] for i, k in enumerate(points)]
    for k in (source, destination):
        if len(np.unique(k, axis=0)) < 4 or np.linalg.matrix_rank(k - k.mean(axis=0)) < 2:
            raise ValueError('Matches need four distinct, non-collinear points in each image.')
    homography, mask = cv2.findHomography(
        source, destination, cv2.RANSAC, ransac_threshold,
    )
    if (homography is None or not np.isfinite(homography).all()
            or np.linalg.matrix_rank(homography) < 3
            or mask is None or np.count_nonzero(mask) < 4):
        raise ValueError('Could not estimate a valid homography from the matches.')
    return homography, mask.ravel().astype(bool)


def blend_images(first, second, mask1, mask2, blend_mode="seam"):
    """Prefer the source farther from its border; soften only a 5px seam.

    Black pixels are valid data. The optional feather mode retains the old
    whole-overlap average for comparison. Neither method fixes parallax.
    """
    valid1, valid2 = mask1 != 0, mask2 != 0
    weights1 = mask_weights(valid1.astype(np.uint8))
    weights2 = mask_weights(valid2.astype(np.uint8))
    if blend_mode == 'feather':
        total = weights1 + weights2
        alpha = weights1 / np.maximum(total, 1e-6)
    else:
        # Hard source ownership prevents ghosting over the whole overlap.
        ownership = (weights1 >= weights2).astype(np.float32)
        alpha = cv2.GaussianBlur(ownership, (5, 5), 0.8)
    alpha[valid1 & ~valid2] = 1
    alpha[~valid1] = 0
    beta = (1 - alpha) * valid2
    if first.ndim == 3:
        alpha, beta = alpha[..., None], beta[..., None]
    return first.astype(np.float32) * alpha + second.astype(np.float32) * beta


class WarpToMatch:
    """Estimate alignment from SuperPoint keypoints and LightGlue index pairs."""

    def __init__(self, ransac_threshold=3.0, max_canvas_pixels=25_000_000, local_refiner=None, blend_mode="seam"):
        if not np.isfinite(ransac_threshold) or ransac_threshold <= 0:
            raise ValueError('ransac_threshold must be positive and finite.')
        if not np.isfinite(max_canvas_pixels) or max_canvas_pixels < 1:
            raise ValueError('max_canvas_pixels must be positive.')
        if blend_mode not in ('seam', 'feather'):
            raise ValueError('blend_mode must be seam or feather.')
        self.blend_mode = blend_mode
        self.local_refiner = local_refiner
        self.ransac_threshold = float(ransac_threshold)
        self.max_canvas_pixels = max_canvas_pixels

    def estimate_homography(self, keypoints1, keypoints2, matches):
        self.homography, self.inlier_mask = estimate_homography(
            keypoints1, keypoints2, matches, self.ransac_threshold)
        return self.homography

    def fit_canvas(self, corners):
        """Compatibility wrapper: reject oversized canvases at native resolution."""
        size, transform = canvas_bounds(corners, self.max_canvas_pixels)
        return size, transform, 1.0

    def blend(self, first, second, mask1, mask2):
        return blend_images(first, second, mask1, mask2, self.blend_mode)

    def warp(self, image1, image2, keypoints1, keypoints2, matches, mask1=None, mask2=None):
        """Return a blended panorama with both images' full extents.

        mask1 optionally marks valid pixels in an existing panorama.
        Images must both be uint8 grayscale or both uint8 BGR. Keypoints must
        describe the supplied images (use undistorted images with undistorted
        keypoints). Exposes homography, inlier_mask, transform1, transform2,
        and coverage_mask after a successful call. Transforms map input pixel
        coordinates into the returned panorama.
        """
        for image in (image1, image2):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim not in (2, 3) or min(image.shape[:2]) == 0
                    or (image.ndim == 3 and image.shape[2] != 3)):
                raise ValueError('Images must be nonempty uint8 grayscale or BGR arrays.')
        if image1.ndim != image2.ndim:
            raise ValueError('Both images must have the same number of channels.')
        if mask1 is None:
            mask1 = np.ones(image1.shape[:2], dtype=np.uint8)
        else:
            mask1 = np.asarray(mask1)
            if mask1.shape != image1.shape[:2]:
                raise ValueError('mask1 must match image1 height and width.')
            mask1 = (mask1 != 0).astype(np.uint8)
        if mask2 is None:
            mask2 = np.ones(image2.shape[:2], dtype=np.uint8)
        else:
            mask2 = np.asarray(mask2)
            if mask2.shape != image2.shape[:2]:
                raise ValueError('mask2 must match image2 height and width.')
            mask2 = (mask2 != 0).astype(np.uint8)
        homography = self.estimate_homography(keypoints1, keypoints2, matches)
        corners2 = validate_warp(image2, homography)
        corners = np.vstack((image_corners(image1), corners2))
        size, translation, self.canvas_scale = self.fit_canvas(corners)
        self.transform1 = translation
        self.transform2 = translation @ homography
        warped, masks = [], []
        for image, transform, mask in (
                (image1, self.transform1, mask1),
                (image2, self.transform2, mask2)):
            warped.append(cv2.warpPerspective(image, transform, size).astype(np.float32))
            # Masks represent geometry, not brightness: black pixels remain valid.
            masks.append(cv2.warpPerspective(
                mask, transform, size,
                flags=cv2.INTER_NEAREST,
            ))
        if self.local_refiner is not None:
            warped[1], masks[1] = self.local_refiner.refine(
                warped[0], warped[1], masks[0], masks[1])
        panorama = self.blend(warped[0], warped[1], masks[0], masks[1])
        self.coverage_mask = (masks[0] | masks[1]).astype(bool)
        return np.clip(np.rint(panorama), 0, 255).astype(np.uint8)


def warp_to_match(image1, image2, keypoints1, keypoints2, matches, **kwargs):
    """Convenience wrapper returning the blended image; kwargs configure WarpToMatch."""
    return WarpToMatch(**kwargs).warp(image1, image2, keypoints1, keypoints2, matches)
