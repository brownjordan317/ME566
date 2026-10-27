"""Small geometry functions shared by selection, warping, and the report."""
import cv2
import numpy as np


class WarpRejected(ValueError):
    """An alignment would create an unsafe or excessively stretched image."""


def image_corners(image):
    height, width = image.shape[:2]
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def project_corners(image, homography):
    points = np.column_stack((image_corners(image), np.ones(4))) @ np.asarray(homography).T
    z = points[:, 2]
    if not np.isfinite(points).all() or not (np.all(z > 1e-9) or np.all(z < -1e-9)):
        raise WarpRejected('Image footprint crosses projective infinity.')
    return points[:, :2] / z[:, None]


def validate_warp(image, homography, max_stretch=4.0):
    """Reject collapsed, flipped, or severely enlarged image footprints.

    max_stretch bounds each edge's length ratio and total area ratio. It is
    intentionally conservative for panoramas, not a general perspective editor.
    """
    corners = project_corners(image, homography)
    source = image_corners(image)
    source_area = cv2.contourArea(source.astype(np.float32), oriented=True)
    area = cv2.contourArea(corners.astype(np.float32), oriented=True)
    if not 1 / max_stretch <= area / source_area <= max_stretch:
        raise WarpRejected(f'Projected area changes by {area / source_area:.2f}x (limit {max_stretch:g}x).')
    edge_ratios = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1) / np.linalg.norm(
        np.roll(source, -1, axis=0) - source, axis=1)
    if np.any(edge_ratios > max_stretch) or np.any(edge_ratios < 1 / max_stretch):
        raise WarpRejected('Projected edges stretch or collapse excessively.')
    return corners


def canvas_bounds(corners, max_canvas_pixels=25_000_000):
    """Return native-resolution canvas size and translation; never shrink a mega warp."""
    points = np.asarray(corners, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise WarpRejected('Canvas requires finite (N, 2) corners.')
    low = np.floor(points.min(axis=0))
    extent = np.ceil(points.max(axis=0)) - low
    if np.any(extent <= 0) or np.any(extent >= 32767):
        raise WarpRejected('Canvas dimensions are invalid or exceed the OpenCV limit.')
    width, height = map(int, extent)
    if width * height > max_canvas_pixels:
        raise WarpRejected(f'Canvas {width} x {height} exceeds {max_canvas_pixels:,} pixels.')
    translation = np.array([[1., 0., -low[0]], [0., 1., -low[1]], [0., 0., 1.]])
    return (width, height), translation


def panorama_layout(images, transforms, max_canvas_pixels=25_000_000):
    """Keep safe image footprints and report rejected ones before allocating pixels."""
    images, transforms = list(images), list(transforms)
    if len(images) != len(transforms):
        raise ValueError('Each image needs a corresponding transform.')
    kept, rejected, corners = [], [], []
    for index, (image, transform) in enumerate(zip(images, transforms, strict=True)):
        try:
            box = validate_warp(image, transform)
            size, translation = canvas_bounds(np.vstack([*corners, box]), max_canvas_pixels)
        except WarpRejected as exc:
            rejected.append({'index': index, 'reason': str(exc)})
            continue
        kept.append(index)
        corners.append(box)
    if not kept:
        raise WarpRejected('No safe image footprints remain for the panorama.')
    return {'size': size, 'transform': translation, 'corners': corners,
            'kept_indices': kept, 'rejected': rejected}
