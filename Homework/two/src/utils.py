"""Shared image I/O and visualization helpers for feature detectors and matchers."""

import cv2
import numpy as np


from src.io.image_reader import ImageReader


def load_image(image_path):
    """Load an image and its grayscale counterpart."""
    if isinstance(image_path, np.ndarray):
        image = image_path.copy()
    else:
        images = ImageReader(str(image_path)).images
        if len(images) != 1:
            raise ValueError(f"Could not read image: {image_path}")
        image = images[0]
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_features(image, keypoints, color):
    """Draw detected keypoints without changing the source image in place."""
    return cv2.drawKeypoints(image, keypoints, None, color=color)


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    """Create a visualization of matched feature pairs."""
    return cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def save_image(output_path, image):
    """Write an image and report an unsuccessful write as an error."""
    if not cv2.imwrite(output_path, image):
        raise ValueError(f"Could not write image: {output_path}")


def require_descriptors(descriptors1, descriptors2, algorithm):
    """Ensure that both images supplied usable descriptors for a matcher."""
    if descriptors1 is None or descriptors2 is None:
        raise ValueError(f"{algorithm} could not compute descriptors in one of the images.")


def match_binary_descriptors(descriptors1, descriptors2):
    """Match binary descriptors with Hamming distance, best matches first."""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return sorted(matcher.match(descriptors1, descriptors2), key=lambda match: match.distance)


def filter_ratio_matches(knn_matches, ratio):
    """Keep FLANN k-NN matches that pass Lowe's ratio test."""
    return [
        first
        for pair in knn_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]
