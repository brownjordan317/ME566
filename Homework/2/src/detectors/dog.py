# DoG == Difference of Gaussians
import cv2
import numpy as np

from src.utils import draw_features, load_image, save_image


class DoG:
    """Detect blob-like image features from Difference-of-Gaussians extrema."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)
        self.keypoints = []

    def detect_features(self, sigma1=1.0, sigma2=2.0, threshold=0.03):
        """Return keypoints at local extrema in the DoG response image."""
        gray = self.gray_image.astype(np.float32) / 255.0
        dog = cv2.GaussianBlur(gray, (0, 0), sigma1) - cv2.GaussianBlur(
            gray, (0, 0), sigma2
        )
        response = np.abs(dog)
        maxima = response == cv2.dilate(response, np.ones((3, 3), np.uint8))
        points = np.argwhere(maxima & (response >= threshold))
        self.keypoints = [
            cv2.KeyPoint(float(x), float(y), float(max(sigma2 * 2, 1)))
            for y, x in points
        ]
        return self.keypoints

    def detect_edges(self, sigma1=1.0, sigma2=2.0):
        """Backward-compatible alias for feature detection."""
        return self.detect_features(sigma1, sigma2)

    def draw_features(self):
        self.image = draw_features(self.image, self.keypoints, color=(0, 0, 255))
        return self.image

    def save_result(self, output_path):
        self.draw_features()
        save_image(output_path, self.image)
