import cv2

from src.utils import draw_features, load_image, save_image


class ShiTomasi:
    """Detect corner features with the Shi-Tomasi criterion."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)
        self.keypoints = []

    def detect_features(self, max_corners=100, quality_level=0.01, min_distance=10):
        gray = cv2.GaussianBlur(self.gray_image, (5, 5), 1.4)
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        self.keypoints = [] if corners is None else [
            cv2.KeyPoint(float(x), float(y), 3) for x, y in corners.reshape(-1, 2)
        ]
        return self.keypoints

    def detect_corners(self, max_corners=100, quality_level=0.01, min_distance=10):
        return self.detect_features(max_corners, quality_level, min_distance)

    def draw_features(self):
        self.image = draw_features(self.image, self.keypoints, color=(0, 255, 0))
        return self.image

    def save_result(self, output_path):
        self.draw_features()
        save_image(output_path, self.image)
