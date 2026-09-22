import cv2

from src.utils import draw_features, load_image, save_image


class Fast:
    """Detect image features with FAST; this class does not perform matching."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)
        self.keypoints = []

    def detect_features(self, threshold=30):
        detector = cv2.FastFeatureDetector_create(threshold)
        self.keypoints = detector.detect(self.gray_image, None)
        return self.keypoints

    def detect_corners(self, threshold=30):
        return self.detect_features(threshold)

    def draw_features(self):
        self.image = draw_features(self.image, self.keypoints, color=(255, 0, 0))
        return self.image

    def save_result(self, output_path):
        self.draw_features()
        save_image(output_path, self.image)
