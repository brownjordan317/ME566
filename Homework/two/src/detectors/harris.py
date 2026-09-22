import cv2

from src.utils import draw_features, load_image, save_image


class Harris:
    """Detect corner features with the Harris corner response."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)
        self.keypoints = []

    def detect_features(self, block_size=2, ksize=3, k=0.04, threshold=0.01):
        gray = cv2.GaussianBlur(self.gray_image, (5, 5), 1.4).astype("float32")
        response = cv2.cornerHarris(gray, block_size, ksize, k)
        maxima = response == cv2.dilate(response, None)
        points = cv2.findNonZero((maxima & (response > threshold * response.max())).astype("uint8"))
        self.keypoints = [] if points is None else [
            cv2.KeyPoint(float(x), float(y), float(block_size)) for x, y in points.reshape(-1, 2)
        ]
        return self.keypoints

    def detect_corners(self, block_size=2, ksize=3, k=0.04):
        return self.detect_features(block_size, ksize, k)

    def draw_features(self):
        self.image = draw_features(self.image, self.keypoints, color=(0, 0, 255))
        return self.image

    def save_result(self, output_path):
        self.draw_features()
        save_image(output_path, self.image)
