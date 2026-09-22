import cv2

from src.utils import (
    draw_matches,
    load_image,
    match_binary_descriptors,
    require_descriptors,
    save_image,
)


class Brief:
    """An end-to-end FAST + BRIEF feature matcher."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)

    @staticmethod
    def _detect_and_describe(gray_image, threshold=30, keypoints=None):
        if keypoints is None:
            fast = cv2.FastFeatureDetector_create(threshold)
            keypoints = fast.detect(gray_image, None)
        if not hasattr(cv2, "xfeatures2d"):
            raise RuntimeError("BRIEF requires the opencv-contrib-python package.")
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        return brief.compute(gray_image, keypoints)

    def match(self, other_image_path, max_matches=10, threshold=30, keypoints1=None, keypoints2=None):
        """Detect, describe, and match features in this image and another image."""
        other_image, other_gray = load_image(other_image_path)
        keypoints1, descriptors1 = self._detect_and_describe(self.gray_image, threshold, keypoints1)
        keypoints2, descriptors2 = self._detect_and_describe(other_gray, threshold, keypoints2)
        require_descriptors(descriptors1, descriptors2, "BRIEF")
        self.keypoints1, self.keypoints2 = keypoints1, keypoints2
        self.descriptors1, self.descriptors2 = descriptors1, descriptors2

        matches = match_binary_descriptors(descriptors1, descriptors2)
        self.matches = matches
        self.matched_image = draw_matches(
            self.image, keypoints1, other_image, keypoints2, matches[:max_matches]
        )
        return self.matched_image

    def match_descriptors(self, other_image_path):
        """Backward-compatible alias for :meth:`match`."""
        return self.match(other_image_path)

    def save_result(self, output_path):
        save_image(output_path, getattr(self, "matched_image", self.image))
