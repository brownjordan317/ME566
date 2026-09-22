import cv2

from src.utils import draw_matches, filter_ratio_matches, load_image, require_descriptors, save_image


class SIFT:
    """An end-to-end SIFT feature matcher using FLANN and Lowe's ratio test."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)

    def match(self, other_image_path, ratio=0.7, keypoints1=None, keypoints2=None):
        """Detect, describe, and match SIFT features in both images."""
        other_image, other_gray = load_image(other_image_path)

        sift = cv2.SIFT_create()
        if keypoints1 is None:
            keypoints1, descriptors1 = sift.detectAndCompute(self.gray_image, None)
        else:
            keypoints1, descriptors1 = sift.compute(self.gray_image, keypoints1)
        if keypoints2 is None:
            keypoints2, descriptors2 = sift.detectAndCompute(other_gray, None)
        else:
            keypoints2, descriptors2 = sift.compute(other_gray, keypoints2)
        require_descriptors(descriptors1, descriptors2, "SIFT")
        self.keypoints1, self.keypoints2 = keypoints1, keypoints2
        self.descriptors1, self.descriptors2 = descriptors1, descriptors2

        index_params = {"algorithm": 1, "trees": 5}  # Using KDTree for SIFT
        search_params = {"checks": 50}
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = filter_ratio_matches(matches, ratio)

        self.matches = good_matches
        self.matched_image = draw_matches(
            self.image, keypoints1, other_image, keypoints2, good_matches
        )
        return self.matched_image

    def match_keypoints(self, other_image_path):
        """Backward-compatible alias for :meth:`match`."""
        return self.match(other_image_path)

    def save_result(self, output_path):
        save_image(output_path, getattr(self, "matched_image", self.image))
