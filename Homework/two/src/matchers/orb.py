import cv2

from src.utils import (
    draw_matches,
    load_image,
    match_binary_descriptors,
    require_descriptors,
    save_image,
)


class Orb:
    """An end-to-end ORB feature matcher."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)

    def match(self, other_image_path, max_matches=10, n_features=500, keypoints1=None, keypoints2=None):
        """Detect, describe, and match ORB features in both images."""
        other_image, other_gray = load_image(other_image_path)

        orb = cv2.ORB_create(n_features)
        if keypoints1 is None:
            keypoints1, descriptors1 = orb.detectAndCompute(self.gray_image, None)
        else:
            keypoints1, descriptors1 = orb.compute(self.gray_image, keypoints1)
        if keypoints2 is None:
            keypoints2, descriptors2 = orb.detectAndCompute(other_gray, None)
        else:
            keypoints2, descriptors2 = orb.compute(other_gray, keypoints2)
        require_descriptors(descriptors1, descriptors2, "ORB")
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
