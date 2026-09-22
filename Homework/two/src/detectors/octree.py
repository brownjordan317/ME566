"""Octree-style spatial feature detector.

The detector first finds FAST keypoint candidates, then recursively partitions
the image into spatial cells and keeps the strongest candidate from each leaf.
This is the same type of spatial keypoint distribution used by ORB.
"""

import cv2

from src.utils import draw_features, load_image, save_image


class Octree:
    """Detect spatially well-distributed FAST features with an octree layout."""

    def __init__(self, image_path):
        self.image, self.gray_image = load_image(image_path)
        self.keypoints = []

    @staticmethod
    def _split_cell(cell):
        """Split one image cell into four non-empty child cells."""
        x0, y0, x1, y1, keypoints = cell
        x_mid, y_mid = (x0 + x1) / 2, (y0 + y1) / 2
        children = [([], x0, y0, x_mid, y_mid), ([], x_mid, y0, x1, y_mid),
                    ([], x0, y_mid, x_mid, y1), ([], x_mid, y_mid, x1, y1)]

        for keypoint in keypoints:
            index = (1 if keypoint.pt[0] >= x_mid else 0) + (
                2 if keypoint.pt[1] >= y_mid else 0
            )
            children[index][0].append(keypoint)

        return [
            (child_x0, child_y0, child_x1, child_y1, child_keypoints)
            for child_keypoints, child_x0, child_y0, child_x1, child_y1 in children
            if child_keypoints
        ]

    def detect_features(self, max_features=500, threshold=30):
        """Return up to ``max_features`` spatially distributed FAST keypoints."""
        candidates = cv2.FastFeatureDetector_create(threshold).detect(
            self.gray_image, None
        )
        if not candidates:
            self.keypoints = []
            return self.keypoints

        height, width = self.gray_image.shape
        cells = [(0, 0, width, height, candidates)]
        while len(cells) < max_features:
            options = []
            for cell in cells:
                if len(cell[4]) <= 1:
                    continue
                children = self._split_cell(cell)
                if len(children) > 1 and len(cells) - 1 + len(children) <= max_features:
                    options.append((cell, children))
            if not options:
                break
            cell, children = max(options, key=lambda item: len(item[0][4]))
            cells.remove(cell)
            cells.extend(children)

        self.keypoints = [
            max(cell[4], key=lambda keypoint: keypoint.response) for cell in cells
        ]
        return self.keypoints

    def draw_features(self):
        self.image = draw_features(self.image, self.keypoints, color=(0, 255, 255))
        return self.image

    def save_result(self, output_path):
        self.draw_features()
        save_image(output_path, self.image)
