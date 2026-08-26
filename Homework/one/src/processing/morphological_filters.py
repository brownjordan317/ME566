import cv2
import numpy as np


class MorphologicalFilter:
    def __init__(
        self, 
        operation="opening", 
        kernel_size=5, 
        iterations=1
    ):
        self.operation = operation.lower()
        self.kernel_size = kernel_size
        self.iterations = iterations

        valid_operations = [
            "erosion",
            "dilation",
            "opening",
            "closing",
            "gradient",
            "tophat",
            "blackhat",
        ]

        if self.operation not in valid_operations:
            raise ValueError(f"operation must be one of: {valid_operations}")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")

        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    def apply(self, image):
        if self.operation == "erosion":
            return cv2.erode(image, self.kernel, iterations=self.iterations)

        if self.operation == "dilation":
            return cv2.dilate(image, self.kernel, iterations=self.iterations)

        operations = {
            "opening": cv2.MORPH_OPEN,
            "closing": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }

        return cv2.morphologyEx(
            image,
            operations[self.operation],
            self.kernel,
            iterations=self.iterations,
        )