import cv2
import numpy as np


class HStackImages:
    def __init__(self, images=None):
        self.images = images
        if self.images is not None:
            self.stacked_image = self.stack_images()
        else:
            self.stacked_image = None

    def stack_images(self, images=None):
        if images is not None:
            self.images = images
        max_height = max(img.shape[0] for img in self.images)

        padded_images = []

        for img in self.images:
            pad_height = max_height - img.shape[0]

            padded = cv2.copyMakeBorder(
                img,
                0, pad_height,
                0, 0,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            padded_images.append(padded)

        return np.hstack(padded_images)