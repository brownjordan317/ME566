import cv2
import numpy as np


class VStackImages:
    def __init__(self, images=None):
        self.images = images
        if self.images is not None:
            self.stacked_image = self.stack_images()
        else:
            self.stacked_image = None

    def stack_images(self):
        max_width = max(img.shape[1] for img in self.images)

        padded_images = []

        for img in self.images:
            pad_width = max_width - img.shape[1]

            padded = cv2.copyMakeBorder(
                img,
                0, 0,
                0, pad_width,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            padded_images.append(padded)

        return np.vstack(padded_images)