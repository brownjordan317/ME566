import cv2


class Thresholder:
    def __init__(
        self,
        image=None,
        threshold_type="adaptive",
    ):
        self.image = image
        self.threshold_type = threshold_type

    def run_thresholding(self):
        if self.image is None:
            raise ValueError("No image provided.")

        image = self.image

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.threshold_type == "adaptive":
            thresholded = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )

        elif self.threshold_type == "simple":
            _, thresholded = cv2.threshold(
                image,
                127,
                255,
                cv2.THRESH_BINARY,
            )

        else:
            raise ValueError(
                "threshold_type must be 'adaptive' or 'simple'."
            )

        return thresholded