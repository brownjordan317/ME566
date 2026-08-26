import cv2

class GaussianFilter:
    def __init__(
        self, 
        kernel_size_x=5,
        kernel_size_y=5, 
        sigma=1.0
    ):
        self.kernel_size = (kernel_size_x, kernel_size_y)
        self.sigma = sigma

    def apply_filter(self, image):
        # Apply Gaussian filter to the image
        return cv2.GaussianBlur(image, self.kernel_size, self.sigma)