import cv2

class MedianFilter:
    def __init__(
        self,
        kernel_size_x=3,
        kernel_size_y=3,
    ):
        self.kernel_size = (kernel_size_x, kernel_size_y)   
        
    def apply_filter(self, image):
        # Apply median filter to the image
        return cv2.medianBlur(image, self.kernel_size[0]) 