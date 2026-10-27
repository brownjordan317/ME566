import cv2
from scipy.ndimage import rank_filter


class ImageFiltering:
    def __init__(
        self, 
        filter_type="gaussian", # "gaussian", "median", or "rank"
        kernel_size=(5, 5), 
        sigma=1.0, 
        rank=1
        ):
        self.filter_type = filter_type
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.rank = rank

    def apply(self, image):
        kx, ky = self.kernel_size

        if self.filter_type == "gaussian":
            return cv2.GaussianBlur(image, (kx, ky), self.sigma)

        if self.filter_type == "median":
            return cv2.medianBlur(image, kx)

        if self.filter_type == "rank":
            return rank_filter(image, rank=self.rank, size=(ky, kx, 1))

        raise ValueError(
            "filter_type must be 'gaussian', 'median', or 'rank'."
        )
