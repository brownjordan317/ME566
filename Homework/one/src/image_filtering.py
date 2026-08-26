import cv2
from scipy.ndimage import rank_filter


class ImageFiltering:
    def __init__(
        self,
        filter_type="gaussian",
        kernel_size_x=5,
        kernel_size_y=5,
        sigma=1.0,
        rank=1,
    ):
        self.filter_type = filter_type
        self.kernel_size = (kernel_size_x, kernel_size_y)
        self.sigma = sigma
        self.rank = rank

        if self.filter_type not in ["gaussian", "median", "rank"]:
            raise ValueError(
                "Invalid filter type. Choose from 'gaussian', 'median', or 'rank'."
            )

        if self.filter_type == "gaussian":
            self.filtered_image = self.apply_gaussian_filter
        elif self.filter_type == "median":
            self.filtered_image = self.apply_median_filter
        else:
            self.filtered_image = self.apply_rank_filter

    def apply_gaussian_filter(self, image):
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError("Kernel size must be odd for Gaussian filter.")

        return cv2.GaussianBlur(
            image,
            self.kernel_size,
            self.sigma,
        )

    def apply_median_filter(self, image):
        if self.kernel_size[0] != self.kernel_size[1]:
            raise ValueError("Median filter requires a square kernel.")

        return cv2.medianBlur(
            image,
            self.kernel_size[0],
        )

    def apply_rank_filter(self, image):
        return rank_filter(
            image,
            rank=self.rank,
            size=(self.kernel_size[1], self.kernel_size[0], 1),
        )