"""SuperPoint features for OpenCV images, using the LightGlue package."""

import cv2
import numpy as np
import torch
from lightglue import SuperPoint


class SuperPointObject:
    """Extract (N, 2) pixel coordinates and (N, 256) descriptors.

    Images are OpenCV BGR/BGRA or grayscale uint8 arrays, or float arrays
    normalized to [0, 1]. Coordinates are in the original image size.
    """

    def __init__(self, image, device='cpu', max_num_keypoints=2048):
        self.image = image
        self.device = torch.device(device)
        self.superpoint = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.keypoints, self.descriptors = self.detect_keypoints_and_descriptors()

    @torch.inference_mode()
    def detect_keypoints_and_descriptors(self):
        image = np.asarray(self.image)
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 8:
            raise ValueError('Expected an image with height and width at least 8 pixels.')
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Float images must contain finite values in [0, 1].')
            image = image.astype(np.float32)
        else:
            raise ValueError('Images must be uint8 or floating point in [0, 1].')
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[..., 0]
            elif image.shape[2] in (3, 4):
                code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
                image = cv2.cvtColor(image, code)
            else:
                raise ValueError('Expected 1, 3, or 4 image channels.')
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None].to(self.device)
        self.features = self.superpoint.extract(tensor, resize=None)
        self.image_size = np.array(image.shape[:2][::-1], dtype=np.float32)
        self.scores = self.features['keypoint_scores'][0].cpu().numpy()
        self.keypoints = self.features['keypoints'][0].cpu().numpy()
        self.descriptors = self.features['descriptors'][0].cpu().numpy()
        return self.keypoints, self.descriptors
