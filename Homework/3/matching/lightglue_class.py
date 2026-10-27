"""Match SuperPoint descriptors with pretrained LightGlue."""

import numpy as np
import torch
from lightglue import LightGlue


class LightGlueMatcher:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.lightglue = LightGlue(features='superpoint').eval().to(self.device)

    def _features(self, keypoints, descriptors, image_size):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError('Keypoints must have shape (N, 2).')
        if descriptors.shape != (len(keypoints), 256):
            raise ValueError('SuperPoint descriptors must have shape (N, 256).')
        if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all():
            raise ValueError('Features must contain finite values.')
        features = {
            'keypoints': torch.from_numpy(np.ascontiguousarray(keypoints))[None].to(self.device),
            'descriptors': torch.from_numpy(np.ascontiguousarray(descriptors))[None].to(self.device),
        }
        if image_size is not None:
            size = np.asarray(image_size, dtype=np.float32)
            if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError('Image size must be (width, height), both positive.')
            features['image_size'] = torch.from_numpy(size)[None].to(self.device)
        return features

    @torch.inference_mode()
    def match(self, keypoints1, descriptors1, keypoints2, descriptors2,
              image_size1=None, image_size2=None):
        """Return an (M, 2) integer array indexing the two keypoint arrays.

        Optional sizes are (width, height); otherwise LightGlue estimates the
        coordinate normalization from the keypoints. Confidence is in scores.
        """
        features1 = self._features(keypoints1, descriptors1, image_size1)
        features2 = self._features(keypoints2, descriptors2, image_size2)
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            self.matches = np.empty((0, 2), dtype=np.int64)
            self.scores = np.empty((0,), dtype=np.float32)
        else:
            result = self.lightglue({'image0': features1, 'image1': features2})
            self.matches = result['matches'][0].cpu().numpy()
            self.scores = result['scores'][0].cpu().numpy()
        return self.matches
