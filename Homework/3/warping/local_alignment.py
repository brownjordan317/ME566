"""Optional bounded optical-flow refinement in the shared image area."""
import cv2
import numpy as np


class LocalAlignment:
    """Experimental local correction; does not resolve occlusions or moving objects."""
    def __init__(self, max_displacement=20.0):
        self.max_displacement = float(max_displacement)
        if not np.isfinite(self.max_displacement) or self.max_displacement <= 0:
            raise ValueError('max_displacement must be finite and positive.')

    def refine(self, reference, moving, mask1, mask2):
        overlap = ((mask1 != 0) & (mask2 != 0)).astype(np.uint8)
        if overlap.sum() < 1024:
            return moving, mask2
        h, w = overlap.shape
        scale = min(1., 800. / max(h, w))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        def gray(image):
            image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image * overlap
            return cv2.resize(image, size)
        flow = cv2.calcOpticalFlowFarneback(gray(reference), gray(moving), None,
                                          .5, 3, 25, 4, 7, 1.5, 0)
        flow = cv2.resize(flow, (w, h))
        flow[..., 0] *= w / size[0]
        flow[..., 1] *= h / size[1]
        # Fade at coverage edges; never extrapolate flow outside overlap.
        fade = np.minimum(cv2.distanceTransform(overlap, cv2.DIST_L2, 3) / 32., 1.)
        magnitude = np.linalg.norm(flow, axis=2)
        flow *= np.minimum(1., self.max_displacement / np.maximum(magnitude, 1e-6))[..., None]
        flow *= fade[..., None]
        y, x = np.indices((h, w), dtype=np.float32)
        map_x, map_y = x + flow[..., 0], y + flow[..., 1]
        corrected = cv2.remap(moving, map_x, map_y, cv2.INTER_LINEAR)
        valid = cv2.remap(mask2.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
        return corrected, valid
