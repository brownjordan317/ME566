"""Quality-scored candidates using projected image footprints."""
import cv2
import numpy as np
from warping.geometry import validate_warp


class OverlapFilter:
    def __init__(self, warper, max_overlap=.8, min_overlap=.15):
        if not 0 < min_overlap < max_overlap < 1:
            raise ValueError('Require 0 < min_overlap < max_overlap < 1.')
        self.warper = warper
        self.max_overlap = max_overlap
        self.min_overlap = min_overlap
        self.transform = np.eye(3)
        self.footprints = []

    @staticmethod
    def footprint(image, transform):
        h, w = image.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ transform.T
        z = corners[:, 2]
        if not (np.all(z > 1e-9) or np.all(z < -1e-9)):
            raise ValueError('Footprint crosses projective infinity')
        polygon = (corners[:, :2] / z[:, None]).astype(np.float32)
        if not np.isfinite(polygon).all() or abs(cv2.contourArea(polygon)) < 1:
            raise ValueError('Invalid projected footprint')
        return polygon

    @staticmethod
    def overlap(a, b):
        # Clip in float64: OpenCV's float32 intersection can return zero for
        # almost coincident footprints (e.g. a return to an earlier viewpoint).
        polygon = [point.astype(np.float64) for point in a]
        clip = np.asarray(b, dtype=np.float64)
        orientation = 1 if cv2.contourArea(b, oriented=True) >= 0 else -1
        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]
        for start, end in zip(clip, np.roll(clip, -1, axis=0)):
            if not polygon:
                return 0.
            output = []
            previous = polygon[-1]
            previous_distance = orientation * cross(end - start, previous - start)
            for point in polygon:
                distance = orientation * cross(end - start, point - start)
                if (distance >= 0) != (previous_distance >= 0):
                    t = previous_distance / (previous_distance - distance)
                    output.append(previous + t * (point - previous))
                if distance >= 0:
                    output.append(point)
                previous, previous_distance = point, distance
            polygon = output
        if len(polygon) < 3:
            return 0.
        points = np.asarray(polygon)
        intersection = abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                                  - points[:, 1] * np.roll(points[:, 0], -1))) / 2
        return float(np.clip(intersection / min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b))), 0, 1))

    def start(self, feature):
        self.footprints = [self.footprint(feature.image, np.eye(3))]
        self.transform = np.eye(3)

    def evaluate(self, first, second, matches):
        """Measure a candidate without modifying the retained-frame state."""
        result = {'accepted': False, 'reason': '', 'overlap': None, 'score': 0.}
        try:
            H = self.warper.estimate_homography(first.keypoints, second.keypoints, matches)
            validate_warp(second.image, H)
            inliers = self.warper.inlier_mask
            count = int(inliers.sum())
            ratio = count / max(len(matches), 1)
            result.update(inliers=count, inlier_ratio=ratio)
            if count < 12 or ratio < .25:
                raise ValueError('need at least 12 inliers and 25% inlier ratio')
            destination = first.keypoints[matches[inliers, 0]]
            source = second.keypoints[matches[inliers, 1]]
            projected = cv2.perspectiveTransform(source.astype(np.float64)[None], H)[0]
            residual = float(np.sqrt(np.mean(np.sum((projected - destination)**2, axis=1))))
            coverage = min(
                abs(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
                / (feature.image.shape[0] * feature.image.shape[1])
                for points, feature in ((destination, first), (source, second))
            )
            result.update(rmse=residual, coverage=coverage)
            if not np.isfinite(residual) or residual > self.warper.ransac_threshold:
                raise ValueError('large reprojection error')
            if coverage < .01:
                raise ValueError('inliers are concentrated in too small an image region')
            transform = self.transform @ H
            footprint = self.footprint(second.image, transform)
            overlaps = [self.overlap(previous, footprint) for previous in self.footprints]
            maximum = max(overlaps)
            result['overlap'] = maximum
            if maximum > self.max_overlap:
                result['reason'] = 'too much overlap'
                return result
            if overlaps[-1] < self.min_overlap:
                result['reason'] = 'too little overlap with last retained image'
                return result
            gray = second.image
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            # A modest sharpness preference; geometric evidence dominates.
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            target = np.clip(.60, self.min_overlap, self.max_overlap)
            spacing = 1 - abs(overlaps[-1] - target)
            score = (2 * ratio + min(count / 100, 1) + min(coverage / .2, 1)
                     + spacing + .15 * min(np.log1p(sharpness) / 8, 1)
                     - residual / max(self.warper.ransac_threshold, 1e-6))
            result.update(accepted=True, reason='eligible', score=float(score),
                          sharpness=sharpness, transform=transform, footprint=footprint)
        except (ValueError, cv2.error) as exc:
            result['reason'] = f'unreliable alignment: {exc}'
        return result

    def commit(self, candidate):
        if not candidate['accepted']:
            raise ValueError('Cannot retain a rejected candidate.')
        self.transform = candidate['transform']
        self.footprints.append(candidate['footprint'])

    def consider(self, first, second, matches):
        """Compatibility helper for immediate selection without lookahead."""
        candidate = self.evaluate(first, second, matches)
        if candidate['accepted']:
            self.commit(candidate)
        return candidate['accepted'], candidate['reason'], candidate['overlap']
