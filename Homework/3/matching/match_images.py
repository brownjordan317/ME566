"""Match consecutive images: python Homework/3/matching/match_images.py path/to/folder"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

HW3_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(HW3_DIR))

import itertools

from matching.lightglue_class import LightGlueMatcher
from matching.overlap_filter import OverlapFilter
from matching.superpoint_class import SuperPointObject
from warping.cam_undistort import CameraUndistorter
from warping.warp_to_match import WarpToMatch

# Python's built-in io module shadows the homework's io directory.
_spec = importlib.util.spec_from_file_location(
    'hw3_image_reader', HW3_DIR / 'io' / 'image_reader.py'
)
_reader_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader_module)
ImageReader = _reader_module.ImageReader


class ImageMatcher:
    """Undistort, match consecutive folder images, and save visualizations."""

    def __init__(self, input_folder, device='cpu', max_keypoints=2048,
                 calibration=None, projector=None, max_overlap=None,
                 min_overlap=.15, min_images=4, selection_lookahead=3):
        self.input_folder = Path(input_folder).resolve()
        if not isinstance(selection_lookahead, int) or selection_lookahead < 0:
            raise ValueError('selection_lookahead must be a nonnegative integer.')
        self.selection_lookahead = selection_lookahead
        self.max_overlap = max_overlap
        self.min_overlap = min_overlap
        self.min_images = min_images
        self.selection_warper = None
        self.selection_log = []
        self.selected_indices = []
        self.projector = projector
        self.device = device
        self.max_keypoints = max_keypoints
        self.calibration = (Path(calibration) if calibration is not None
                            else HW3_DIR / 'camera_cal.json')
        if not self.input_folder.is_dir():
            raise ValueError(f'Not an image folder: {self.input_folder}')
        if max_keypoints <= 0:
            raise ValueError('max_keypoints must be positive')
        self.undistorter = self._load_calibration()
        self.reader = ImageReader(self.input_folder)
        if len(self.reader.images) < 2:
            raise ValueError(f'Need at least two readable images in {self.input_folder}')
        self.output_dir = HW3_DIR / 'outputs' / 'matches' / self.input_folder.name
        self.matcher = LightGlueMatcher(device)
        self.output_paths = []

    def _load_calibration(self):
        try:
            with self.calibration.open(encoding='utf-8') as stream:
                calibration = json.load(stream)
            matrix = np.asarray(calibration['camera_matrix'], dtype=np.float64)
            distortion = np.asarray(calibration['dist_coeffs'], dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError('camera_matrix must be a finite 3-by-3 matrix')
            if (distortion.ndim not in (1, 2)
                    or (distortion.ndim == 2 and 1 not in distortion.shape)
                    or distortion.size not in (4, 5, 8, 12, 14)
                    or not np.isfinite(distortion).all()):
                raise ValueError('dist_coeffs must be a finite vector of 4, 5, 8, 12, or 14 values')
            return CameraUndistorter(matrix, distortion.reshape(-1))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'Cannot load calibration {self.calibration}: {exc}') from exc

    def _extract_features(self, image):
        corrected = self.undistorter.undistort(image)
        mask = self.undistorter.undistort(np.full(image.shape[:2], 255, np.uint8)) == 255
        if self.projector is not None:
            corrected, mask = self.projector.project(corrected, mask)
        features = SuperPointObject(corrected, self.device, self.max_keypoints)
        # Ignore artificial features along invalid projection borders.
        valid = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
        xy = np.rint(features.keypoints).astype(int)
        keep = valid[np.clip(xy[:, 1], 0, valid.shape[0]-1),
                     np.clip(xy[:, 0], 0, valid.shape[1]-1)].astype(bool)
        features.keypoints = features.keypoints[keep]
        features.descriptors = features.descriptors[keep]
        features.scores = features.scores[keep]
        features.coverage_mask = mask
        return features

    def iter_matches(self):
        """Yield accepted pairs; skipped frames do not advance the reference."""
        self.selected_indices = [0]
        self.selection_log = []
        first = self._extract_features(self.reader.images[0])
        self._selection_features = {0: first}
        selector = None
        if self.max_overlap is not None:
            selector = OverlapFilter(self.selection_warper or WarpToMatch(),
                                     self.max_overlap, self.min_overlap)
            selector.start(first)
        cache = {}
        index = 1
        with tqdm(total=len(self.reader.images) - 1, desc='Scanning frames', unit='image',
                  disable=selector is None) as progress:
            while index < len(self.reader.images):
                candidates = []
                stop = len(self.reader.images)
                cursor = index
                while cursor < stop:
                    if cursor not in cache:
                        cache[cursor] = self._extract_features(self.reader.images[cursor])
                        self._selection_features[cursor] = cache[cursor]
                        progress.update(1)
                    second = cache[cursor]
                    matches = self.matcher.match(
                        first.keypoints, first.descriptors,
                        second.keypoints, second.descriptors,
                        first.image_size, second.image_size,
                    )
                    candidate = (selector.evaluate(first, second, matches) if selector else
                                 {'accepted': True, 'reason': 'eligible', 'overlap': None, 'score': 0.})
                    row = {k: v for k, v in candidate.items() if k not in ('transform', 'footprint')}
                    row.update(index=cursor, reference_index=self.selected_indices[-1],
                               path=str(self.reader.image_paths[cursor]), accepted=False)
                    self.selection_log.append(row)
                    if candidate['accepted']:
                        candidates.append((cursor, second, matches, candidate, row))
                        if len(candidates) == 1:
                            stop = min(stop, cursor + (self.selection_lookahead if selector else 0) + 1)
                    cursor += 1
                if not candidates:
                    break
                # Preserve room for the minimum image count near the end.
                remaining_needed = max(0, self.min_images - len(self.selected_indices) - 1)
                viable = [item for item in candidates
                          if len(self.reader.images) - item[0] - 1 >= remaining_needed]
                chosen = max(viable or candidates, key=lambda item: item[3]['score'])
                chosen_index, second, matches, candidate, row = chosen
                for item in candidates:
                    item[4]['reason'] = 'lower-ranked lookahead candidate'
                row.update(accepted=True, reason='selected')
                if selector:
                    selector.commit(candidate)
                self.selected_indices.append(chosen_index)
                progress.set_postfix(retained=len(self.selected_indices),
                                     frame=self.reader.image_paths[chosen_index].name)
                yield chosen_index, first, second, matches
                first = second
                index = chosen_index + 1
                # Keep unconsumed lookahead features; rematch to the new anchor.
                cache = {i: feature for i, feature in cache.items() if i >= index}
        self.selection_meets_requirement = len(self.selected_indices) >= self.min_images
        if len(self.selected_indices) < 2:
            raise ValueError('No usable overlapping image pair was found. Inspect selection_log.')

    def select_pairs(self, refresh=False):
        """Select once and cache the results until explicitly refreshed.

        An assignment minimum is a reported status, not a reason to discard a
        usable pair. The overlap limits are never relaxed to meet that minimum.
        """
        if refresh or not hasattr(self, '_selected_pairs'):
            try:
                self._selected_pairs = list(self.iter_matches())
            except ValueError as exc:
                if str(exc) != 'No usable overlapping image pair was found. Inspect selection_log.':
                    raise
                self._selected_pairs = []
            if getattr(self, 'max_overlap', None) is not None and not self.selection_meets_requirement:
                self._selected_pairs = self._search_alternative_sequences(self._selected_pairs)
        return list(self._selected_pairs)

    def _search_alternative_sequences(self, greedy_pairs):
        """Search alternative starts and branches before declaring a short sequence.

        Direct pair estimates avoid accumulating homography errors during overlap
        decisions. Search stops upon finding the requested count; otherwise it
        reports the longest sequence under the measured pair constraints.
        """
        from matching.sequence_search import find_sequence
        features = self._selection_features
        for i in range(len(self.reader.images)):
            if i not in features:
                features[i] = self._extract_features(self.reader.images[i])
        measurements = {}
        warper = self.selection_warper or WarpToMatch()
        with tqdm(desc='Checking alternative pairs', unit='pair') as progress:
            def measure(i, j):
                if (i, j) not in measurements:
                    first, second = features[i], features[j]
                    matches = self.matcher.match(first.keypoints, first.descriptors,
                                                 second.keypoints, second.descriptors,
                                                 first.image_size, second.image_size)
                    selector = OverlapFilter(warper, self.max_overlap, self.min_overlap)
                    selector.start(first)
                    result = selector.evaluate(first, second, matches)
                    measurements[i, j] = (result, matches)
                    progress.update(1)
                return measurements[i, j][0]
            indices = find_sequence(len(self.reader.images), measure,
                                    self.min_images, self.max_overlap)
        self.search_summary = {'checked_pairs': len(measurements), 'selected_count': len(indices),
                                   'target_count': self.min_images}
        self.alternative_log = [dict(reference_index=i, index=j,
                                    **{k: v for k, v in result.items()
                                       if k not in ('transform', 'footprint')})
                                for (i, j), (result, _) in measurements.items()]
        self.selected_indices = indices
        self.selection_meets_requirement = len(indices) >= self.min_images
        if len(indices) < 2:
            raise ValueError('Alternative-start search found no reliable overlapping pair.')
        print(f'Alternative search: {len(indices)} retained frames; '
              f'{len(measurements)} directly measured pairs.')
        return [(j, features[i], features[j], measurements[i, j][1])
                for i, j in itertools.pairwise(indices)]

    def selected_overlap_report(self, pairs):
        """Audit overlap without selection; invalid pair geometry is unknown.

        Unknown overlap must not be counted as zero or as passing a limit. A
        nonadjacent pair can have an unreliable homography even when adjacent
        pairs stitch successfully, so preserve the remaining audit results.
        """
        pairs = list(pairs)
        if not pairs:
            return []
        features = [pairs[0][1]] + [pair[2] for pair in pairs]
        if len(self.selected_indices) != len(features):
            raise ValueError('Selected indices do not correspond to the supplied pairs.')
        known = {(row['reference_index'], row['index']): row
                 for row in getattr(self, 'alternative_log', [])}
        rows = []
        warper = self.selection_warper or WarpToMatch()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                key = self.selected_indices[i], self.selected_indices[j]
                overlap, reason = None, ''
                try:
                    if key in known:
                        overlap = known[key]['overlap']
                        if overlap is None:
                            reason = known[key].get('reason', 'No reliable cached overlap estimate')
                    else:
                        first, second = features[i], features[j]
                        matches = (pairs[i][3] if j == i + 1 else self.matcher.match(
                            first.keypoints, first.descriptors, second.keypoints,
                            second.descriptors, first.image_size, second.image_size))
                        H = warper.estimate_homography(first.keypoints, second.keypoints, matches)
                        overlap = OverlapFilter.overlap(
                            OverlapFilter.footprint(first.image, np.eye(3)),
                            OverlapFilter.footprint(second.image, H))
                    if overlap is None or not np.isfinite(overlap) or not 0 <= overlap <= 1:
                        overlap = None
                        reason = reason or 'No reliable overlap estimate'
                except (ValueError, cv2.error) as exc:
                    overlap, reason = None, str(exc)
                rows.append({'first': key[0], 'second': key[1], 'overlap': overlap,
                             'status': 'unknown' if overlap is None else 'measured',
                             'reason': reason})
        return rows

    def match_selected(self):
        """Match every adjacent pair in an already-selected reader, without filtering."""
        if self.max_overlap is not None:
            raise ValueError('match_selected requires an explicitly preselected, unfiltered input.')
        if hasattr(self, '_unfiltered_pairs'):
            return list(self._unfiltered_pairs)
        pairs = []
        first = self._extract_features(self.reader.images[0])
        for index in tqdm(range(1, len(self.reader.images)),
                          desc='Matching selected frames', unit='pair'):
            second = self._extract_features(self.reader.images[index])
            matches = self.matcher.match(first.keypoints, first.descriptors,
                                         second.keypoints, second.descriptors,
                                         first.image_size, second.image_size)
            pairs.append((index, first, second, matches))
            first = second
        self.selected_indices = list(range(len(self.reader.images)))
        self._unfiltered_pairs = pairs
        return list(pairs)

    @staticmethod
    def draw_matches(first, second, matches):
        """Draw indexed matches on the same images used for extraction."""
        keypoints = [[cv2.KeyPoint(float(x), float(y), 1) for x, y in obj.keypoints]
                     for obj in (first, second)]
        return cv2.drawMatches(
            first.image, keypoints[0], second.image, keypoints[1],
            [cv2.DMatch(int(i), int(j), 0.0) for i, j in matches],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    def match(self):
        """Process the folder and return a list of saved visualization Paths.

        Intermediate features are reused for the next pair. Calling again
        reprocesses the images and overwrites the same output filenames.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_paths = []
        with tqdm(self.iter_matches(), total=len(self.reader.images) - 1,
                  desc='Matching images', unit='pair') as progress:
            for index, first, second, matches in progress:
                drawn = self.draw_matches(first, second, matches)
                name1, name2 = [self.reader.image_paths[i] for i in self.selected_indices[-2:]]
                output = self.output_dir / f'{index:04d}_{name1.stem}__{name2.stem}.jpg'
                if not cv2.imwrite(str(output), drawn):
                    raise OSError(f'Could not write {output}')
                self.output_paths.append(output)
                progress.set_postfix(matches=len(matches), pair=f'{name1.name} -> {name2.name}')
        return list(self.output_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_folder', type=Path)
    parser.add_argument('--calibration', type=Path, default=HW3_DIR / 'camera_cal.json',
                        help='Camera calibration JSON (default: HW3/camera_cal.json)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-keypoints', type=int, default=2048)
    args = parser.parse_args()
    try:
        matcher = ImageMatcher(args.input_folder, args.device, args.max_keypoints,
                               args.calibration)
    except ValueError as exc:
        parser.error(str(exc))
    outputs = matcher.match()
    print(f'Saved {len(outputs)} match visualizations to {matcher.output_dir}')


if __name__ == '__main__':
    main()
