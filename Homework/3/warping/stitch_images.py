"""Stitch an ordered image folder using our matcher and custom warper."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from matching.match_images import HW3_DIR, ImageMatcher
from tqdm.auto import tqdm
from warping.cylindrical import CylindricalProjector, CylindricalWarper
from warping.geometry import WarpRejected
from warping.local_alignment import LocalAlignment
from warping.warp_to_match import WarpToMatch


class ImageStitcher:
    """Build a panorama from consecutive overlapping images in sorted order.

    Reuses ImageMatcher for loading, calibration, extraction and matching, and
    WarpToMatch for alignment and blending. No OpenCV Stitcher is used.
    """

    def __init__(self, input_folder, device='cpu', max_keypoints=2048,
                 calibration=None, ransac_threshold=3.0,
                 projection="planar", local_alignment=False,
                 max_overlap=.8, min_overlap=.15, min_images=4, selected_indices=None):
        self.matcher = ImageMatcher(input_folder, device, max_keypoints, calibration,
                                    max_overlap=max_overlap, min_overlap=min_overlap,
                                    min_images=min_images)
        if selected_indices is not None:
            indices = list(selected_indices)
            if (len(indices) < 2 or indices != sorted(set(indices))
                    or any(not isinstance(i, (int, np.integer)) or i < 0
                           or i >= len(self.matcher.reader.images) for i in indices)):
                raise ValueError('selected_indices must contain at least two ordered unique valid indices.')
            self.matcher.reader.images = [self.matcher.reader.images[i] for i in indices]
            self.matcher.reader.image_paths = [self.matcher.reader.image_paths[i] for i in indices]
            # These frames were selected upstream; do not select them again.
            self.matcher.max_overlap = None
        if projection not in ('planar', 'cylindrical'):
            raise ValueError('projection must be planar or cylindrical')
        self.projection = projection
        if projection == 'cylindrical':
            self.matcher.projector = CylindricalProjector(self.matcher.undistorter.camera_matrix)
        warper_class = CylindricalWarper if projection == 'cylindrical' else WarpToMatch
        self.warper = warper_class(ransac_threshold=ransac_threshold,
                                  local_refiner=LocalAlignment() if local_alignment else None)
        self.matcher.selection_warper = warper_class(ransac_threshold=ransac_threshold)
        self.output_path = (HW3_DIR / 'outputs' / 'stitches'
                            / self.matcher.input_folder.name / 'panorama.jpg')
        if projection != 'planar' or local_alignment:
            suffix = '_local' if local_alignment else ''
            self.output_path = self.output_path.with_name(f'panorama_{projection}{suffix}.jpg')
        self.panorama = None
        self.coverage_mask = None

    def stitch(self, pairs=None, selected_indices=None):
        """Render cached pairs, or match/select on demand when pairs is None.

        Cached features must use this stitcher's projection and calibration.
        selected_indices maps cached features to this matcher's image paths.
        """
        self.panorama = None
        self.coverage_mask = None
        previous_transform = np.eye(3)
        anchor = None
        accepted_indices = []
        self.rejected_warps = []
        if pairs is not None:
            pairs = list(pairs)
            indices = list(selected_indices) if selected_indices is not None else [0] + [p[0] for p in pairs]
            if (not pairs or len(indices) != len(pairs) + 1
                    or indices != sorted(set(indices))
                    or any(i < 0 or i >= len(self.matcher.reader.images) for i in indices)):
                raise ValueError('Cached pairs require valid corresponding selected_indices.')
            self.matcher.selected_indices = indices
        if pairs is None:
            pairs = (self.matcher.select_pairs() if self.matcher.max_overlap is not None
                     else self.matcher.match_selected())
        source = pairs
        total = len(pairs)
        with tqdm(source, total=total,
                  desc='Stitching images', unit='pair') as progress:
            for pair_number, (index, first, second, matches) in enumerate(progress):
                source_indices = self.matcher.selected_indices[pair_number:pair_number + 2]
                if anchor is None:
                    anchor = first
                    self.panorama = first.image
                    self.coverage_mask = first.coverage_mask
                    accepted_indices.append(source_indices[0])
                if first is not anchor:
                    # A rejected image must never become the next frame's anchor.
                    matches = self.matcher.matcher.match(
                        anchor.keypoints, anchor.descriptors, second.keypoints,
                        second.descriptors, anchor.image_size, second.image_size)
                first = anchor
                # Matching stays in the original images. Warping uses coordinates
                # of the preceding image mapped into the growing panorama.
                points = cv2.perspectiveTransform(
                    first.keypoints.astype(np.float64)[None], previous_transform,
                )[0] if len(first.keypoints) else first.keypoints
                try:
                    self.panorama = self.warper.warp(
                        self.panorama, second.image, points, second.keypoints,
                        matches, mask1=self.coverage_mask, mask2=second.coverage_mask,
                    )
                except WarpRejected as exc:
                    path = self.matcher.reader.image_paths[source_indices[1]]
                    self.rejected_warps.append({'path': str(path), 'reason': str(exc)})
                    tqdm.write(f'Rejected warp for {path.name}: {exc}')
                    continue
                except ValueError as exc:
                    names = [self.matcher.reader.image_paths[i] for i in
                             self.matcher.selected_indices[pair_number:pair_number + 2]]
                    raise ValueError(f'Cannot stitch {names[0].name} -> {names[1].name}: {exc}') from exc
                anchor = second
                accepted_indices.append(source_indices[1])
                previous_transform = self.warper.transform2.copy()
                self.coverage_mask = self.warper.coverage_mask.copy()
                progress.set_postfix(matches=len(matches), inliers=int(self.warper.inlier_mask.sum()))
        if len(accepted_indices) < 2:
            raise ValueError('No safe image pair could be warped; inspect rejected_warps.')
        self.selected_paths = [self.matcher.reader.image_paths[i] for i in accepted_indices]
        self.meets_image_requirement = len(self.selected_paths) >= self.matcher.min_images
        self.save_result()
        return self.panorama

    def save_result(self, output_path=None):
        """Save the current panorama and return its path."""
        if self.panorama is None:
            raise RuntimeError('Call stitch() before saving a panorama.')
        path = Path(output_path) if output_path is not None else self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.panorama):
            raise OSError(f'Could not write {path}')
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_folder', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-keypoints', type=int, default=2048)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--ransac-threshold', type=float, default=3.0)
    parser.add_argument('--projection', choices=['planar', 'cylindrical'], default='planar')
    parser.add_argument('--local-alignment', action='store_true')
    parser.add_argument('--max-overlap', type=float, default=.8)
    parser.add_argument('--min-overlap', type=float, default=.15)
    args = parser.parse_args()
    try:
        stitcher = ImageStitcher(
            args.input_folder,
            device=args.device,
            max_keypoints=args.max_keypoints,
            calibration=args.calibration,
            ransac_threshold=args.ransac_threshold,
            projection=args.projection,
            local_alignment=args.local_alignment,
            max_overlap=args.max_overlap,
            min_overlap=args.min_overlap,
        )
        stitcher.stitch()
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Saved panorama to {stitcher.output_path}')
    if not stitcher.meets_image_requirement:
        print(f'Assignment incomplete: only {len(stitcher.selected_paths)} usable frames; '
              f'{stitcher.matcher.min_images} required. Overlap limits were not relaxed.')


if __name__ == '__main__':
    main()
