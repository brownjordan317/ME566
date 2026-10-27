"""Reject extreme geometry before allocation and recover the last valid anchor."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warping.geometry import WarpRejected, canvas_bounds, panorama_layout, validate_warp
from warping.stitch_images import ImageStitcher
from warping.warp_to_match import WarpToMatch, blend_images


class WarpGeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((100, 100, 3), 100, np.uint8)

    def test_native_canvas_preserves_negative_translation(self):
        size, transform = canvas_bounds([[-20, -10], [120, 100]])
        self.assertEqual(size, (140, 110))
        np.testing.assert_allclose(transform[:2, 2], [20, 10])

    def test_oversized_canvas_is_rejected_instead_of_scaled(self):
        with self.assertRaises(WarpRejected):
            WarpToMatch().fit_canvas([[0, 0], [50000, 40000]])

    def test_severe_stretch_and_infinity_are_rejected(self):
        for transform in [np.diag([10., 10., 1.]),
                          np.array([[1., 0, 0], [0, 1, 0], [-.02, 0, 1]])]:
            with self.assertRaises(WarpRejected):
                validate_warp(self.image, transform)

    def test_layout_discards_only_extreme_image(self):
        layout = panorama_layout([self.image] * 3,
                                 [np.eye(3), np.diag([10., 10., 1.]),
                                  np.array([[1., 0, 40], [0, 1, 0], [0, 0, 1]])])
        self.assertEqual(layout['kept_indices'], [0, 2])
        self.assertEqual(layout['rejected'][0]['index'], 1)
        self.assertEqual(layout['size'], (140, 100))

    def test_translation_warp_and_function_blend(self):
        points = np.array([[10, 10], [80, 10], [80, 80], [10, 80]], float)
        matches = np.column_stack([np.arange(4)] * 2)
        result = WarpToMatch().warp(self.image, self.image, points + [25, 0], points, matches)
        self.assertLessEqual(abs(result.shape[1] - 125), 2)
        mask = np.ones((100, 100), np.uint8)
        np.testing.assert_allclose(blend_images(self.image, self.image, mask, mask), self.image)

    def test_rejected_frame_never_becomes_next_anchor(self):
        frames = [SimpleNamespace(image=self.image.copy(), keypoints=np.zeros((4, 2)),
                                  descriptors=np.zeros((4, 256)), image_size=(100, 100),
                                  coverage_mask=np.ones((100, 100), bool)) for _ in range(3)]
        matches = np.column_stack([np.arange(4)] * 2)
        stitcher = ImageStitcher.__new__(ImageStitcher)
        stitcher.matcher = SimpleNamespace(
            reader=SimpleNamespace(images=[f.image for f in frames],
                                   image_paths=[Path('a'), Path('b'), Path('c')]),
            matcher=Mock(), min_images=4)
        stitcher.matcher.matcher.match.return_value = matches
        stitcher.warper = Mock(transform2=np.eye(3), coverage_mask=frames[0].coverage_mask,
                              inlier_mask=np.ones(4, bool))
        stitcher.warper.warp.side_effect = [WarpRejected('extreme stretch'), self.image]
        stitcher.save_result = Mock()
        stitcher.stitch(pairs=[(1, frames[0], frames[1], matches),
                              (2, frames[1], frames[2], matches)])
        self.assertIs(stitcher.matcher.matcher.match.call_args.args[0], frames[0].keypoints)
        self.assertIs(stitcher.matcher.matcher.match.call_args.args[2], frames[2].keypoints)
        self.assertEqual(stitcher.selected_paths, [Path('a'), Path('c')])
        self.assertEqual(len(stitcher.rejected_warps), 1)


if __name__ == '__main__':
    unittest.main()
