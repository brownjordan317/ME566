"""Regression checks: selection is cached and rendering cannot select again."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matching.match_images import ImageMatcher


class SelectionReuseTests(unittest.TestCase):
    def test_selection_is_cached_until_explicit_refresh(self):
        matcher = ImageMatcher.__new__(ImageMatcher)
        pair = (1, object(), object(), np.empty((0, 2), int))
        matcher.iter_matches = Mock(side_effect=lambda: iter([pair]))
        self.assertEqual(len(matcher.select_pairs()), 1)
        self.assertEqual(len(matcher.select_pairs()), 1)
        matcher.iter_matches.assert_called_once()
        matcher.select_pairs(refresh=True)
        self.assertEqual(matcher.iter_matches.call_count, 2)

    def test_selected_matching_does_not_enter_selector(self):
        matcher = ImageMatcher.__new__(ImageMatcher)
        matcher.max_overlap = None
        matcher.reader = SimpleNamespace(images=[object(), object()])
        feature = SimpleNamespace(keypoints=np.zeros((12, 2)),
                                  descriptors=np.zeros((12, 256)), image_size=(100, 100))
        matcher._extract_features = Mock(return_value=feature)
        matcher.matcher = Mock()
        matcher.matcher.match.return_value = np.column_stack([np.arange(12)] * 2)
        matcher.iter_matches = Mock(side_effect=AssertionError('Selector called twice'))
        self.assertEqual(len(matcher.match_selected()), 1)
        self.assertEqual(len(matcher.match_selected()), 1)
        matcher.iter_matches.assert_not_called()
        self.assertEqual(matcher._extract_features.call_count, 2)
        matcher.matcher.match.assert_called_once()

    def test_two_valid_frames_are_returned_with_incomplete_status(self):
        matcher = ImageMatcher.__new__(ImageMatcher)
        x, y = np.meshgrid(np.linspace(30, 90, 5), np.linspace(10, 90, 5))
        points = np.column_stack([x.ravel(), y.ravel()])
        images = [np.zeros((100, 100, 3), np.uint8) for _ in range(2)]
        features = [SimpleNamespace(image=im, keypoints=points-[25*i, 0],
                                    descriptors=np.zeros((25, 256)), image_size=(100, 100))
                    for i, im in enumerate(images)]
        matcher.reader = SimpleNamespace(images=images, image_paths=[Path('a.jpg'), Path('b.jpg')])
        matcher._extract_features = Mock(side_effect=features)
        matcher.matcher = Mock()
        matcher.matcher.match.return_value = np.column_stack([np.arange(25)] * 2)
        matcher.max_overlap, matcher.min_overlap = .8, .15
        matcher.min_images, matcher.selection_lookahead = 4, 3
        matcher.selection_warper = None
        self.assertEqual(len(matcher.select_pairs()), 1)
        self.assertFalse(matcher.selection_meets_requirement)
        self.assertEqual(matcher.selected_indices, [0, 1])


class AlternativeSearchTests(unittest.TestCase):
    def test_backtracks_from_dead_end_and_changes_start(self):
        from matching.sequence_search import find_sequence
        # Greedy 0 -> 3 is a dead end. A valid four-frame path starts at 1.
        edges = {(0, 3), (1, 2), (2, 4), (4, 5)}
        def measure(i, j):
            return {'accepted': (i, j) in edges, 'overlap': .5}
        self.assertEqual(find_sequence(6, measure, 4, .8), [1, 2, 4, 5])

    def test_revisits_are_rejected_even_with_valid_adjacent_edges(self):
        from matching.sequence_search import find_sequence
        def measure(i, j):
            return {'accepted': j == i + 1, 'overlap': .95 if (i, j) == (0, 2) else .5}
        self.assertEqual(find_sequence(3, measure, 3, .8), [0, 1])


class OverlapAuditTests(unittest.TestCase):
    def make_matcher(self):
        matcher = ImageMatcher.__new__(ImageMatcher)
        matcher.selected_indices = [0, 1, 2]
        matcher.selection_warper = Mock()
        matcher.matcher = Mock()
        matches = np.column_stack([np.arange(4)] * 2)
        matcher.matcher.match.return_value = matches
        feature = SimpleNamespace(
            image=np.zeros((100, 100, 3), np.uint8),
            keypoints=np.array([[10, 10], [80, 10], [80, 80], [10, 80]], float),
            descriptors=np.zeros((4, 256)), image_size=(100, 100),
        )
        pairs = [(1, feature, feature, matches), (2, feature, feature, matches)]
        return matcher, pairs

    def test_projective_infinity_is_unknown_and_other_pairs_are_audited(self):
        matcher, pairs = self.make_matcher()
        crossing = np.array([[1., 0, 0], [0, 1, 0], [-.02, 0, 1]])
        matcher.selection_warper.estimate_homography.side_effect = [np.eye(3), crossing, np.eye(3)]
        rows = matcher.selected_overlap_report(pairs)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]['overlap'], 1.)
        self.assertIsNone(rows[1]['overlap'])
        self.assertEqual(rows[1]['status'], 'unknown')
        self.assertIn('projective infinity', rows[1]['reason'])
        self.assertAlmostEqual(rows[2]['overlap'], 1.)

    def test_failed_estimation_does_not_abort_audit(self):
        matcher, pairs = self.make_matcher()
        matcher.selection_warper.estimate_homography.side_effect = [
            ValueError('At least four matches are required'), np.eye(3), np.eye(3),
        ]
        rows = matcher.selected_overlap_report(pairs)
        self.assertIsNone(rows[0]['overlap'])
        self.assertEqual(rows[2]['status'], 'measured')

    def test_unknown_cached_measurement_stays_unknown(self):
        matcher, pairs = self.make_matcher()
        matcher.alternative_log = [{'reference_index': i, 'index': j, 'overlap': None,
                                       'reason': 'unreliable alignment'}
                                   for i, j in [(0, 1), (0, 2), (1, 2)]]
        rows = matcher.selected_overlap_report(pairs)
        self.assertTrue(all(row['status'] == 'unknown' for row in rows))
        matcher.matcher.match.assert_not_called()
        matcher.selection_warper.estimate_homography.assert_not_called()


class RenderedComplianceTests(unittest.TestCase):
    def test_rejected_input_does_not_invalidate_verified_four_image_result(self):
        from report_helpers import audit_rendered_panorama
        matcher = ImageMatcher.__new__(ImageMatcher)
        matcher.selected_indices = [0, 1, 2, 3, 4]
        matcher.reader = SimpleNamespace(image_paths=[Path(str(i)) for i in range(5)])
        matcher.selection_warper = Mock()
        matcher.alternative_log = [
            {'reference_index': i, 'index': j, 'overlap': None if j == 4 else .5,
             'reason': 'unbounded' if j == 4 else ''}
            for i in range(5) for j in range(i + 1, 5)
        ]
        feature = SimpleNamespace()
        pairs = [(i, feature, feature, np.empty((0, 2), int)) for i in range(1, 5)]
        result = audit_rendered_panorama(matcher, pairs, matcher.reader.image_paths[:4])
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['overlaps']), 6)
        self.assertEqual(matcher.selected_indices, [0, 1, 2, 3, 4])
        unknown = audit_rendered_panorama(matcher, pairs, matcher.reader.image_paths)
        self.assertFalse(unknown['complete'])


if __name__ == '__main__':
    unittest.main()
