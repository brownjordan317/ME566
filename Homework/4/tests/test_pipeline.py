"""Synthetic geometric and CLI regression checks; no physical camera needed."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MarkerDetector = importlib.import_module('Homework.4.aruco.detection').MarkerDetector
CameraMotion = importlib.import_module('Homework.4.aruco.motion').CameraMotion


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[1400., 0, 400], [0, 1400, 300], [0, 0, 1]])
        self.distortion = np.zeros(5)
        self.detector = MarkerDetector(self.matrix, self.distortion, .12)

    def frame(self, x=0.):
        # Marker faces the camera, with local Y pointing upward.
        rotation = np.array([2.8, .15, .2])
        translation = np.array([x, .01, .7])
        corners = cv2.projectPoints(self.detector.object_points, rotation, translation,
                                   self.matrix, self.distortion)[0].reshape(4, 2).astype(np.float32)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 240)
        transform = cv2.getPerspectiveTransform(
            np.array([[0, 0], [239, 0], [239, 239], [0, 239]], np.float32), corners)
        gray = cv2.warpPerspective(marker, transform, (800, 600),
                                   flags=cv2.INTER_NEAREST, borderValue=255)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_camera_displacement_and_missing_reference(self):
        initial = self.detector.detect(self.frame())[2]
        current = self.detector.detect(self.frame(-.035))[2]
        self.assertIn(7, initial)
        tracker = CameraMotion(initial)
        np.testing.assert_allclose(tracker.estimate(initial), np.eye(4), atol=1e-10)
        # A marker shifted left corresponds to the camera moving right.
        np.testing.assert_allclose(tracker.estimate(current)[:3, 3], [.035, 0, 0], atol=.008)
        self.assertIsNone(tracker.estimate({}))
        with self.assertRaises(ValueError):
            CameraMotion({})

    def test_rotation_composition(self):
        from types import SimpleNamespace
        initial = np.eye(4)
        initial[:3, 3] = [0, 0, 1]
        expected = np.eye(4)
        expected[:3, :3] = cv2.Rodrigues(np.array([.1, -.2, .05]))[0]
        expected[:3, 3] = [.03, -.02, .01]
        observed = np.linalg.inv(expected) @ initial
        tracker = CameraMotion({7: SimpleNamespace(transform=initial)})
        actual = tracker.estimate({7: SimpleNamespace(transform=observed)})
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_cli_reader_annotation_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / 'images'
            inputs.mkdir()
            cv2.imwrite(str(inputs / 'frame_10.png'), self.frame(-.035))
            cv2.imwrite(str(inputs / 'frame_2.png'), self.frame())
            cv2.imwrite(str(inputs / 'frame_11.png'), np.full((600, 800, 3), 255, np.uint8))
            calibration = root / 'calibration.json'
            calibration.write_text(json.dumps({'camera_matrix': self.matrix.tolist(),
                                               'dist_coeffs': self.distortion.tolist()}))
            result = subprocess.run([
                sys.executable, str(ROOT / 'Homework/4/main.py'), str(inputs),
                '--calibration', str(calibration), '--marker-length', '.12',
                '--output', str(root / 'output')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'output/motion.json').read_text())
            frames = report['frames']
            self.assertEqual(Path(frames[0]['source']).name, 'frame_2.png')
            self.assertEqual(frames[2]['status'], 'reference_unavailable')
            self.assertIsNone(frames[2]['position_m'])
            self.assertAlmostEqual(frames[1]['position_m'][0], .035, delta=.008)
            image = cv2.imread(frames[0]['annotated_image'])
            self.assertTrue(np.any(image[:, :, 0] != image[:, :, 1]))


if __name__ == '__main__':
    unittest.main()
