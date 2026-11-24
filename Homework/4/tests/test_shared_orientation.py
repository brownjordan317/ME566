"""Verify shared-orientation geometry against known synthetic translations."""
import importlib
import unittest
import cv2
import numpy as np

fit = importlib.import_module('Homework.4.shared_orientation').fit_shared_rotation

class SharedOrientationTests(unittest.TestCase):
    def test_exact_and_noisy_translation_recovery(self):
        obj = np.array([[-.05,.05,0],[.05,.05,0],[.05,-.05,0],[-.05,-.05,0]], float)
        matrix = np.array([[2800.,0,1496],[0,2800,1496],[0,0,1.]])
        rotation = np.array([2.9,.1,-.2])
        # Deliberately unequal steps, so fitting cannot depend on equal spacing.
        translations = np.array([[.20,.1,2.],[.13,.11,2.01],[.03,.09,2.02],[-.02,.1,2.]])
        points = [cv2.projectPoints(obj,rotation,t,matrix,np.zeros(5))[0].reshape(-1,2) for t in translations]
        for noise,tolerance in [(0.,1e-6),(.1,.003)]:
            with self.subTest(noise=noise):
                observations = [q+np.random.default_rng(i).normal(0,noise,q.shape) for i,q in enumerate(points)]
                _, fitted, _ = fit(obj,observations,matrix,np.zeros(5))
                np.testing.assert_allclose(fitted[0]-fitted,translations[0]-translations,atol=tolerance)
