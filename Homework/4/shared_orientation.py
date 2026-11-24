"""Batch-fit a stationary marker with shared rotation and per-image translations.

Requires fixed camera orientation and fixed marker; uses all supplied frames.
Expected spacing is used only after fitting, for comparison.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares


def fit_shared_rotation(object_points, observations, matrix, distortion):
    """Minimize pixel residuals; one rotation and one translation per view."""
    initial = []
    for points in observations:
        ok, r, t = cv2.solvePnP(object_points, points, matrix, distortion,
                               flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError('Cannot initialize marker pose')
        initial.append((r.reshape(3), t.reshape(3)))

    def residual(x):
        translations = x[3:].reshape(-1, 3)
        return np.concatenate([
            (cv2.projectPoints(object_points, x[:3], t, matrix, distortion)[0]
             .reshape(-1, 2) - points).ravel()
            for points, t in zip(observations, translations)])

    # Test each independently initialized orientation; choose pixel-fit error,
    # never agreement with the known camera displacement.
    candidates = []
    for r, _ in initial:
        x = np.concatenate([r, np.array([t for _, t in initial]).ravel()])
        fit = least_squares(residual, x, method='lm', max_nfev=3000,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        translations = fit.x[3:].reshape(-1, 3)
        rotation = cv2.Rodrigues(fit.x[:3])[0]
        depths = [(rotation @ object_points.T).T[:, 2] + t[2] for t in translations]
        if fit.success and all(np.all(z > 0) for z in depths):
            candidates.append(fit)
    if not candidates:
        raise ValueError('No converged positive-depth shared-orientation fit')
    fit = min(candidates, key=lambda result: float(np.sum(result.fun**2)))
    return fit.x[:3], fit.x[3:].reshape(-1, 3), residual(fit.x).reshape(len(observations), -1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--marker-length', type=float, default=.10)
    parser.add_argument('--expected-step', type=float, default=.0762,
                        help='Comparison only, meters; never used in pose optimization')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    detector_class = importlib.import_module('Homework.4.aruco.detection').MarkerDetector
    calibration = importlib.import_module('Homework.4.aruco.calibration').load_calibration
    annotate = importlib.import_module('Homework.4.aruco.visualization').annotate
    matrix, distortion = calibration(args.calibration)
    detector = detector_class(matrix, distortion, args.marker_length)
    files = sorted(args.input.glob('*.jpg'))
    if not files:
        raise ValueError('No jpg captures found')
    observations = []
    for path in files:
        _, _, poses = detector.detect(cv2.imread(str(path)))
        if 0 not in poses:
            raise ValueError(f'Reference marker 0 unavailable: {path}')
        observations.append(poses[0].corners)
    r, translations, errors = fit_shared_rotation(detector.object_points, observations, matrix, distortion)
    # With common R, C0-from-Ci has identity rotation and translation t0-ti.
    positions = translations[0] - translations
    output = args.output.resolve()
    if output == args.input.resolve() or args.input.resolve() in output.parents:
        raise ValueError('Output must be outside input')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (path, t, position, error) in enumerate(zip(files, translations, positions, errors)):
        image = cv2.imread(str(path)); corners, ids, poses = detector.detect(image)
        pose = poses[0];pose.rvec = r.reshape(3,1);pose.tvec = t.reshape(3,1)
        rmse = float(np.sqrt(np.mean(np.sum(error**2,axis=1))))
        transform = np.eye(4);transform[:3,3] = position
        annotated = annotate(image, corners, ids, poses, matrix, distortion,
                             args.marker_length/2, transform, 0)
        target = output/f'{i:05d}_{path.stem}.png'
        if not cv2.imwrite(str(target), annotated):raise OSError(target)
        records.append({'frame':i,'source':str(path.resolve()),'position_m':position.tolist(),
                        'rotation_vector_rad':[0.,0.,0.],'rotation_angle_deg':0.,
                        'T_initial_camera_from_current_camera':transform.tolist(),
                        'marker_reprojection_rmse_px':rmse,'annotated_image':str(target)})
    steps = np.linalg.norm(np.diff(positions,axis=0),axis=1)
    report = {'method':'batch shared marker-to-camera rotation; independent translation per frame',
              'assumption':'stationary marker and zero camera rotation across all frames',
              'calibration':str(args.calibration.resolve()),'marker_length_m':args.marker_length,
              'shared_marker_rvec_rad':r.tolist(),'fit_rmse_px':float(np.sqrt(np.mean(np.sum(errors**2,axis=2)))),
              'expected_step_m':args.expected_step,'expected_spacing_used_in_fit':False,
              'mean_step_m':float(steps.mean()),'median_step_m':float(np.median(steps)),
              'step_rmse_m':float(np.sqrt(np.mean((steps-args.expected_step)**2))),
              'steps_m':steps.tolist(),'frames':records}
    (output/'motion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['frames','steps_m']},indent=2))


if __name__ == '__main__':main()
