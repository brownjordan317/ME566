"""Calibrate HW4 captures and compare ArUco camera steps with measured spacing."""
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

root = Path(sys.argv[1]).resolve()
hw = root / 'Homework/4'
out = hw / 'outputs/captures_2inch'
out.mkdir(parents=True, exist_ok=True)
files = sorted((hw / 'hw4_captures/cal_caps').glob('*.jpg'))
obj = np.zeros((54, 3), np.float32)
obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objects, corners, accepted, rejected = [], [], [], []
size = None
for path in files:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        rejected.append(str(path)); continue
    current = image.shape[::-1]
    if size is None: size = current
    if current != size: raise ValueError('Calibration image sizes differ')
    ok, points = cv2.findChessboardCornersSB(image, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        objects.append(obj.copy()); corners.append(points); accepted.append(str(path))
    else: rejected.append(str(path))
if len(objects) < 10: raise ValueError('Too few usable calibration views')
rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(objects, corners, size, None, None)
errors = []
for points, rv, tv in zip(corners, rotations, translations):
    pred = cv2.projectPoints(obj, rv, tv, matrix, distortion)[0]
    errors.append(float(np.sqrt(np.mean(np.sum((pred.reshape(-1,2)-points.reshape(-1,2))**2, axis=1)))))
cal = {'camera_matrix': matrix.tolist(), 'dist_coeffs': distortion.tolist(),
       'image_size_wh': list(size), 'rms_error_px': rms, 'inner_corners_wh': [9,6],
       'checkerboard_square_units': 'unit squares; physical size unknown; only intrinsics used',
       'accepted_images': accepted, 'rejected_images': rejected, 'per_image_rmse_px': errors}
calpath = out / 'camera_cal.json'
calpath.write_text(json.dumps(cal, indent=2)+'\n')
print('Calibration:', len(accepted), 'accepted,', len(rejected), 'rejected; RMS', rms, flush=True)
subprocess.run([sys.executable, str(hw/'main.py'), str(hw/'hw4_captures/aruco_caps'),
                '--calibration', str(calpath), '--marker-length','0.10',
                '--dictionary','DICT_4X4_50','--reference-id','0','--output',str(out/'aruco')],check=True,cwd=root)
frames = json.loads((out/'aruco/motion.json').read_text())['frames']
steps = []
for first, second in zip(frames, frames[1:]):
    a,b = first['position_m'],second['position_m']
    distance = None if a is None or b is None else float(np.linalg.norm(np.array(b)-np.array(a)))
    steps.append({'from':Path(first['source']).name,'to':Path(second['source']).name,
                  'distance_m':distance,'expected_m':0.0508,
                  'error_m':None if distance is None else distance-0.0508})
valid = [step['distance_m'] for step in steps if step['distance_m'] is not None]
summary = {'marker_black_square_side_m':0.10,'expected_consecutive_camera_translation_m':0.0508,
           'calibration_views':len(accepted),'calibration_rms_px':rms,
           'valid_motion_frames':sum(f['status']=='ok' for f in frames),'total_frames':len(frames),
           'mean_step_m':float(np.mean(valid)) if valid else None,
           'median_step_m':float(np.median(valid)) if valid else None,
           'step_rmse_m':float(np.sqrt(np.mean((np.array(valid)-0.0508)**2))) if valid else None,
           'steps':steps,
           'limitations':['Assumes marker side means outer black square.',
                          'Expected spacing is a comparison only, never fitted into calibration or poses.',
                          'Marker/camera rotation, blur, planar pose ambiguity and calibration quality can affect translation.',
                          'Capture-device crop, focus and lens consistency have not been independently verified.']}
(out/'step_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['steps','limitations']},indent=2))
