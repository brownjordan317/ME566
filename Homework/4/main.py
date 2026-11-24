"""Command-line image sequence processing using the provided io/ImageReader."""
import argparse
import importlib
import json
from pathlib import Path

import cv2
import numpy as np

# Make the repository namespace available for the existing io package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_package = 'Homework.4'
load_calibration = importlib.import_module(f'{_package}.aruco.calibration').load_calibration
MarkerDetector = importlib.import_module(f'{_package}.aruco.detection').MarkerDetector
CameraMotion = importlib.import_module(f'{_package}.aruco.motion').CameraMotion
annotate = importlib.import_module(f'{_package}.aruco.visualization').annotate
ImageReader = importlib.import_module(f'{_package}.io.image_reader').ImageReader


def run(args):
    source, output = args.input.resolve(), args.output.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError('Output must be outside the input directory to avoid reading old annotations.')
    matrix, distortion = load_calibration(args.calibration)
    reader = ImageReader(source)
    if not reader.images:
        raise ValueError('No readable input images found.')
    shape = reader.images[0].shape[:2]
    if any(image.shape[:2] != shape for image in reader.images):
        raise ValueError('All frames must have the same resolution and match the calibration.')
    detector = MarkerDetector(matrix, distortion, args.marker_length, args.dictionary, args.max_error)
    output.mkdir(parents=True, exist_ok=True)
    records, tracker = [], None
    for index, (path, image) in enumerate(zip(reader.image_paths, reader.images)):
        corners, ids, poses = detector.detect(image)
        if index == 0:
            tracker = CameraMotion(poses, args.reference_id)
        transform = tracker.estimate(poses)
        rotation = None if transform is None else cv2.Rodrigues(transform[:3, :3])[0].reshape(3).tolist()
        record = {
            'frame': index, 'source': str(path),
            'detected_ids': [] if ids is None else ids.flatten().tolist(),
            'reference_id': tracker.reference_id,
            'status': 'ok' if transform is not None else 'reference_unavailable',
            'T_initial_camera_from_current_camera': None if transform is None else transform.tolist(),
            'position_m': None if transform is None else transform[:3, 3].tolist(),
            'rotation_vector_rad': rotation,
            'rotation_angle_deg': None if rotation is None else float(np.rad2deg(np.linalg.norm(rotation))),
            'marker_reprojection_rmse_px': {str(k): v.reprojection_rmse for k, v in poses.items()},
        }
        annotated = annotate(image, corners, ids, poses, matrix, distortion,
                             args.marker_length * .5, transform, tracker.reference_id)
        target = output / f'{index:05d}_{path.stem}.png'
        if not cv2.imwrite(str(target), annotated):
            raise OSError(f'Cannot write {target}')
        record['annotated_image'] = str(target)
        records.append(record)
    report = {'units': 'meters and radians',
              'convention': 'T_initial_camera_from_current_camera maps current camera points '
                            'into initial camera coordinates; camera axes are X right, Y down, Z forward.',
              'dictionary': args.dictionary, 'marker_length_m': args.marker_length,
              'frames': records}
    (output / 'motion.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Image or naturally sorted image folder')
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--marker-length', type=float, required=True, help='Outer black square side in meters')
    parser.add_argument('--dictionary', default='DICT_4X4_50')
    parser.add_argument('--reference-id', type=int, help='Default: lowest valid ID in the first image')
    parser.add_argument('--max-error', type=float, default=3., help='Maximum pose reprojection RMSE in pixels')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    try:
        records = run(args)
    except (ValueError, OSError, KeyError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Processed {len(records)} images; motion available for '
          f'{sum(r["status"] == "ok" for r in records)}. Outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
