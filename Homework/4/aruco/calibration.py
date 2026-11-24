"""Read the JSON calibration format used in the earlier homework."""
import json
from pathlib import Path

import numpy as np


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    matrix = np.asarray(data['camera_matrix'], dtype=np.float64)
    distortion = np.asarray(data['dist_coeffs'], dtype=np.float64).reshape(-1)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError('Invalid camera_matrix: expected a finite 3x3 intrinsic matrix.')
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError('dist_coeffs must contain 4, 5, 8, 12, or 14 finite values.')
    return matrix, distortion
