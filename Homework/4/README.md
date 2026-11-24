# Homework 4: ArUco camera motion

Detect ArUco markers in a naturally ordered image sequence, outline and label each
marker, draw its coordinate frame, and estimate each camera pose relative to the
**first input image**. The program calls the existing `io/image_reader.py` without
modifying it. No GUI is required.

## Run

From the repository root:

```bash
python -m pip install -r Homework/4/requirements.txt
python Homework/4/main.py /path/to/images \
    --calibration /path/to/camera_cal.json \
    --marker-length 0.05 \
    --dictionary DICT_4X4_50 \
    --output Homework/4/outputs
```

Replace `0.05` with the measured outer black-square side length **in meters**.
The dictionary must match the printed markers. `--reference-id 7` selects a
specific stationary marker; otherwise the lowest valid ID in the initial frame
is used. That marker must be visible in the first image. It must remain fixed
in the scene for the camera-motion interpretation to hold. Duplicate IDs in a
single frame are outlined but excluded from pose estimation.

Calibration JSON uses `camera_matrix` (3 by 3) and `dist_coeffs` (4, 5, 8, 12, or
14 coefficients), matching the earlier homework format. Use calibration for the
actual camera, resolution, crop, and focus. Earlier calibration is usable only
if those conditions match. Process raw distorted images: distortion is included
in pose estimation and drawing. Do not supply already undistorted images with
nonzero distortion coefficients. Mixed-resolution sequences are rejected.
The existing reader loads images into memory; use a manageable sequence.
Output must be outside the input folder, which is searched recursively.

## Results and conventions

Each frame produces a PNG with marker outlines, IDs, and centered coordinate axes:
**X red, Y green, Z blue**. Marker X points right, Y up, and Z out of the printed
face. Outlines can still appear when a marker pose fails validation.

`motion.json` records detected IDs, accepted pose reprojection errors, status,
camera position in meters, a rotation vector in radians, rotation angle in degrees,
and the full 4 by 4 `T_initial_camera_from_current_camera` matrix. Image filenames
are naturally sorted, so frame 2 precedes frame 10. The first successfully loaded
image is the initial frame; unreadable images are skipped by the existing reader.

OpenCV PnP yields `T_Ci_M`, which maps marker coordinates into camera coordinates.
The reported camera pose is:

```text
T_C0_Ci = T_C0_M @ inverse(T_Ci_M)
```

Its translation is the current camera center expressed in the initial camera frame.
Its rotation describes the current camera's orientation in that frame. Camera axes
are X right, Y down, Z forward. The initial pose is identity. This is a direct
initial-to-current measurement, with no accumulation of frame-to-frame motion.
If you need to transform initial-camera points into the current camera instead,
invert the reported matrix.

When the reference marker is absent or rejected, the frame remains annotated but
motion values are JSON `null`, with `reference_unavailable` status. The reference
is never silently changed. Other detected markers receive their own axes but do
not substitute for the reference without a known spatial marker map.

Pose solutions must place all corners in front of the camera and have corner
reprojection RMSE at most 3 pixels (`--max-error` changes this threshold). Square
markers can have ambiguous poses, especially when small or nearly front-facing;
a low reprojection error does not guarantee accurate motion. Prefer a large,
sharp marker at a modest viewing angle. Incorrect marker size scales translation.

## Modules and verification

- `aruco/calibration.py`: calibration loading and validation.
- `aruco/detection.py`: detection, square pose estimation, and quality checks.
- `aruco/motion.py`: rigid transforms and initial-relative camera poses.
- `aruco/visualization.py`: marker outlines, axes, and motion annotations.
- `main.py`: existing-reader integration, CLI, and PNG/JSON export.

Run the synthetic geometry and end-to-end CLI checks:

```bash
python -m unittest discover -s Homework/4/tests -v
```

Implementation follows the OpenCV marker detection and pose conventions:
https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html


## Captured fixed-orientation sequence

The capture measurements are 10 cm marker sides and **3-inch (7.62 cm)** camera translations with no rotation. Use `shared_orientation.py` for the batch constant-orientation fit; the general `main.py` pipeline remains available for unconstrained camera motion. See [results and reproduction](outputs/captures_3inch_shared/README.md). This fit estimates mean steps of 8.06 cm and does not force agreement with the expected spacing.


## Final assignment writeup

See [the six-part PDF](latex/report.pdf) and [editable LaTeX source](latex/report.tex). The source package is `outputs/hw4_source.zip`; submit it with `hw4_captures.zip` and the PDF. The report includes conditional pose covariance, all frame measurements, equations, parameters and reproduction commands.
