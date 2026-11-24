# HW4 capture measurements

- `cal_caps/`: checkerboard photos for camera calibration (9 × 6 inner corners).
- `aruco_caps/`: chronological camera translations of **3 inches (7.62 cm)** between photos, as most recently confirmed by the user. The earlier 2-inch correction was superseded. Camera orientation stayed fixed.
- ArUco marker: **10 cm × 10 cm**. The pose pipeline interprets this as the outer black-square size, excluding white margins.

Shared-orientation results are in `../outputs/captures_3inch_shared/`. Earlier `captures_2inch` and calibration-audit outputs are preserved as historical runs with superseded spacing assumptions.
