# HW3: image matching and panoramas

## Setup

```bash
python -m pip install -r Homework/3/requirements.txt
```

Use the same Python environment in your notebook. Pretrained SuperPoint and
LightGlue weights download on first use. CPU is the default; `--device cuda`
enables a compatible GPU. Upstream: https://github.com/cvg/LightGlue.

## Report notebook

Set `INPUT_FOLDER` and `CALIBRATION_FILE` in the first code cell of `report.ipynb`,
then run from the top. Relative input paths are resolved against HW3. The setup
cell reloads edited modules so the kernel uses their current interfaces.

The workflow is:

1. Read the folder in natural filename order, including subfolders.
2. Undistort images and select overlapping frames once during matching.
3. Use the cached planar matches for the homography analysis and panorama.
4. Match the same selected frames in cylindrical coordinates once, then reuse
   those matches for cylindrical rendering with and without local alignment.

The preview shows up to 12 images; selection scans the entire folder. The overlap
filter accepts at most 80% footprint overlap with every retained frame and at least
15% with the last retained frame. Four retained images are required for the assignment. If only two or three are
usable, rendering continues and the report marks the assignment incomplete.
Overlap limits are never relaxed. A sequence with no usable pair still fails.
These are greedy, homography-based estimates, not ground-truth measurements.
The comparisons share the initial selection rather than selecting again in each
projection. Inspect the generated plots before drawing conclusions.

## Command line

Match consecutive images and save match visualizations:

```bash
python Homework/3/matching/match_images.py path/to/images
```

Select frames and create a panorama:

```bash
python Homework/3/warping/stitch_images.py path/to/images
python Homework/3/warping/stitch_images.py path/to/images --projection cylindrical
python Homework/3/warping/stitch_images.py path/to/images --projection cylindrical --local-alignment
```

Both commands accept `--device`, `--max-keypoints`, and `--calibration`.
Stitching also accepts `--ransac-threshold`, `--max-overlap`, and `--min-overlap`.
Standalone matching visualizes all consecutive pairs; standalone stitching selects
frames automatically.

## Classes

From a notebook or Python script rooted in HW3:

```python
from matching.match_images import ImageMatcher
from warping.stitch_images import ImageStitcher

matcher = ImageMatcher('images', max_overlap=0.8)
pairs = matcher.select_pairs()
selected = matcher.selected_indices.copy()

stitcher = ImageStitcher('images')
panorama = stitcher.stitch(pairs=pairs, selected_indices=selected)
```

`ImageMatcher.match()` saves match visualizations. `iter_matches()` exposes feature
pairs for analysis. Selection details are in `selection_log` and `selected_indices`.
`ImageStitcher.stitch()` selects and matches when no cached pairs are supplied.
Passing `selected_indices` to its constructor restricts inputs to an already
selected subset and disables repeated selection. Cached pairs must use the same
projection and calibration as the renderer. `save_result(path)` saves another copy.

`WarpToMatch.warp(image1, image2, keypoints1, keypoints2, matches)` performs custom
RANSAC homography alignment, canvas expansion, and narrow-seam blending. No class uses
OpenCV Stitcher. SuperPoint keypoints have shape `(N, 2)` in `(x, y)` pixel units,
descriptors `(N, 256)`, and LightGlue match indices `(M, 2)`.

## Outputs and limitations

Outputs are rooted in HW3, independent of the working directory:

- `outputs/matches/<folder name>/`: pair visualizations.
- `outputs/stitches/<folder name>/panorama.jpg`: planar panorama.
- `outputs/stitches/<folder name>/panorama_cylindrical.jpg`: cylindrical panorama.
- `outputs/stitches/<folder name>/panorama_cylindrical_local.jpg`: local refinement.

Calibration defaults to `camera_cal.json`, containing `camera_matrix` and
`dist_coeffs`. It must match the camera, resolution, and crop; the matrix is not
automatically scaled. Validity masks exclude empty projection borders.

Cylindrical projection uses calibrated focal lengths and similarity alignment to
reduce planar stretching. Optional optical-flow refinement is capped at 20 pixels
inside overlap areas. It can improve doubled edges but may distort moving objects
or occlusions. Sequential alignment can accumulate drift; neither mode reconstructs
scene depth or guarantees correction of parallax.

The default blend selects one source per pixel based on distance from valid-image
borders, with a five-pixel transition at the join. This avoids averaging misaligned
objects across the entire overlap. A visible seam may remain where alignment or
exposure differs. For comparison, `WarpToMatch(blend_mode='feather')` retains the
old whole-overlap average. Local alignment remains optional.

Selection uses up to three frames of lookahead after the first eligible candidate
(`ImageMatcher(selection_lookahead=3)`). Candidates need at least 12 inliers, 25%
inlier ratio, inlier-hull coverage of 1% in both images, and RMSE within the RANSAC
threshold. Ranking combines match support, spatial coverage, reprojection error,
spacing near 60% overlap, and a modest sharpness preference. Hard overlap limits
remain unchanged. Lookahead features are cached and rematched when the reference
changes. Progress is summarized; per-candidate diagnostics are in `selection_log`.
This is bounded lookahead, not exhaustive global sequence optimization.

`select_pairs()` caches selection within the matcher instance. Re-running the
analysis cell does not rescan; use `select_pairs(refresh=True)` to request a fresh
selection. `match_selected()` is the separate, unfiltered path for cylindrical
features after the subset has been chosen; it never calls the selector. It also
caches its matches. Rerun notebook setup after changing input or configuration.
`selection_meets_requirement` records whether the four-frame assignment minimum
was met; usable results are not discarded merely because it is false.

If quick lookahead selection yields fewer than four frames, `select_pairs()` now
searches alternative starts and backtracks through different frame sequences.
It uses direct pair geometry and caches each measurement. It stops at the requested
count, or returns the longest compatible sequence under those measurements. A
short greedy result alone is no longer treated as the final selection. The fallback
may evaluate up to N(N-1)/2 pairs and can be slower for long, redundant sequences.
Inspect `search_summary` and `alternative_log` when this fallback runs.

Geometry is shared through functions in `warping/geometry.py`:
`project_corners()`, `validate_warp()`, `canvas_bounds()`, and `panorama_layout()`.
Homography estimation and blending also have standalone functions in
`warping/warp_to_match.py`; the existing classes delegate to them.

Extreme warps are rejected before allocation: projective infinity, flipped or
collapsed footprints, projected area outside 0.25–4x, edge scaling outside
0.25–4x, canvases over 25 million pixels, or dimensions reaching OpenCV's limit.
Oversized results are no longer automatically shrunk. The layout reports retained
and rejected images. Stitching records `rejected_warps`, keeps the last accepted
anchor, and rematches the next candidate to that anchor. Its final image count
reflects only successfully rendered inputs.

The executed rubric-aligned report uses `images_3` by default and includes equations,
OpenCV parameter tables, generated overlap/error results, and observed panorama
issues. `METHODS.md` contains a readable methods companion. `report_helpers.py`
checks the successfully rendered subset, so a rejected candidate is documented
without invalidating verified inputs that actually appear in the result. The
submission bundle includes the executed notebook, code, calibration, input
`images_3` sequence, and generated primary results. Extra credit is not claimed.


The IEEE-style written report is `latex/report.pdf`, with editable source in
`latex/report.tex`. It is authored by Jordan Brown. Build instructions are in
`latex/README.md`; the official IEEEtran class and all figures are bundled.
`outputs/hw3_ieee_report.zip` is the self-contained LaTeX project, while the full
`outputs/hw3_submission.zip` also includes code, input sequence, and notebook.
