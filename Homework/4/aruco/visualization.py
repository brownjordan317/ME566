"""Draw marker IDs, boundaries, axes, and motion status."""
import cv2


def annotate(image, corners, ids, poses, matrix, distortion, axis_length,
             motion, reference_id):
    output = image.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)
    for pose in poses.values():
        cv2.drawFrameAxes(output, matrix, distortion, pose.rvec, pose.tvec, axis_length, 2)
    lines = ['Marker axes: X red / Y green / Z blue', f'Reference marker: {reference_id}']
    if motion is None:
        lines.append('Camera motion unavailable: reference pose missing/rejected')
    else:
        x, y, z = motion[:3, 3]
        lines.append(f'Camera position in initial frame (m): {x:+.4f}, {y:+.4f}, {z:+.4f}')
    for i, line in enumerate(lines):
        position = (10, 24 + i * 24)
        cv2.putText(output, line, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
        cv2.putText(output, line, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return output
