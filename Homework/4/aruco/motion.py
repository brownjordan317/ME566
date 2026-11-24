"""Rigid camera poses expressed in the initial camera coordinate frame."""
import numpy as np


def inverse_rigid(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


class CameraMotion:
    def __init__(self, initial_poses, reference_id=None):
        if reference_id is None and initial_poses:
            reference_id = min(initial_poses)
        if reference_id not in initial_poses:
            raise ValueError('The initial image must contain a valid reference marker pose. '
                             'Check calibration, dictionary, marker size, and reference ID.')
        self.reference_id = reference_id
        self.initial = initial_poses[reference_id].transform.copy()

    def estimate(self, poses):
        pose = poses.get(self.reference_id)
        if pose is None:
            return None
        # T_C0_M @ T_M_Ci maps the current camera into the initial camera.
        return self.initial @ inverse_rigid(pose.transform)
