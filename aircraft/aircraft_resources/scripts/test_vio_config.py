"""Geometry regression checks; run with Python and NumPy from rc-aircraft-image."""
import sys
from pathlib import Path
import numpy as np
from generate_vio_config import calibration

models = Path(sys.argv[1])
for pitch in (0., np.pi/2, -0.3):
    cameras = calibration(models, pitch)
    left, right = [np.array(cameras[f'cam{i}']['T_imu_cam']) for i in range(2)]
    relative = np.linalg.inv(left) @ right
    np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(relative[:3, 3], [.05, 0, 0], atol=1e-12)
    np.testing.assert_allclose(left[:3, :3].T @ left[:3, :3], np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(left[:3, :3]) - 1) < 1e-12
    assert cameras['cam0']['distortion_coeffs'] == [0., 0., 0., 0.]
    assert cameras['cam0']['resolution'] == [1280, 720]
# Analytic forward-facing optical-to-FRD transform and left optical origin.
forward = np.array(calibration(models, 0.)['cam0']['T_imu_cam'])
np.testing.assert_allclose(forward[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-12)
np.testing.assert_allclose(forward[:3, 3], [.175, -.025, .0665], atol=1e-12)
print('SDF calibration: 50 mm baseline, optical axes, IMU rotation, mount translation and pitch verified')
