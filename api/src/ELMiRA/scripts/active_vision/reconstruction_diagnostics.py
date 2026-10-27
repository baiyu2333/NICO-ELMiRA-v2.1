"""Offline real-data audit and explicitly synthetic viewpoint controls.

No cameras, ROS, serial ports or robot commands. Virtual camera poses below
are mathematical controls only and must never be treated as robot waypoints.
"""
import argparse
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .common import read_json, transform, write_json
from .geometry import baseline_and_angle
from .sparse_preview import silhouette_consistent


def axial_control_views(center, distance=.25):
    center = np.asarray(center, np.float64)
    poses = []
    for axis in np.eye(3):
        for sign in (-1, 1):
            offset = axis*distance*sign
            forward = -offset/distance
            up = np.array([0., -1., 0.]) if abs(forward[1]) < .9 else np.array([0., 0., 1.])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            pose = np.eye(4)
            pose[:3, :3] = np.stack((right, np.cross(forward, right), forward), axis=1)
            pose[:3, 3] = center+offset
            poses.append(transform(pose))
    return poses


def cube_silhouette(K, size, pose, vertices):
    inverse = np.linalg.inv(pose)
    camera = vertices @ inverse[:3, :3].T+inverse[:3, 3]
    if np.any(camera[:, 2] <= .001):
        raise ValueError('Synthetic cube is not fully in front of its virtual camera')
    projected = camera @ K.T
    pixels = np.rint(projected[:, :2]/projected[:, 2:]).astype(np.int32)
    if np.any(pixels < 0) or np.any(pixels[:, 0] >= size[0]) or np.any(pixels[:, 1] >= size[1]):
        raise ValueError('Synthetic cube must fit inside the virtual image')
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pixels), 255)
    return mask


def strict_intersection(K, size, poses, masks, voxels):
    keep = np.ones(len(voxels), dtype=bool)
    for pose, mask in zip(poses, masks):
        inverse = np.linalg.inv(pose)
        camera = voxels @ inverse[:3, :3].T+inverse[:3, 3]
        projected = camera @ K.T
        valid = camera[:, 2] > .001
        pixels = np.zeros((len(voxels), 2), np.int64)
        pixels[valid] = np.rint(projected[valid, :2]/projected[valid, 2:]).astype(np.int64)
        valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < size[0])
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < size[1]))
        inside = np.zeros(len(voxels), dtype=bool)
        inside[valid] = mask[pixels[valid, 1], pixels[valid, 0]] > 0
        keep &= inside
    return keep


def surface_samples(voxels, keep, resolution):
    grid = keep.reshape(resolution, resolution, resolution)
    interior = np.zeros_like(grid)
    interior[1:-1, 1:-1, 1:-1] = (grid[1:-1, 1:-1, 1:-1]
        & grid[:-2, 1:-1, 1:-1] & grid[2:, 1:-1, 1:-1]
        & grid[1:-1, :-2, 1:-1] & grid[1:-1, 2:, 1:-1]
        & grid[1:-1, 1:-1, :-2] & grid[1:-1, 1:-1, 2:])
    return voxels[(grid & ~interior).ravel()]


def cube_observability(poses, K, size, center, half_extent, side_m=.036, resolution=64):
    """Known synthetic cube; no size prior is applied to any real reconstruction."""
    if len(poses) < 2 or not 8 <= resolution <= 80 or not 0 < side_m/2 < half_extent:
        raise ValueError('Need multiple poses, valid grid and volume larger than synthetic cube')
    poses = [transform(p) for p in poses]
    center = np.asarray(center, np.float64)
    vertices = np.array(list(product((-side_m/2, side_m/2), repeat=3)))+center
    line = np.linspace(-half_extent, half_extent, resolution)
    zz, yy, xx = np.meshgrid(line, line, line, indexing='ij')
    voxels = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)+center
    results, surfaces = {}, {}
    for name, sequence in (('same_candidate_views', poses),
                           ('six_virtual_axial_views', axial_control_views(center))):
        masks = [cube_silhouette(K, size, pose, vertices) for pose in sequence]
        keep = strict_intersection(K, size, sequence, masks, voxels)
        points = voxels[keep]
        results[name] = {
            'synthetic_only': True, 'view_count': len(sequence), 'occupied_voxels': len(points),
            'span_mm': (np.ptp(points, axis=0)*1000).tolist() if len(points) else [],
            'touches_bounds': bool(np.any(np.abs(points-center) >= half_extent-1e-6)),
            'robot_motion_commanded': False}
        surfaces[name] = surface_samples(voxels, keep, resolution)
    return results, surfaces


def save_comparison(path, real_points, controls, center, half_extent):
    import matplotlib
    matplotlib.use('Agg')
    from mpl_toolkits.mplot3d import Axes3D  # Registers3d on older matplotlib.
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 5.5))
    groups = [('REAL: unreliable hull', real_points, '#ad4c36'),
              ('SYNTHETIC: same narrow views', controls['same_candidate_views'], '#777777'),
              ('SYNTHETIC: six virtual directions', controls['six_virtual_axial_views'], '#408b70')]
    for index, (label, points, color) in enumerate(groups, 1):
        axes = figure.add_subplot(1, 3, index, projection='3d')
        local = (points-center)*1000
        if len(local):
            take = np.linspace(0, len(local)-1, min(3500, len(local))).astype(int)
            axes.scatter(*local[take].T, s=2, c=color)
        for axis in (axes.set_xlim, axes.set_ylim, axes.set_zlim):
            axis(-half_extent*1000, half_extent*1000)
        axes.set_title(label, fontsize=11)
        axes.set_xlabel('X (mm)'); axes.set_ylabel('Y (mm)'); axes.set_zlabel('Z (mm)')
        axes.view_init(elev=22, azim=-55)
    figure.text(.06, .06, 'Right two panels: known 36 mm SYNTHETIC cube; not real reconstruction or robot viewpoints.', fontsize=10)
    figure.text(.06, .02, 'Same display scale. Real calibration, object dimensions and metric accuracy remain unverified.', fontsize=10)
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


def diagnose(manifest_path, reconstruction, output):
    path, reconstruction, output = Path(manifest_path).resolve(), Path(reconstruction).resolve(), Path(output).resolve()
    manifest = read_json(path)
    calibration = Calibration.load(path.parent/manifest['calibration'], require_verified=False)
    source = read_json(reconstruction/'sparse_preview_report.json')
    hull = source['experimental_visual_hull']
    center = np.asarray(hull['volume_center_candidate_m'])
    half_extent = float(hull['half_extent_candidate_m'])
    frames = [{'pose': transform(ob['T_wc']),
               'mask': cv2.imread(str(path.parent/ob['mask']), 0)} for ob in manifest['observations']]
    poses = [frame['pose'] for frame in frames]
    real = np.load(str(reconstruction/'experimental_visual_hull/experimental_object_cloud.npz'))['xyz']
    valid = silhouette_consistent(real, frames, calibration)
    controls, surfaces = cube_observability(poses, calibration.rectified_K, calibration.size, center, half_extent)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'real_shape_not_validated', 'manifest': str(path), 'reconstruction': str(reconstruction),
        'object_center_source': 'Unverified median of experimental sparse cloud; used for ray-angle diagnostic only',
        'maximum_candidate_object_ray_change_deg': max(baseline_and_angle(a, b, center)[1] for a, b in combinations(poses, 2)),
        'original_surface_samples': len(real), 'samples_consistent_with_all_native_masks_2px': int(valid.sum()),
        'samples_failing_native_mask_check': int((~valid).sum()),
        'hull_touches_estimated_volume_boundary': hull['touches_estimated_volume_boundary'],
        'synthetic_controls': controls, 'synthetic_ground_truth_side_mm': 36,
        'synthetic_control_is_real_object_reconstruction': False,
        'real_object_size_prior_applied': False, 'calibration_verified': False,
        'metric_accuracy_verified': False, 'camera_hardware_accessed': False,
        'real_robot_motion_commanded': False,
        'conclusions': [
            'Even perfect synthetic silhouettes at these candidate views leave a clipped, elongated intersection.',
            'The original exported hull also contains samples inconsistent with some native object masks.',
            'Wider independent directions help in the synthetic control, not proof of real-world accuracy.',
            'Real matches, object-region calibration, mask boundaries and timing still need separate review.']}
    save_comparison(output/'viewpoint_control.png', real, surfaces, center, half_extent)
    write_json(output/'diagnosis.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--reconstruction', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = diagnose(args.manifest, args.reconstruction, args.output)
    print('Real object ray change (candidate):', report['maximum_candidate_object_ray_change_deg'])
    print('Original surface samples failing native masks:', report['samples_failing_native_mask_check'])
    print('Synthetic controls only:', report['synthetic_controls'])
    print('Report:', Path(args.output).resolve()/'diagnosis.json')


if __name__ == '__main__':
    main()
