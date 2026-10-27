"""Offline experimental sparse reconstruction; never marks inputs verified.

Uses original RGB images, object mask proposals, candidate intrinsics and
image-derived board poses. Production reconstruction's verification gates are
unchanged. No ROS/serial imports or hardware operations occur here.
"""
import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .common import read_json, sha256, transform, write_json
from .geometry import GuardedGeometry, triangulate_checked
from .masks import validate_mask


def object_features(image, mask, detector_name='orb'):
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, int(xs.min())-16), min(image.shape[1], int(xs.max())+17)
    y0, y1 = max(0, int(ys.min())-16), min(image.shape[0], int(ys.max())+17)
    # Upsampling supports small targets; it does not add camera detail.
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    region = cv2.resize(mask[y0:y1, x0:x1], None, fx=3, fy=3,
                        interpolation=cv2.INTER_NEAREST)
    if detector_name == 'akaze':
        detector = cv2.AKAZE_create(threshold=.0001)
    elif detector_name == 'orb':
        detector = cv2.ORB_create(nfeatures=1500, edgeThreshold=8, patchSize=21,
                                  fastThreshold=5, scaleFactor=1.15)
    else:
        raise ValueError('Supported experimental feature detectors: orb or akaze')
    keypoints, descriptors = detector.detectAndCompute(gray, region)
    pixels = np.asarray([[p.pt[0]/3+x0, p.pt[1]/3+y0] for p in keypoints],
                        np.float64).reshape(-1, 2)
    return pixels, descriptors


def undistort_pixels(pixels, calibration):
    points = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    if not len(points):
        return np.empty((0, 2), np.float64)
    if not calibration.pixel_domain_report(points)['all_pixels_valid']:
        raise ValueError('Feature pixels leave the calibration inverse domain; do not triangulate this view')
    if calibration.model == 'equidistant':
        return cv2.fisheye.undistortPoints(points, calibration.K, calibration.D,
                                          P=calibration.K).reshape(-1, 2)
    return cv2.undistortPoints(points, calibration.K, calibration.D,
                               P=calibration.K).reshape(-1, 2)


def epipolar_inliers(K, first_pose, second_pose, first, second):
    relative = np.linalg.inv(second_pose) @ first_pose
    x, y, z = relative[:3, 3]
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    inverse = np.linalg.inv(K)
    F = inverse.T @ cross @ relative[:3, :3] @ inverse
    a, b = np.c_[first, np.ones(len(first))], np.c_[second, np.ones(len(second))]
    Fa, Ftb = a @ F.T, b @ F
    numerator = np.sum(b*Fa, axis=1)**2
    denominator = (Fa[:, :2]**2).sum(axis=1)+(Ftb[:, :2]**2).sum(axis=1)
    return numerator/np.maximum(denominator, 1e-15) <= 1.


def native_pixels(points, frame, calibration):
    C = np.linalg.inv(frame['pose'])
    camera = points @ C[:3, :3].T + C[:3, 3]
    if calibration.model == 'equidistant':
        normalized = (camera[:, :2]/camera[:, 2:]).reshape(-1, 1, 2)
        raw = cv2.fisheye.distortPoints(normalized, calibration.K, calibration.D)
    else:
        raw = cv2.projectPoints(camera, np.zeros(3), np.zeros(3),
                                calibration.K, calibration.D)[0]
    return raw.reshape(-1, 2), camera[:, 2]


def silhouette_consistent(points, frames, calibration):
    if not len(points):
        return np.empty(0, dtype=bool)
    keep = np.ones(len(points), dtype=bool)
    for frame in frames:
        uv, depth = native_pixels(points, frame, calibration)
        good = np.isfinite(uv).all(axis=1) & (depth > .001)
        indices = np.rint(np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1)).astype(np.int64)
        height, width = frame['mask'].shape
        good &= ((indices[:, 0] >= 0) & (indices[:, 0] < width)
                 & (indices[:, 1] >= 0) & (indices[:, 1] < height))
        expanded = cv2.dilate(frame['mask'], np.ones((5, 5), np.uint8))
        inside = np.zeros(len(points), dtype=bool)
        inside[good] = expanded[indices[good, 1], indices[good, 0]] > 0
        keep &= inside
    return keep


def colors_at_points(points, frame, calibration):
    raw, _ = native_pixels(points, frame, calibration)
    uv = np.rint(raw).astype(int)
    uv[:, 0] = np.clip(uv[:, 0], 0, frame['image'].shape[1]-1)
    uv[:, 1] = np.clip(uv[:, 1], 0, frame['image'].shape[0]-1)
    return frame['image'][uv[:, 1], uv[:, 0]][:, ::-1].copy()


def export_cloud(output, points, colors, title='Experimental object sparse cloud',
                 caption='Partial visible surfaces; calibration, masks and metric accuracy unverified.'):
    with (output / 'experimental_object_cloud.ply').open('w', encoding='ascii') as handle:
        handle.write('ply\nformat ascii 1.0\ncomment %s; scale unverified\n' % title)
        handle.write('element vertex %d\n' % len(points))
        handle.write('property float x\nproperty float y\nproperty float z\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for xyz, rgb in zip(points, colors):
            handle.write('%.8f %.8f %.8f %d %d %d\n' % (*xyz, *rgb))
    np.savez_compressed(str(output / 'experimental_object_cloud.npz'), xyz=points, rgb=colors)
    import matplotlib
    matplotlib.use('Agg')
    from mpl_toolkits.mplot3d import Axes3D  # Registers3d on older Ubuntu matplotlib.
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection='3d')
    if len(points):
        axes.scatter(*points.T, c=colors/255., s=16)
        center = (points.min(axis=0)+points.max(axis=0))/2
        radius = max(float(np.ptp(points, axis=0).max())/2, .005)
        axes.set_xlim(center[0]-radius, center[0]+radius)
        axes.set_ylim(center[1]-radius, center[1]+radius)
        axes.set_zlim(center[2]-radius, center[2]+radius)
    else:
        axes.text2D(.12, .5, 'No consistent 3D points; no object reconstruction', transform=axes.transAxes)
    axes.set_xlabel('Board X (candidate m)')
    axes.set_ylabel('Board Y (candidate m)')
    axes.set_zlabel('Board Z (candidate m)')
    axes.view_init(elev=22, azim=-55)
    axes.set_title('%s: %d points' % (title, len(points)))
    figure.text(.08, .02, caption, fontsize=9)
    figure.savefig(str(output / 'experimental_object_cloud.png'), dpi=160)
    plt.close(figure)


def visual_hull_preview(output, frames, calibration, sparse_points):
    """Exercise the professor's hull branch with explicitly estimated bounds."""
    if len(sparse_points) < 10:
        return {'status': 'insufficient_sparse_seed', 'volume_verified': False}
    center = np.median(sparse_points, axis=0)
    half_extent = max(.03, float(np.ptp(sparse_points, axis=0).max())*.85)
    geometry = None
    view_results = []
    for frame in frames:
        image, mask, pose, K = calibration.rectify_object(frame['image'], frame['mask'], frame['pose'])
        if geometry is None:
            geometry = GuardedGeometry(K, calibration.size, center, half_extent, resolution=48)
        view_results.append(geometry.update_checked(image, mask, pose))
    grid = (geometry.occupied & (geometry.support >= 2)).reshape(48, 48, 48)
    boundary = any(np.any(np.take(grid, edge, axis=axis))
                   for axis in range(3) for edge in (0, -1))
    interior = np.zeros_like(grid)
    interior[1:-1, 1:-1, 1:-1] = (grid[1:-1, 1:-1, 1:-1]
        & grid[:-2, 1:-1, 1:-1] & grid[2:, 1:-1, 1:-1]
        & grid[1:-1, :-2, 1:-1] & grid[1:-1, 2:, 1:-1]
        & grid[1:-1, 1:-1, :-2] & grid[1:-1, 1:-1, 2:])
    surface = geometry.voxels[(grid & ~interior).ravel()]
    candidate_surface_count = len(surface)
    # The original carver leaves unseen voxels occupied and uses a rectified
    # pixel margin. Export only samples consistent with every native RGB mask.
    if len(surface):
        surface = surface[silhouette_consistent(surface, frames, calibration)]
    destination = output / 'experimental_visual_hull'
    destination.mkdir(exist_ok=False)
    colors = colors_at_points(surface, frames[0], calibration) if len(surface) else np.empty((0, 3), np.uint8)
    quality = 'UNRELIABLE: clipped volume' if boundary else 'UNVALIDATED partial hull'
    export_cloud(destination, surface, colors, title='Experimental hull - '+quality,
                 caption='Silhouette intersection, not a recovered solid object. Shape/size unverified.')
    report = {'status': 'experimental_visual_hull' if len(surface) else 'insufficient_hull_support',
              'rectification': 'Object-centred virtual pinhole rotation; physical optical centres unchanged',
              'algorithm': 'Professor RGBPoseGeometry visual hull via NICO GuardedGeometry',
              'volume_source': 'median and span of experimental sparse points, not measured object dimensions',
              'volume_center_candidate_m': center.tolist(), 'half_extent_candidate_m': half_extent,
              'volume_verified': False, 'calibration_verified': False,
              'surface_voxels': len(surface), 'occupied_supported_voxels': int(grid.sum()),
              'candidate_surface_voxels_before_native_mask_check': candidate_surface_count,
              'rejected_surface_voxels_by_native_mask_check': candidate_surface_count-len(surface),
              'surface_check': 'Every exported sample must project inside every native mask, with2px margin',
              'quality_status': 'unreliable_clipped_volume' if boundary else 'unvalidated_partial_geometry',
              'usable_as_validated_object_shape': False,
              'touches_estimated_volume_boundary': bool(boundary),
              'views': view_results, 'complete_object_surface': False,
              'real_robot_motion_commanded': False, 'depth_sensor_used': False,
              'limitation': 'Sparse-derived bounds can truncate or constrain the hull. Narrow views do not reveal hidden faces or concavities. No independent metric accuracy is established.'}
    write_json(destination / 'visual_hull_report.json', report)
    return report


def reconstruct_preview(manifest_path, output, detector_name='orb'):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get('object_static') is not True:
        raise ValueError('A stationary object setup is required even for an experimental preview')
    observations = manifest.get('observations', [])
    if not 2 <= len(observations) <= 12:
        raise ValueError('Need2..12 original images, masks and candidate poses')
    calibration_path = path.parent / manifest['calibration']
    calibration = Calibration.load(calibration_path, require_verified=False)
    if manifest.get('world_frame') is None or manifest.get('units') != 'metres':
        raise ValueError('Need an explicit board frame and candidate metre convention')
    frames, provenance, digests = [], [], set()
    for observation in observations:
        image_path, mask_path = path.parent / observation['image'], path.parent / observation['mask']
        digest = sha256(image_path)
        if digest in digests:
            raise ValueError('Duplicate image; repeated frames do not supply another view')
        digests.add(digest)
        image = cv2.imread(str(image_path))
        if image is None or (image.shape[1], image.shape[0]) != calibration.size:
            raise ValueError('Image unavailable or dimensions differ from calibration')
        if observation['camera_id'] != calibration.record['camera_id']:
            raise ValueError('Camera identity mismatch')
        if float(observation['pose_reprojection_rms_px']) > 1.:
            raise ValueError('Candidate board pose fails unchanged1px image-fit gate')
        pose = transform(observation['T_wc'])
        mask = validate_mask(cv2.imread(str(mask_path), 0), image.shape)
        pixels, descriptors = object_features(image, mask, detector_name)
        frames.append({'image': image, 'mask': mask, 'pose': pose,
                       'pixels': undistort_pixels(pixels, calibration), 'descriptors': descriptors})
        provenance.append({'image': str(image_path), 'image_sha256': digest,
                           'mask': str(mask_path), 'mask_sha256': sha256(mask_path),
                           'mask_reviewed': observation.get('mask_reviewed') is True,
                           'pose_verified': observation.get('pose_verified') is True,
                           'pose_source': observation.get('pose_source'), 'feature_count': len(pixels)})
    all_points, all_colors, pairs = [], [], []
    for i, j in combinations(range(len(frames)), 2):
        first, second = frames[i], frames[j]
        distance = float(np.linalg.norm(first['pose'][:3, 3]-second['pose'][:3, 3]))
        record = {'first_view': i, 'second_view': j, 'candidate_baseline_m': distance}
        pairs.append(record)
        if distance < .01:
            record['status'] = 'baseline_below10mm'
            continue
        a, b = first['descriptors'], second['descriptors']
        if a is None or b is None or min(len(a), len(b)) < 2:
            record['status'] = 'insufficient_features'
            continue
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        forward, backward = matcher.knnMatch(a, b, k=2), matcher.knnMatch(b, a, k=2)
        reverse = {m[0].queryIdx: m[0].trainIdx for m in backward if len(m) == 2 and m[0].distance < .8*m[1].distance}
        matches = [m[0] for m in forward if len(m) == 2 and m[0].distance < .8*m[1].distance
                   and reverse.get(m[0].trainIdx) == m[0].queryIdx]
        record['mutual_ratio_matches'] = len(matches)
        if not matches:
            record['status'] = 'no_consistent_matches'
            continue
        p1 = first['pixels'][[m.queryIdx for m in matches]]
        p2 = second['pixels'][[m.trainIdx for m in matches]]
        good = epipolar_inliers(calibration.K, first['pose'], second['pose'], p1, p2)
        record['epipolar_inliers'] = int(good.sum())
        points = triangulate_checked(calibration.K, first['pose'], second['pose'],
                                     p1[good], p2[good], min_angle_deg=1., max_error_px=2.)
        record.update(status='triangulated' if len(points) else 'no_geometrically_valid_points',
                      triangulated_points=len(points))
        if len(points):
            points = points[silhouette_consistent(points, frames, calibration)]
        record['all_view_silhouette_consistent_points'] = len(points)
        if len(points):
            all_points.append(points)
            all_colors.append(colors_at_points(points, first, calibration))
    points = np.concatenate(all_points) if all_points else np.empty((0, 3))
    colors = np.concatenate(all_colors) if all_colors else np.empty((0, 3), np.uint8)
    raw_count = len(points)
    if len(points):
        _, keep = np.unique(np.floor(points/.001).astype(np.int64), axis=0, return_index=True)
        points, colors = points[keep], colors[keep]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    export_cloud(output, points, colors)
    hull = visual_hull_preview(output, frames, calibration, points)
    report = {'status': 'experimental_sparse_cloud' if len(points) >= 10 else 'insufficient_points',
              'target_object': manifest.get('target_object', 'not specified'),
              'manifest': str(path), 'manifest_sha256': sha256(path),
              'calibration': str(calibration_path), 'calibration_sha256': sha256(calibration_path),
              'world_frame': manifest['world_frame'], 'units': 'candidate_metres_unverified',
              'detector': detector_name, 'views': provenance, 'pairs': pairs,
              'pairwise_points_before_1mm_merge': raw_count, 'point_count': len(points),
              'silhouette_check': 'Must project inside every object mask, with2px segmentation margin',
              'calibration_verified': False, 'verified_for_reconstruction': False,
              'metric_accuracy_verified': False, 'complete_object_surface': False,
              'experimental_visual_hull': hull,
              'object_shape_reconstruction_validated': False,
              'shape_quality': hull.get('quality_status', hull['status']),
              'depth_sensor_used': False, 'real_robot_motion_commanded': False,
              'limitation': 'Experimental sparse triangulation from candidate poses and mask proposals. Repeated patterns or weak texture can cause ambiguous matches. Hidden faces/concavities and full-object dimensions are not recovered.',
              'outputs': {name: str(output/name) for name in ('experimental_object_cloud.ply',
                          'experimental_object_cloud.npz', 'experimental_object_cloud.png')}}
    write_json(output / 'sparse_preview_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='New output folder; never overwrite a previous run')
    parser.add_argument('--detector', choices=['orb', 'akaze'], default='orb')
    args = parser.parse_args(argv)
    try:
        report = reconstruct_preview(args.manifest, args.output, args.detector)
        print('Status:', report['status'], 'Points:', report['point_count'])
        print('Object shape: NOT VALIDATED; hull quality:', report['shape_quality'])
        print('Report:', Path(args.output).resolve() / 'sparse_preview_report.json')
        return 0 if report['status'] == 'experimental_sparse_cloud' else 2
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        print('Experimental reconstruction unavailable:', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
