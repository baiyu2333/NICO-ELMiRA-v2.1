"""Offline native-pixel landmark audit; no cube fitting or hardware access.

Assistant-selected colour junctions and optical-flow tracks are provisional
correspondences, not independent human ground truth. Fixed candidate poses and
intrinsics are never optimized to make these tracks agree.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .calibration import Calibration
from .common import read_json, sha256, transform, write_json
from .sparse_preview import native_pixels, undistort_pixels


def triangulate_native(calibration, poses, pixels, fit_views):
    """Keep every supplied landmark, including unsuccessful geometric results."""
    poses = [transform(p) for p in poses]
    pixels = np.asarray(pixels, dtype=np.float64)
    if (pixels.ndim != 3 or pixels.shape[0] != len(poses)
            or pixels.shape[2] != 2 or not pixels.shape[1]
            or not np.isfinite(pixels).all()):
        raise ValueError('Expected finite native pixels with shape views x landmarks x 2')
    a, b = fit_views
    if a == b or min(a, b) < 0 or max(a, b) >= len(poses) or len(poses) < 3:
        raise ValueError('Need two distinct fit views and at least one held-out view')
    if np.linalg.norm(poses[a][:3, 3]-poses[b][:3, 3]) < 1e-6:
        raise ValueError('No optical-centre baseline')
    pa, pb = (undistort_pixels(pixels[i], calibration) for i in (a, b))
    ca, cb = np.linalg.inv(poses[a]), np.linalg.inv(poses[b])
    homogeneous = cv2.triangulatePoints(calibration.K @ ca[:3],
                                       calibration.K @ cb[:3], pa.T, pb.T)
    if np.any(np.abs(homogeneous[3]) < 1e-10):
        raise ValueError('Triangulation produced points at infinity')
    linear_xyz = (homogeneous[:3]/homogeneous[3]).T
    xyz = []
    for j, initial in enumerate(linear_xyz):
        # Optimize only the point against the two fit images in native pixels;
        # no held-out observations, object size, intrinsics or pose adjustments.
        def residual(point):
            return np.concatenate([native_pixels(point.reshape(1, 3), {'pose': poses[i]}, calibration)[0].ravel()
                                   - pixels[i, j] for i in fit_views])
        result = least_squares(residual, initial, max_nfev=100,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('Native-pixel point refinement did not converge')
        xyz.append(result.x)
    xyz = np.asarray(xyz)
    projected, depths = zip(*(native_pixels(xyz, {'pose': p}, calibration) for p in poses))
    errors = np.linalg.norm(np.asarray(projected)-pixels, axis=2)
    held_out = [i for i in range(len(poses)) if i not in fit_views]
    va, vb = xyz-poses[a][:3, 3], xyz-poses[b][:3, 3]
    cosine = np.sum(va*vb, axis=1)/(np.linalg.norm(va, axis=1)*np.linalg.norm(vb, axis=1))
    return {'fit_views': list(fit_views), 'held_out_views': held_out,
            'xyz_m': xyz.tolist(), 'linear_initial_xyz_m': linear_xyz.tolist(),
            'triangulation_method': 'DLT initialization then native-pixel point-only least squares on the two fit views',
            'native_predictions_px': np.asarray(projected).tolist(),
            'native_errors_px': errors.tolist(),
            'positive_depth_in_all_views': bool((np.asarray(depths) > .001).all()),
            'triangulation_angles_deg': np.degrees(np.arccos(np.clip(cosine, -1, 1))).tolist(),
            'held_out_median_error_px': float(np.median(errors[held_out])),
            'held_out_max_error_px': float(np.max(errors[held_out])),
            'held_out_over_2px_count': int((errors[held_out] > 2).sum()),
            'held_out_observation_count': int(errors[held_out].size)}


def track_junctions(images, masks, annotation):
    """LK uses image appearance only, not epipolar geometry or reconstructed size."""
    reference = int(annotation['reference_view'])
    seeds = np.asarray([p['pixel'] for p in annotation['landmarks']], np.float32).reshape(-1, 1, 2)
    if not np.isfinite(seeds).all():
        raise ValueError('Landmark seeds must be finite')
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    # Small local refinement; preserve seeds as provenance and reject large jumps.
    refined = cv2.cornerSubPix(gray[reference], seeds.copy(), (3, 3), (-1, -1), (3, 50, .001))
    shifts = np.linalg.norm(refined-seeds, axis=2).ravel()
    if np.any(shifts > 2):
        raise ValueError('Reference refinement moved over two pixels; review the annotations')
    centers = []
    for mask in masks:
        yy, xx = np.where(mask > 0)
        if not len(xx):
            raise ValueError('Empty target mask')
        centers.append(np.array([xx.mean(), yy.mean()]))
    pixels, diagnostics = [], []
    settings = dict(winSize=(9, 9), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                    criteria=(3, 50, .001))
    for i in range(len(images)):
        if i == reference:
            q, status, fb, error = refined.copy(), np.ones(len(seeds), bool), np.zeros(len(seeds)), np.zeros(len(seeds))
        else:
            initial = (refined+centers[i]-centers[reference]).astype(np.float32)
            q, forward, error = cv2.calcOpticalFlowPyrLK(gray[reference], gray[i], refined, initial, **settings)
            if q is None:
                raise ValueError('Optical flow unavailable; manually review this view')
            back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[reference], q, refined.copy(), **settings)
            if back is None:
                raise ValueError('Reverse optical flow unavailable')
            fb = np.linalg.norm(back-refined, axis=2).ravel()
            error = error.ravel()
            status = forward.ravel().astype(bool) & reverse.ravel().astype(bool)
        for j, point in enumerate(q.reshape(-1, 2)):
            x, y = np.rint(point).astype(int)
            inside = 0 <= y < masks[i].shape[0] and 0 <= x < masks[i].shape[1] and masks[i][y, x] > 0
            valid = bool(status[j] and fb[j] <= .5 and error[j] <= 15 and inside)
            diagnostics.append({'view': i, 'landmark': annotation['landmarks'][j]['name'],
                                'forward_backward_error_px': float(fb[j]),
                                'mean_lk_patch_error_8bit': float(error[j]), 'accepted': valid})
        pixels.append(q.reshape(-1, 2))
    if not all(row['accepted'] for row in diagnostics):
        raise ValueError('Some tracks failed appearance checks; do not silently discard them')
    return np.asarray(pixels), diagnostics, shifts.tolist()


def run_audit(manifest_path, annotation_path, output):
    manifest_path, annotation_path, output = Path(manifest_path), Path(annotation_path), Path(output)
    root, manifest, annotation = manifest_path.parent, read_json(manifest_path), read_json(annotation_path)
    if len(manifest['observations']) != 7 or len(annotation['landmarks']) != 3:
        raise ValueError('This diagnostic expects seven bilateral views and three reviewed junctions')
    output.mkdir(parents=True, exist_ok=True)
    cal = Calibration.load(root/manifest['calibration'], require_verified=False)
    images, masks, poses = [], [], []
    for row in manifest['observations']:
        for key in ('image', 'mask'):
            if sha256(root/row[key]) != row[key+'_sha256']:
                raise ValueError('Input hash mismatch: '+row[key])
        im, mask = cv2.imread(str(root/row['image'])), cv2.imread(str(root/row['mask']), 0)
        if im is None or mask is None or im.shape[:2] != mask.shape or (im.shape[1], im.shape[0]) != cal.size:
            raise ValueError('Missing or mismatched image/mask')
        images.append(im)
        masks.append(mask)
        poses.append(transform(row['T_wc']))
    pixels, tracks, shifts = track_junctions(images, masks, annotation)
    fit_views = tuple(annotation['fit_views'])
    primary = triangulate_native(cal, poses, pixels, fit_views)
    alternatives = [triangulate_native(cal, poses, pixels, pair) for pair in ((0, 3), (0, 6), (2, 5))]
    # Sensitivity only: do not refit intrinsics/poses or infer confidence in accuracy.
    rng = np.random.RandomState(20261019)
    jitter_xyz, jitter_error = [], []
    for _ in range(300):
        perturbed = pixels.copy()
        for i in fit_views:
            perturbed[i] += rng.normal(0, .5, perturbed[i].shape)
        result = triangulate_native(cal, poses, perturbed, fit_views)
        jitter_xyz.append(result['xyz_m'])
        jitter_error.append(result['held_out_median_error_px'])
    report = {'status': 'object_region_geometry_inconsistent' if primary['held_out_median_error_px'] > 2 else 'provisional_pixel_consistency_only',
              'annotation_source': annotation['annotation_source'],
              'independent_human_ground_truth': False, 'calibration_verified': False,
              'manifest_sha256': sha256(manifest_path), 'annotation_sha256': sha256(annotation_path),
              'calibration_sha256': sha256(root/manifest['calibration']),
              'landmark_names': [p['name'] for p in annotation['landmarks']],
              'reference_refinement_shifts_px': shifts, 'native_observed_pixels': pixels.tolist(),
              'tracking': tracks, 'primary': primary, 'alternative_pairs_not_selected_for_quality': alternatives,
              'jitter_sensitivity': {'distribution': 'independent Gaussian sigma 0.5 native pixel in two fit views only',
                                     'samples': 300, 'random_seed': 20261019,
                                     'xyz_std_mm': (np.std(jitter_xyz, axis=0)*1000).tolist(),
                                     'held_out_median_error_5_50_95_percentiles_px': np.percentile(jitter_error, [5, 50, 95]).tolist(),
                                     'note': 'Pixel perturbation sensitivity only; not a metric accuracy confidence interval.'},
              'object_size_constraint_used': False, 'intrinsics_or_poses_refitted': False,
              'real_robot_motion_commanded': False, 'camera_hardware_accessed': False}
    write_json(output/'corner_audit.json', report)
    with (output/'corner_reprojection.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['view', 'image', 'role', 'landmark', 'observed_x', 'observed_y',
                         'predicted_x', 'predicted_y', 'native_error_px'])
        for i, row in enumerate(manifest['observations']):
            for j, name in enumerate(report['landmark_names']):
                writer.writerow([i, row['image'], 'fit' if i in fit_views else 'held_out', name,
                                 *pixels[i, j], *primary['native_predictions_px'][i][j], primary['native_errors_px'][i][j]])
    render_review(output, images, masks, manifest, report)
    return report


def render_review(output, images, masks, manifest, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), squeeze=False)
    observed = np.asarray(report['native_observed_pixels'])
    predicted = np.asarray(report['primary']['native_predictions_px'])
    colors = ['#ec008c', '#00a6d6', '#ff9900']
    for i, (image, mask) in enumerate(zip(images, masks)):
        ax = axes.flat[i]
        yy, xx = np.where(mask > 0)
        x0, x1, y0, y1 = max(0, xx.min()-7), min(image.shape[1], xx.max()+8), max(0, yy.min()-7), min(image.shape[0], yy.max()+8)
        ax.imshow(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
                  extent=(x0-.5, x1-.5, y1-.5, y0-.5), interpolation='nearest')
        for j, color in enumerate(colors):
            a, b = observed[i, j], predicted[i, j]
            ax.plot(a[0], a[1], 'o', markerfacecolor='none', markeredgecolor=color, markersize=9)
            ax.plot(b[0], b[1], 'x', color=color, markersize=7)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color)
            ax.annotate(str(j+1), a, xytext=(5, 4), textcoords='offset points', color='white')
        role = 'FIT' if i in report['primary']['fit_views'] else 'HELD OUT'
        error = np.median(report['primary']['native_errors_px'][i])
        ax.set_title('View %d: %s; median %.2f px' % (i, role, error), fontsize=11)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.tick_params(labelsize=8)
    axes.flat[7].axis('off')
    axes.flat[7].text(.02, .95,
        'Circle: appearance-tracked pixel\nCross: projection from two-view 3D\n\n1: top colour junction\n2: top/front edge seam\n3: front colour junction\n\nNo cube-size fitting\nNo camera/robot hardware access\nProvisional tracks, not human GT',
        va='top', fontsize=12, linespacing=1.5)
    fig.suptitle('Native-pixel cross-view check: fixed candidate calibration and board poses', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(str(output/'corner_reprojection_review.png'), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = run_audit(args.manifest, args.annotations, args.output)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, 'Offline audit unavailable: %s\n' % exc)
    print('Status:', report['status'])
    print('Held-out median: %.3f native px' % report['primary']['held_out_median_error_px'])
    print('Real robot motion commanded: false')
    print('Report:', Path(args.output).resolve()/'corner_audit.json')


if __name__ == '__main__':
    main()
