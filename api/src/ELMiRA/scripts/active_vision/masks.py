"""Pixel segmentation proposals and explicit human/ground-truth review."""
import cv2
import numpy as np


def validate_mask(mask, shape):
    if mask is None or mask.ndim != 2 or mask.shape != tuple(shape[:2]):
        raise ValueError("Mask must match the exact image height and width")
    if not np.isfinite(mask).all() or not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError("Expected a binary mask (0/1 or 0/255), not a box or probability map")
    binary = (mask > 0).astype(np.uint8) * 255
    count = int(np.count_nonzero(binary))
    if count < 9 or count > binary.size * .8:
        raise ValueError("Empty, tiny or near-full-image mask")
    return binary


def segment(bgr, box, scribbles=None, foreground_point=None, foreground_radius=2):
    """GrabCut box is initialization only. Scribbles use OpenCV's 0/1/2/3 classes."""
    if bgr is None or bgr.ndim != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    x, y, w, h = map(int, box)
    height, width = bgr.shape[:2]
    if x < 1 or y < 1 or w < 3 or h < 3 or x+w >= width or y+h >= height:
        raise ValueError("Box (x,y,width,height) must fit inside the image with background padding")
    labels = np.zeros((height, width), np.uint8)
    mode = cv2.GC_INIT_WITH_RECT
    if foreground_point is not None:
        px, py = map(int, foreground_point)
        radius = int(foreground_radius)
        if radius < 1 or not x+radius <= px < x+w-radius or not y+radius <= py < y+h-radius:
            raise ValueError("Foreground point must lie inside the object box")
        labels[y:y+h, x:x+w] = cv2.GC_PR_FGD
        cv2.circle(labels, (px, py), radius, cv2.GC_FGD, -1)
        mode = cv2.GC_INIT_WITH_MASK
    if scribbles is not None:
        if scribbles.shape != labels.shape or not set(np.unique(scribbles)).issubset({0, 1, 2, 3}):
            raise ValueError("Scribbles must match image and use GrabCut classes 0..3")
        labels = scribbles.astype(np.uint8).copy()
        if not np.any(labels == 1) or not np.any(labels == 0):
            raise ValueError("Scribbles need definite foreground and background")
        mode = cv2.GC_INIT_WITH_MASK
    cv2.setRNGSeed(0)
    cv2.grabCut(bgr, labels, (x, y, w, h), np.zeros((1, 65)), np.zeros((1, 65)), 5, mode)
    mask = validate_mask(np.isin(labels, [1, 3]).astype(np.uint8), bgr.shape)
    if np.count_nonzero(mask) < .10*w*h:
        raise ValueError("GrabCut collapsed to a small fragment; add foreground/background scribbles, do not use as object mask")
    return mask, {"method": "opencv_grabcut", "initial_box_xywh": [x, y, w, h],
                  "foreground_point_xy": list(foreground_point) if foreground_point is not None else None,
                  "foreground_radius": int(foreground_radius) if foreground_point is not None else None,
                  "mask_reviewed": False, "accuracy": "unverified without pixel ground truth",
                  "foreground_pixels": int(np.count_nonzero(mask))}


def mask_metrics(prediction, ground_truth):
    a = validate_mask(prediction, ground_truth.shape) > 0
    b = validate_mask(ground_truth, ground_truth.shape) > 0
    intersection = int(np.count_nonzero(a & b))
    return {"iou": intersection / float(np.count_nonzero(a | b)),
            "dice": 2 * intersection / float(a.sum() + b.sum())}


def overlay(bgr, mask):
    result = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 230, 0), 1)
    return result
