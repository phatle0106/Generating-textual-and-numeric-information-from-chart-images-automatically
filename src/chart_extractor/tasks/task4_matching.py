from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment


def euclidean(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


def angle_between(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def assign_legend_patches(legend_boxes, patch_rects, y_tol=20, prefer_left=True, max_cost=None):
    if not legend_boxes or not patch_rects:
        return [None] * len(legend_boxes)

    L, P = len(legend_boxes), len(patch_rects)
    cost = np.full((L, P), 1e6, dtype=np.float32)

    for i, (_, (tx, ty, tw, th)) in enumerate(legend_boxes):
        tcy = ty + th / 2.0
        tcx = tx + tw / 2.0
        for j, (px, py, pw, ph) in enumerate(patch_rects):
            pcy = py + ph / 2.0
            pcx = px + pw / 2.0
            dy = abs(tcy - pcy)
            dx = abs(tcx - pcx)
            c = dy * 3.0 + dx
            if prefer_left and pcx <= tcx:
                c *= 0.7
            if dy > y_tol:
                c *= 1.5
            cost[i, j] = c

    rows, cols = linear_sum_assignment(cost)
    out = [None] * L
    for i, j in zip(rows, cols):
        if max_cost is not None and cost[i, j] > max_cost:
            continue
        out[i] = patch_rects[j]
    return out


def shrink_legend_bbox(bbox_xyxy, image_size):
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw = w * 0.65
    nh = h * 0.65
    nx1, ny1 = cx - nw / 2, cy - nh / 2
    nx2, ny2 = cx + nw / 2, cy + nh / 2
    W, H = image_size
    nx1 = max(0, min(W - 1, nx1))
    nx2 = max(1, min(W, nx2))
    ny1 = max(0, min(H - 1, ny1))
    ny2 = max(1, min(H, ny2))
    return int(nx1), int(ny1), int(nx2), int(ny2)


def shrink_bar_bbox_vertical(bbox_xyxy, image_size):
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw = w * 0.55
    nh = h * 0.55
    nx1, ny1 = cx - nw / 2, cy - nh / 2
    nx2, ny2 = cx + nw / 2, cy + nh / 2
    W, H = image_size
    nx1 = max(0, min(W - 1, nx1))
    nx2 = max(1, min(W, nx2))
    ny1 = max(0, min(H - 1, ny1))
    ny2 = max(1, min(H, ny2))
    return int(nx1), int(ny1), int(nx2), int(ny2)


def central_crop_to_size(patch_pil: Image.Image, target_size):
    tw, th = target_size
    w, h = patch_pil.size
    left = max(0, (w - tw) // 2)
    top = max(0, (h - th) // 2)
    right = min(w, left + tw)
    bottom = min(h, top + th)
    return patch_pil.crop((left, top, right, bottom))


def match_legend_to_bars(legend_embs: torch.Tensor, bar_embs: torch.Tensor):
    l = F.normalize(legend_embs, dim=1)
    b = F.normalize(bar_embs, dim=1)
    sim = l @ b.T
    return {"similarity_matrix": sim}

