import os
import json
import sys
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from ultralytics import YOLO

import Config

# ==========================================
# 1. CONFIG
# ==========================================
TASK4_CONFIG = Config.returnTestTask4_Config()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ==========================================
# 2. GEOMETRY / TEXT HELPERS
# ==========================================
def clean_text(blocks):
    return [(text, (x, y, w, h)) for text, (x, y, w, h) in blocks if text.strip() != "I"]


def point_line_distance(px, py, x1, y1, x2, y2):
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)


def get_probable_labels(image, doc, xaxis, yaxis):
    """
    Returns:
        image (unchanged),
        x_tick_list: [(text, rect)], rect = (x, y, w, h)
        x_title    : list[str]
        y_tick_list: [(text, rect)]
        y_title    : [(text, rect)]
        legend_text_boxes: [(text, rect)]
        image_text: cleaned text blocks
    """
    try:
        text_blocks = doc["task3"]["input"]["task2_output"]["text_blocks"]
    except KeyError:
        text_blocks = doc["task2"]["output"]["text_blocks"]

    id_to_text = {}
    id_to_rect = {}
    raw_image_text = []

    for block in text_blocks:
        bid = block["id"]
        txt = block["text"]
        poly = block["polygon"]
        xs = [poly["x0"], poly["x1"], poly["x2"], poly["x3"]]
        ys = [poly["y0"], poly["y1"], poly["y2"], poly["y3"]]
        x_min, y_min = min(xs), min(ys)
        w = max(xs) - x_min
        h = max(ys) - y_min
        id_to_text[bid] = txt
        id_to_rect[bid] = (x_min, y_min, w, h)
        raw_image_text.append((txt, (x_min, y_min, w, h)))

    image_text = clean_text(raw_image_text)

    text_roles = doc["task3"]["output"]["text_roles"]
    id_to_role = {item["id"]: item["role"] for item in text_roles}

    tick_blocks = []
    axis_blocks = []
    legend_blocks = []

    for bid, role in id_to_role.items():
        if bid not in id_to_text:
            continue
        text = id_to_text[bid]
        rect = id_to_rect[bid]
        if role == "tick_label":
            tick_blocks.append((text, rect))
        elif role == "axis_title":
            axis_blocks.append((text, rect))
        elif role == "legend_label":
            legend_blocks.append((text, rect))

    (x1, y1, x2, y2) = xaxis
    (yx1, yy1, yx2, yy2) = yaxis

    x_tick_list = []
    y_tick_list = []
    for text, (tx, ty, w, h) in tick_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0
        side_xaxis = np.sign((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1))
        side_yaxis = np.sign((yx2 - yx1) * (cy - yy1) - (yy2 - yy1) * (cx - yx1))
        if side_yaxis == 1:
            y_tick_list.append((text, (tx, ty, w, h)))
        elif side_xaxis == 1 and side_yaxis == -1:
            x_tick_list.append((text, (tx, ty, w, h)))

    x_title = []
    y_title = []
    for text, (tx, ty, w, h) in axis_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0
        dist_to_x = point_line_distance(cx, cy, x1, y1, x2, y2)
        dist_to_y = point_line_distance(cx, cy, yx1, yy1, yx2, yy2)
        if dist_to_y < dist_to_x:
            y_title.append((text, (tx, ty, w, h)))
        else:
            x_title.append(text)

    legend_text_boxes = legend_blocks[:]

    return image, x_tick_list, x_title, y_tick_list, y_title, legend_text_boxes, image_text


def infer_ndigits_from_ticks(y_tick_list, default=1, cap=3):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    decs = []
    for text, _ in y_tick_list:
        text = text.strip()
        nums = re.findall(pattern, text)
        if not nums:
            continue
        s = max(nums, key=len)
        if "e" in s.lower():
            return min(cap, max(default, 2))
        if "." in s:
            frac = s.split(".", 1)[1]
            frac = frac.split("e", 1)[0].split("E", 1)[0]
            decs.append(len(frac.rstrip("0")))
        else:
            decs.append(0)
    if not decs:
        return default
    return min(cap, max(decs))


def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]


def get_ratio_optimized(y_tick_list):
    list_text = []
    list_ticks = []
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    for text, (textx, texty, w, h) in y_tick_list:
        text = text.strip()
        numbers = re.findall(pattern, text)
        if numbers:
            try:
                best_match = max(numbers, key=len)
                val = float(best_match)
                list_text.append(val)
                list_ticks.append(float(texty + h))
            except ValueError:
                continue

    if len(list_text) < 2:
        return sorted(list_text), 0.0, (0.0, 0.0)

    text_sorted = sorted(list_text)
    ticks_sorted = sorted(list_ticks)
    ticks_diff = [ticks_sorted[i] - ticks_sorted[i - 1] for i in range(1, len(ticks_sorted))]
    text_diff = [text_sorted[i] - text_sorted[i - 1] for i in range(1, len(text_sorted))]

    ticks_diff = reject_outliers(np.array(ticks_diff), m=1)
    text_diff = reject_outliers(np.array(text_diff), m=1)

    if len(ticks_diff) == 0 or np.array(ticks_diff).mean() == 0:
        return text_sorted, 0.0, (0.0, 0.0)

    normalize_ratio = np.array(text_diff).mean() / np.array(ticks_diff).mean()
    min_val = text_sorted[0]
    min_pixel = ticks_sorted[0]
    return text_sorted, normalize_ratio, (min_val, min_pixel)


# ==========================================
# 3. LEGEND-TEXT / PATCH MATCHING
# ==========================================
def assign_legend_patches(legend_boxes, patch_rects, y_tol=20, prefer_left=True, max_cost=None):
    """
    legend_boxes: list[(text, (tx, ty, tw, th))]
    patch_rects : list[(x, y, w, h)]
    Returns list same length as legend_boxes with matched patch or None.
    """
    nL = len(legend_boxes)
    nP = len(patch_rects)
    if nL == 0 or nP == 0:
        return [None] * nL

    BIG = 1e6
    cost = np.full((nL, nP), BIG, dtype=np.float32)
    for i, (_, (tx, ty, tw, th)) in enumerate(legend_boxes):
        cx_L = tx + tw / 2.0
        cy_L = ty + th / 2.0
        for j, (x, y, w, h) in enumerate(patch_rects):
            cx_P = x + w / 2.0
            cy_P = y + h / 2.0
            dy = abs(cy_L - cy_P)
            if dy > y_tol:
                continue
            dx = cx_L - cx_P
            if prefer_left:
                if dx <= 0:
                    continue
                dist = dx + 0.3 * dy
            else:
                dist = float(np.hypot(dx, dy))
            cost[i, j] = dist

    assignments = [None] * nL
    used_patches = set()
    for i in range(nL):
        j = int(np.argmin(cost[i]))
        c = float(cost[i, j])
        if c >= BIG or (max_cost is not None and c > max_cost) or j in used_patches:
            continue
        assignments[i] = patch_rects[j]
        used_patches.add(j)
    return assignments


# ==========================================
# 4. EMBEDDING HELPERS
# ==========================================
def shrink_legend_bbox(bbox, img_size, ratio=0.12, min_px=2, max_px=3):
    x1, y1, x2, y2 = bbox
    W, H = img_size
    w = x2 - x1
    h = y2 - y1
    if w <= 2 or h <= 2:
        return x1, y1, x2, y2
    dx = w * ratio
    dy = h * ratio
    max_dx_allowed = max(0, (w - 2) / 2)
    max_dy_allowed = max(0, (h - 2) / 2)
    if max_dx_allowed <= 0 or max_dy_allowed <= 0:
        return x1, y1, x2, y2
    dx = min(max(dx, min_px), max_px, max_dx_allowed)
    dy = min(max(dy, min_px), max_px, max_dy_allowed)
    x1_new = max(0, min(x1 + dx, W - 1))
    x2_new = max(0, min(x2 - dx, W))
    y1_new = max(0, min(y1 + dy, H - 1))
    y2_new = max(0, min(y2 - dy, H))
    if x2_new <= x1_new or y2_new <= y1_new:
        return x1, y1, x2, y2
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def shrink_bar_bbox_vertical(bbox, img_size, ratio_x=0.12, min_px=2, max_px=4, shrink_y_px=0):
    x1, y1, x2, y2 = bbox
    W, H = img_size
    w = x2 - x1
    h = y2 - y1
    if w <= 2 or h <= 2:
        return x1, y1, x2, y2
    dx = w * ratio_x
    max_dx_allowed = max(0, (w - 2) / 2)
    dx = min(max(dx, min_px), max_px, max_dx_allowed) if max_dx_allowed > 0 else 0
    dy = min(shrink_y_px, max(0, (h - 2) / 2))
    x1_new = max(0, min(x1 + dx, W - 1))
    x2_new = max(0, min(x2 - dx, W))
    y1_new = max(0, min(y1 + dy, H - 1))
    y2_new = max(0, min(y2 - dy, H))
    if x2_new <= x1_new or y2_new <= y1_new:
        return x1, y1, x2, y2
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def central_crop_to_size(patch_pil: Image.Image, target_size):
    w_t, h_t = target_size
    w, h = patch_pil.size
    if w_t <= 0 or h_t <= 0:
        return patch_pil
    w_t = min(w_t, w)
    h_t = min(h_t, h)
    left = (w - w_t) // 2
    top = (h - h_t) // 2
    right = left + w_t
    bottom = top + h_t
    return patch_pil.crop((left, top, right, bottom))


def get_backbone(model_type: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "resnet50":
        model_name = "resnet50"
    elif model_type == "clip_vitb32":
        model_name = "vit_base_patch32_clip_224.openai"
    elif model_type == "swin_tiny":
        model_name = "swin_tiny_patch4_window7_224"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return {"model": model, "transform": transform, "device": device}


def extract_patch_embedding(image, bbox, backbone, kind: str = "generic", legend_ref_size=None):
    model = backbone["model"]
    transform = backbone["transform"]
    device = backbone["device"]

    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("image must be a numpy array or PIL.Image")

    W, H = image_pil.size
    x1, y1, x2, y2 = bbox

    if kind == "legend":
        x1, y1, x2, y2 = shrink_legend_bbox((x1, y1, x2, y2), (W, H))
    elif kind == "bar":
        x1, y1, x2, y2 = shrink_bar_bbox_vertical((x1, y1, x2, y2), (W, H))

    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after preprocessing: {(x1, y1, x2, y2)} for image size {(W, H)}")

    patch = image_pil.crop((x1, y1, x2, y2))
    if kind == "bar" and legend_ref_size is not None:
        patch = central_crop_to_size(patch, legend_ref_size)

    patch_tensor = transform(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(patch_tensor)
        feat = feat.view(feat.size(0), -1)
        feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu()


def match_legend_to_bars(legend_embs: torch.Tensor, bar_embs: torch.Tensor):
    if legend_embs.ndim != 2 or bar_embs.ndim != 2:
        raise ValueError("legend_embs and bar_embs must be 2D tensors")
    if legend_embs.size(1) != bar_embs.size(1):
        raise ValueError("Embedding dim mismatch between legend and bar")
    legend_embs = legend_embs.float()
    bar_embs = bar_embs.float()
    sim_matrix = torch.matmul(legend_embs, bar_embs.t())
    scores, indices = torch.max(sim_matrix, dim=1)
    return {
        "legend_to_bar": indices.tolist(),
        "scores": scores.tolist(),
        "similarity_matrix": sim_matrix,
    }


# ==========================================
# 5. DEBUG VISUALIZATION
# ==========================================
def draw_debug_image(
    base_image_rgb: np.ndarray,
    xaxis,
    yaxis,
    legend_patches,
    legend_text_boxes,
    bar_rects,
    x_tick_list,
    save_path: Path,
):
    """Draw axes, legend patches/text, bars, and x-ticks for quick inspection."""
    vis = base_image_rgb.copy()

    # Axes
    cv2.line(vis, (int(xaxis[0]), int(xaxis[1])), (int(xaxis[2]), int(xaxis[3])), (0, 0, 255), 2)
    cv2.line(vis, (int(yaxis[0]), int(yaxis[1])), (int(yaxis[2]), int(yaxis[3])), (0, 0, 255), 2)

    # Legend patches (detected color patches)
    for idx, (x, y, w, h) in enumerate(legend_patches):
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(vis, f"L{idx}", (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Legend text boxes (from Task3 roles)
    for idx, box in enumerate(legend_text_boxes):
        # box may be (text, (x,y,w,h)) or simply (x,y,w,h)
        if isinstance(box, (list, tuple)) and len(box) == 2 and isinstance(box[1], (list, tuple)):
            x, y, w, h = box[1]
        else:
            x, y, w, h = box
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 2)
        cv2.putText(vis, f"T{idx}", (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # Bars
    for idx, (x, y, w, h) in enumerate(bar_rects):
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (255, 165, 0), 2)
        cv2.putText(vis, f"B{idx}", (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA)

    # X tick boxes
    for idx, (text, (x, y, w, h)) in enumerate(x_tick_list):
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
        cv2.putText(vis, f"X{idx}", (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ==========================================
# 6. CORE PIPELINE
# ==========================================
def run_yolo_single(detector, img_path, predict_dir):
    return detector.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.4,
        save=False,
        project=str(predict_dir),
        name="predict",
        device=0,
        verbose=False,
    )[0]


def get_y_values(img_dir, json_dir, detector, backbone):
    img_dir = Path(img_dir)
    json_dir = Path(json_dir)
    predict_dir = Path(TASK4_CONFIG["output_json"])
    predict_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXT]
    image_paths.sort()

    y_value_dict = {}

    for path in image_paths:
        json_path = json_dir / f"{path.stem}.json"
        if not json_path.exists():
            print(f"[WARN] Missing JSON for {path.name}, skip.")
            continue

        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read JSON {json_path}: {e}")
            continue

        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] Cannot read image {path}, skip.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape

        result = run_yolo_single(detector, path, predict_dir)
        boxes = result.boxes
        names = result.names
        detections_by_class = {name: [] for name in names.values()}
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = names[cls_id]
            detections_by_class[cls_name].append({"bbox": [x1, y1, x2, y2], "score": conf})

        legend_dets = detections_by_class.get("legend", [])
        bar_dets = detections_by_class.get("bar", [])
        plot_dets = detections_by_class.get("plot", [])

        if not plot_dets:
            print(f"[WARN] No plot detected for {path.name}, skip.")
            continue

        best_plot = max(plot_dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        x1, y1, x2, y2 = map(int, best_plot["bbox"])
        xaxis = (x1, y2, x2, y2)
        yaxis = (x1, y1, x1, y2)

        image, x_tick_list, x_title, y_tick_list, y_title, legend_text_boxes, image_text = get_probable_labels(
            image, doc, xaxis, yaxis
        )

        _, normalize_ratio, anchor = get_ratio_optimized(y_tick_list)
        if normalize_ratio == 0:
            print(f"[WARN] Cannot compute ratio for {path.name}, skip.")
            continue
        min_val = anchor[0] if anchor else 0.0
        nd = infer_ndigits_from_ticks(y_tick_list, default=1, cap=3)

        legendtexts = []
        legendrects = []
        legend_text_rects = []

        if legend_text_boxes:
            legend_patch_boxes = [(d["bbox"][0], d["bbox"][1], d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) for d in legend_dets]
            assignments = assign_legend_patches(legend_text_boxes, legend_patch_boxes, y_tol=20, prefer_left=True)
            for (text, (tx, ty, tw, th)), patch in zip(legend_text_boxes, assignments):
                if patch is None:
                    continue
                legendtexts.append(text)
                legendrects.append(patch)
                legend_text_rects.append((tx, ty, tw, th))

        if not legendtexts:
            legendtexts = ["series_0"]
            legendrects = []
            legend_text_rects = []

        bar_rects = []
        for det in bar_dets:
            x1, y1, x2, y2 = det["bbox"]
            bar_rects.append((x1, y1, x2 - x1, y2 - y1))

        if not bar_rects:
            print(f"[WARN] No bars detected for {path.name}, skip.")
            continue

        # Build legend reference size
        legend_ref_size = None
        if legendrects:
            w_avg = int(sum(r[2] for r in legendrects) / len(legendrects))
            h_avg = int(sum(r[3] for r in legendrects) / len(legendrects))
            legend_ref_size = (w_avg, h_avg)

        # Embeddings
        legend_embs_list = []
        for (lx, ly, lw, lh) in legendrects:
            bbox_xyxy = (lx, ly, lx + lw, ly + lh)
            try:
                emb = extract_patch_embedding(image, bbox_xyxy, backbone, kind="legend")
                legend_embs_list.append(emb)
            except Exception as e:
                print(f"[WARN] Legend embedding failed for {path.name}: {e}")

        bar_embs_list = []
        for (bx, by, bw, bh) in bar_rects:
            bbox_xyxy = (bx, by, bx + bw, by + bh)
            try:
                emb = extract_patch_embedding(image, bbox_xyxy, backbone, kind="bar", legend_ref_size=legend_ref_size)
                bar_embs_list.append(emb)
            except Exception as e:
                print(f"[WARN] Bar embedding failed for {path.name}: {e}")

        legend_for_bar = None
        if legend_embs_list and bar_embs_list and len(legend_embs_list) == len(legendtexts):
            try:
                legend_embs = torch.stack(legend_embs_list)
                bar_embs = torch.stack(bar_embs_list)
                sim_matrix = match_legend_to_bars(legend_embs, bar_embs)["similarity_matrix"]
                legend_for_bar = torch.argmax(sim_matrix, dim=0).tolist()
            except Exception as e:
                print(f"[WARN] Legend-bar matching failed for {path.name}: {e}")
                legend_for_bar = None

        if legend_for_bar is None:
            legend_for_bar = [0] * len(bar_rects)

        # Debug visualization
        debug_dir = Path(TASK4_CONFIG["output_json"]).parent / "debug_viz_task4"
        debug_path = debug_dir / f"{path.stem}_debug.jpg"
        draw_debug_image(
            base_image_rgb=image,
            xaxis=xaxis,
            yaxis=yaxis,
            legend_patches=legendrects,
            legend_text_boxes=legend_text_rects,
            bar_rects=bar_rects,
            x_tick_list=x_tick_list,
            save_path=debug_path,
        )

        # If no x ticks, generate sequential labels (1..N)
        if not x_tick_list:
            sorted_bars = sorted(bar_rects, key=lambda r: r[0] + r[2] / 2.0)
            x_tick_list = [(str(i + 1), rect) for i, rect in enumerate(sorted_bars)]

        text_boxes = []
        labels = []
        for rect_box in bar_rects:
            min_distance = sys.maxsize
            closest_box = None
            labeltext = None
            for text, text_box in x_tick_list:
                d = abs((rect_box[0] + rect_box[2] / 2.0) - (text_box[0] + text_box[2] / 2.0))
                if d < min_distance:
                    min_distance = d
                    closest_box = text_box
                    labeltext = text
            text_boxes.append(closest_box)
            labels.append(labeltext)

        y_vals = []
        for rect in bar_rects:
            height = float(rect[3])
            val = round(min_val + height * normalize_ratio, nd + 1)
            y_vals.append((rect, val))

        data = {legend_text: {x_label: 0.0 for x_label, _ in x_tick_list} for legend_text in legendtexts}

        for legend_idx, legend_text in enumerate(legendtexts):
            for x_label, box in x_tick_list:
                value = 0.0
                dist = sys.maxsize
                for idx_bar, item in enumerate(y_vals):
                    if legend_for_bar[idx_bar] != legend_idx:
                        continue
                    if labels[idx_bar] == x_label:
                        vx, vy, vw, vh = item[0]
                        cx_bar = vx + vw / 2.0
                        cx_lbl = box[0] + box[2] / 2.0
                        d = abs(cx_lbl - cx_bar)
                        if d < dist:
                            dist = d
                            value = item[1]
                data[legend_text][x_label] = value

        y_value_dict[path.name] = data

    return y_value_dict


# ==========================================
# 6. MAIN
# ==========================================
def main():
    detector = YOLO(TASK4_CONFIG["yolo_weight"])
    backbone = get_backbone("resnet50")

    y_value_dict = get_y_values(TASK4_CONFIG["input_images"], TASK4_CONFIG["input_json"], detector, backbone)

    rows = []
    for img_name, legends_dict in y_value_dict.items():
        for legend, xdict in legends_dict.items():
            for x_label, val in xdict.items():
                rows.append({"image": img_name, "legend": legend, "x_label": x_label, "value": val})

    df = pd.DataFrame(rows)
    excel_path = Path(TASK4_CONFIG["output_excel"])
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(excel_path, index=False, encoding="utf-8-sig")
    print("Saved aggregated results to:", excel_path)

    individual_dir = excel_path.parent / "individual_results"
    individual_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving per-image CSVs to: {individual_dir}")
    for img_name, legends_dict in y_value_dict.items():
        img_rows = []
        for legend, xdict in legends_dict.items():
            for x_label, val in xdict.items():
                img_rows.append({"image": img_name, "legend": legend, "x_label": x_label, "value": val})
        if img_rows:
            sub_df = pd.DataFrame(img_rows)
            safe_name = Path(img_name).stem
            sub_csv_path = individual_dir / f"{safe_name}.csv"
            sub_df.to_csv(sub_csv_path, index=False, encoding="utf-8-sig")
            print(f"  -> {sub_csv_path.name}")


if __name__ == "__main__":
    main()
