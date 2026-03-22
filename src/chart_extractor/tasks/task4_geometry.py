from __future__ import annotations

import re
import numpy as np


def lineIntersectsRectX(candx, rect):
    x, y, w, h = rect
    return x <= candx <= x + w


def lineIntersectsRectY(candy, rect):
    x, y, w, h = rect
    return y <= candy <= y + h


def cleanText(image_text):
    return [(text, (textx, texty, w, h)) for text, (textx, texty, w, h) in image_text if text.strip() != "I"]


def point_line_distance(px, py, x1, y1, x2, y2):
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)


def mean_color_rgb(image_np, rect):
    x, y, w, h = rect
    h = max(1, h)
    w = max(1, w)
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    h_img, w_img = image_np.shape[:2]
    x1 = max(0, min(w_img, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1))
    y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0]
    patch = image_np[y1:y2, x1:x2, :]
    mean = patch.reshape(-1, 3).mean(axis=0)
    return [int(mean[0]), int(mean[1]), int(mean[2])]


def getProbableLabels(image, d, xaxis, yaxis):
    try:
        text_blocks = d["task3"]["input"]["task2_output"]["text_blocks"]
    except KeyError:
        text_blocks = d["task2"]["output"]["text_blocks"]

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

    image_text = cleanText(raw_image_text)
    text_roles = d["task3"]["output"]["text_roles"]
    id_to_role = {item["id"]: item["role"] for item in text_roles}

    tick_blocks, axis_blocks, legend_blocks = [], [], []
    for bid, role in id_to_role.items():
        if bid not in id_to_text:
            continue
        pair = (id_to_text[bid], id_to_rect[bid])
        if role == "tick_label":
            tick_blocks.append(pair)
        elif role == "axis_title":
            axis_blocks.append(pair)
        elif role == "legend_label":
            legend_blocks.append(pair)

    x1, y1, x2, y2 = xaxis
    yx1, yy1, yx2, yy2 = yaxis
    x_tick_list, y_tick_list = [], []
    for text, (tx, ty, w, h) in tick_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0
        side_xaxis = np.sign((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1))
        side_yaxis = np.sign((yx2 - yx1) * (cy - yy1) - (yy2 - yy1) * (cx - yx1))
        if side_yaxis == 1:
            y_tick_list.append((text, (tx, ty, w, h)))
        elif side_xaxis == 1 and side_yaxis == -1:
            x_tick_list.append((text, (tx, ty, w, h)))

    x_title, y_title = [], []
    for text, (tx, ty, w, h) in axis_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0
        if point_line_distance(cx, cy, yx1, yy1, yx2, yy2) < point_line_distance(cx, cy, x1, y1, x2, y2):
            y_title.append((text, (tx, ty, w, h)))
        else:
            x_title.append((text, (tx, ty, w, h)))

    return image, x_tick_list, x_title, y_tick_list, y_title, legend_blocks[:], image_text


def infer_ndigits_from_ticks(y_tick_list, default=1, cap=3):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    decs = []
    for text, _ in y_tick_list:
        nums = re.findall(pattern, text.strip())
        if not nums:
            continue
        s = max(nums, key=len)
        if "e" in s.lower():
            return min(cap, max(default, 2))
        if "." in s:
            frac = s.split(".", 1)[1].split("e", 1)[0].split("E", 1)[0]
            decs.append(len(frac.rstrip("0")))
        else:
            decs.append(0)
    if not decs:
        return default
    return min(cap, max(decs))


def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]


def getRatio_optimized(y_tick_list):
    list_text, list_ticks = [], []
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    for text, (textx, texty, w, h) in y_tick_list:
        numbers = re.findall(pattern, text.strip())
        if not numbers:
            continue
        try:
            val = float(max(numbers, key=len))
            list_text.append(val)
            list_ticks.append(float(texty + h))
        except ValueError:
            continue

    if len(list_text) < 2:
        return sorted(list_text), 0, (0, 0)

    text_sorted = sorted(list_text)
    ticks_sorted = sorted(list_ticks)
    ticks_diff = reject_outliers(np.array([ticks_sorted[i] - ticks_sorted[i - 1] for i in range(1, len(ticks_sorted))]), m=1)
    text_diff = reject_outliers(np.array([text_sorted[i] - text_sorted[i - 1] for i in range(1, len(text_sorted))]), m=1)
    if len(ticks_diff) == 0 or np.array(ticks_diff).mean() == 0:
        return text_sorted, 0, (0, 0)

    normalize_ratio = np.array(text_diff).mean() / np.array(ticks_diff).mean()
    return text_sorted, normalize_ratio, (text_sorted[0], ticks_sorted[0])


def RectDist(rectA, rectB):
    ax, ay, aw, ah = rectA
    bx, by, bw, bh = rectB
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

