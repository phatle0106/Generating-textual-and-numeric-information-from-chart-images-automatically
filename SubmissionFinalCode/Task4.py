import os
import json
import sys
import math
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import torch.nn as nn

# Import Config chung của dự án
import Config

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
TASK4_CONFIG = Config.returnTestTask4_Config()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 2. HÀM ĐỌC INPUT
# ==========================================
def lineIntersectsRectX(candx, rect):
    (x, y, w, h) = rect
    return x <= candx <= x + w

def lineIntersectsRectY(candy, rect):
    (x, y, w, h) = rect
    return y <= candy <= y + h

def cleanText(image_text):
    return [(text, (textx, texty, w, h)) for text, (textx, texty, w, h) in image_text if text.strip() != 'I']

def point_line_distance(px, py, x1, y1, x2, y2):
    """
    Khoảng cách từ điểm (px, py) đến đường thẳng đi qua (x1, y1) - (x2, y2)
    """
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / \
           np.hypot(y2 - y1, x2 - x1)


def mean_color_rgb(image_np, rect):
    """
    Compute mean RGB color of a rect (x, y, w, h) on an RGB numpy image.
    Returns [r, g, b] ints.
    """
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
    """
    image: ảnh RGB (H, W, 3)
    d    : object json đã load bằng json.loads(...)
    xaxis, yaxis: tuple (x1, y1, x2, y2)

    Yêu cầu:
    - d["task3"]["input"]["task2_output"]["text_blocks"]
    - d["task3"]["output"]["text_roles"] với role: tick_label, axis_title, legend_label
    """

    # ===== 1. Lấy text_blocks từ task3.input.task2_output =====
    try:
        text_blocks = d["task3"]["input"]["task2_output"]["text_blocks"]
    except KeyError:
        # fallback đơn giản nếu cấu trúc khác (có thể chỉnh tuỳ dataset)
        text_blocks = d["task2"]["output"]["text_blocks"]

    # Map id -> (text, rect) và xây list image_text gốc
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

    # image_text cuối cùng (giống format cũ, sau khi clean)
    image_text = cleanText(raw_image_text)

    # ===== 2. Lấy text_roles từ task3.output =====
    text_roles = d["task3"]["output"]["text_roles"]
    id_to_role = {item["id"]: item["role"] for item in text_roles}

    # ===== 3. Gom theo role: tick_label, axis_title, legend_label =====
    tick_blocks   = []  # [(text, rect), ...]
    axis_blocks   = []  # [(text, rect), ...]
    legend_blocks = []  # [(text, rect), ...]

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
        else:
            # các role khác (nếu có) tạm thời bỏ qua
            pass

    # ===== 4. Chia tick_label thành Y tick và X tick bằng cross product (logic cũ) =====
    (x1,  y1,  x2,  y2)  = xaxis
    (yx1, yy1, yx2, yy2) = yaxis

    x_tick_list = []  # [(text, rect), ...]
    y_tick_list = []  # [(text, rect), ...]
    # x_labels = []   # [text, ...]
    # y_labels = []   # [text, ...]

    for text, (tx, ty, w, h) in tick_blocks:
        # dùng tâm bbox để ổn định hơn
        cx = tx + w / 2.0
        cy = ty + h / 2.0

        side_xaxis = np.sign((x2  - x1)  * (cy - y1)  - (y2  - y1)  * (cx - x1))
        side_yaxis = np.sign((yx2 - yx1) * (cy - yy1) - (yy2 - yy1) * (cx - yx1))

        # Giữ đúng logic phân vùng như code cũ:

        # To the left of y-axis and top of x-axis -> Y tick
        if side_yaxis == 1:
            y_tick_list.append((text, (tx, ty, w, h)))
            # y_labels.append(text)

        # To the right of y-axis and bottom of x-axis -> X tick
        elif side_xaxis == 1 and side_yaxis == -1:
            x_tick_list.append((text, (tx, ty, w, h)))
            # x_labels.append(text)

        # Các trường hợp còn lại (side_xaxis/side_yaxis = 0 hoặc vùng khác) bỏ qua

    # ===== 5. Chia axis_title thành x_text (tiêu đề trục X) và y_text_list (tiêu đề/mô tả trục Y) =====
    x_title = []    # list string
    y_title = []    # list (text, rect)

    for text, (tx, ty, w, h) in axis_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0

        dist_to_x = point_line_distance(cx, cy, x1, y1, x2, y2)
        dist_to_y = point_line_distance(cx, cy, yx1, yy1, yx2, yy2)

        # Nếu gần trục Y hơn -> coi là y_title (bạn vẫn trả về dưới dạng (text, rect))
        if dist_to_y < dist_to_x:
            y_title.append((text, (tx, ty, w, h)))
        else:
            # Gần trục X hơn -> x_title
            x_title.append((text, (tx, ty, w, h)))

    # ===== 6. Legend: dùng trực tiếp legend_label =====
    legend_text_boxes = legend_blocks[:]  # giữ dạng [(text, rect), ...] cho tương thích output cũ

    # ===== 7. Trả về đúng format cũ =====
    return (
        image,
        x_tick_list,
        x_title,
        y_tick_list,
        y_title,
        legend_text_boxes,
        image_text,
    )

# ==========================================
# 3. HÀM TÍNH RATIO
# ==========================================
def infer_ndigits_from_ticks(y_tick_list, default=1, cap=3):
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    decs = []
    for text, _ in y_tick_list:
        text = text.strip()
        nums = re.findall(pattern, text)
        if not nums:
            continue
        s = max(nums, key=len)

        # scientific notation -> thường nên hạn chế, dùng 2-3 chữ số là đủ
        if 'e' in s.lower():
            return min(cap, max(default, 2))

        if '.' in s:
            frac = s.split('.', 1)[1]
            frac = frac.split('e', 1)[0].split('E', 1)[0]
            decs.append(len(frac.rstrip('0')))
        else:
            decs.append(0)

    if not decs:
        return default
    return min(cap, max(decs))

def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]

def getRatio_optimized(y_tick_list):
    list_text = []
    list_ticks = []

    # 1. Trích xuất giá trị số và tọa độ Y
    for text, (textx, texty, w, h) in y_tick_list: # Hoặc y_labels_list
        text = text.strip()

        # --- BẮT ĐẦU SỬA ĐỔI ---
        # Loại bỏ các ký tự gây nhiễu nếu cần (ví dụ dấu phẩy hàng nghìn)
        # text = text.replace(',', '')

        # Regex mới bao quát mọi trường hợp
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        numbers = re.findall(pattern, text)

        if bool(numbers):
            # Hàm float() của Python tự động hiểu '1.0E-03' là 0.001
            # nên không cần xử lý chuỗi thủ công thêm nữa.
            try:
                # Lấy chuỗi khớp dài nhất hoặc đầu tiên
                # Đôi khi OCR ra "Ver 1.0E-02", regex bắt được 2 số, ưu tiên số có định dạng phức tạp
                best_match = max(numbers, key=len)
                val = float(best_match)

                list_text.append(val)
                # Logic lấy tọa độ tick giữ nguyên
                list_ticks.append(float(texty + h))
            except ValueError:
                continue # Bỏ qua nếu không convert được

    # Nếu không đủ điểm để tính khoảng cách (ít nhất 2 điểm)
    if len(list_text) < 2:
        return sorted(list_text), 0, (0, 0) # Hoặc handle lỗi tùy ý

    # 2. Sắp xếp (Sort)
    # Cần sort để đảm bảo tính khoảng cách giữa các điểm liền kề
    text_sorted = sorted(list_text)
    ticks_sorted = sorted(list_ticks)

    # 3. Tính Delta (Khoảng cách)
    ticks_diff = [ticks_sorted[i] - ticks_sorted[i-1] for i in range(1, len(ticks_sorted))]
    text_diff = [text_sorted[i] - text_sorted[i-1] for i in range(1, len(text_sorted))]

    # 4. Loại bỏ ngoại lai (Vẫn cần thiết!)
    # Dù đã phân loại tick, nhưng tọa độ bounding box có thể bị lệch 1-2 pixel
    # hoặc OCR nhận diện sai số (ví dụ: 10 thành 18), nên vẫn cần reject_outliers
    ticks_diff = reject_outliers(np.array(ticks_diff), m=1)
    text_diff = reject_outliers(np.array(text_diff), m=1)

    # 5. Tính Ratio
    if len(ticks_diff) == 0 or np.array(ticks_diff).mean() == 0:
        return text_sorted, 0 # Tránh chia cho 0

    normalize_ratio = np.array(text_diff).mean() / np.array(ticks_diff).mean()

    # Lấy giá trị nhỏ nhất tìm thấy làm điểm neo (anchor)
    min_val = text_sorted[0]
    min_pixel = ticks_sorted[0]     # Lưu ý: ticks_sorted cần khớp chiều với text

    return text_sorted, normalize_ratio, (min_val, min_pixel)

# ==========================================
# 4. SAVE RESULT
# ==========================================
def build_task_outputs(yValueDict: dict, axes_store: dict, legend_store: dict):
    """
    Build task4/5/6 JSON outputs from intermediate stores.
    - task4: axes (plot bbox and axis endpoints)
    - task5: legend_pairs (legend text -> color)
    - task6: data series (values per legend/x_label)
    """
    task4_list = []
    task5_list = []
    task6_list = []

    for img_name, data in yValueDict.items():
        # task4: axes
        ax_entry = axes_store.get(img_name, {})
        task4_list.append({
            "image": img_name,
            "task4": {
                "output": {
                    "_plot_bb": ax_entry.get("_plot_bb", {}),
                    "axes": ax_entry.get("axes", {}),
                }
            }
        })

        # task5: legend_pairs
        lg_entry = legend_store.get(img_name, [])
        task5_list.append({
            "image": img_name,
            "task5": {
                "output": {
                    "legend_pairs": lg_entry
                }
            }
        })

        # task6: data series
        series = []
        legends_dict = data
        for legend, legend_data in legends_dict.items():
            if isinstance(legend_data, dict):
                for x_label, val in legend_data.items():
                    series.append({
                        "legend_label": legend,
                        "x_label": x_label,
                        "value": float(val),
                    })
            else:
                series.append({
                    "legend_label": legend,
                    "x_label": "Value",
                    "value": float(legend_data),
                })

        task6_list.append({
            "image": img_name,
            "task6": {
                "output": {
                    "data series": series,
                    "visual elements": []
                }
            }
        })

    return {
        "task4_outputs": task4_list,
        "task5_outputs": task5_list,
        "task6_outputs": task6_list,
    }


def save_results(df: pd.DataFrame,
                 yValueDict: dict,
                 axes_store: dict,
                 legend_store: dict,
                 excel_path: str | Path,
                 json_path: str | Path):
    excel_path = Path(excel_path)
    json_path  = Path(json_path)

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Save CSV (matches pipeline expectation) ----
    df.to_csv(excel_path, index=False, encoding="utf-8-sig")

    # ---- 2. Save JSON ----
    # JSON sẽ chứa cả raw nested + bản records (dễ load lại thành DataFrame)

    if json_path.suffix == '':   # Nếu là thư mục (không có đuôi .json)
        json_path.mkdir(parents=True, exist_ok=True)
        json_path = json_path / "task4_results.json"   # Tự gán tên file mặc định
    else:
        json_path.parent.mkdir(parents=True, exist_ok=True)

    tasks_payload = build_task_outputs(yValueDict, axes_store, legend_store)

    payload = {
        "meta": {
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        },
        "yValueDict": yValueDict,
        "records": df.to_dict(orient="records"),
        "task_outputs": tasks_payload,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(">> Saved Excel:", str(excel_path))
    print(">> Saved JSON :", str(json_path))

    # Additionally save per-image JSONs for evaluation (task4/5/6 at root)
    for t4, t5, t6 in zip(tasks_payload["task4_outputs"], tasks_payload["task5_outputs"], tasks_payload["task6_outputs"]):
        img_name = t4.get("image") or t5.get("image") or t6.get("image")
        if not img_name:
            continue
        per_img = {
            "task4": t4.get("task4", {}),
            "task5": t5.get("task5", {}),
            "task6": t6.get("task6", {}),
        }
        out_path = json_path.parent / img_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as pf:
            json.dump(per_img, pf, ensure_ascii=False, indent=2)

    # Save per-image CSVs for app visualization
    individual_dir = excel_path.parent / "individual_results"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for img_name, legends_dict in yValueDict.items():
        rows = []
        for legend, legend_data in legends_dict.items():
            if isinstance(legend_data, dict):
                for x_label, val in legend_data.items():
                    rows.append({"image": img_name, "legend": legend, "x_label": x_label, "value": float(val)})
            else:
                rows.append({"image": img_name, "legend": legend, "x_label": "Value", "value": float(legend_data)})
        df_img = pd.DataFrame(rows)
        stem = Path(img_name).stem
        df_img.to_csv(individual_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")


# ==========================================
# 5. HELPER FUNCTION
# ==========================================
def euclidean(v1, v2):
    return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5

def angle_between(p1, p2):
    deltaX = p1[0] - p2[0]
    deltaY = p1[1] - p2[1]

    return math.atan2(deltaY, deltaX) / math.pi * 180

def RectDist(rectA, rectB):
    (rectAx, rectAy, rectAw, rectAh) = rectA
    (rectBx, rectBy, rectBw, rectBh) = rectB

    return abs(rectAx + rectAx / 2 - rectBx - rectBx / 2)

# ==========================================
# 6. ASSIGN LEGEND PATCHES
# ==========================================
BIG_COST = 1e6

def assign_legend_patches(
    legend_boxes,
    patch_rects,
    y_tol=20,           # dung sai lệch theo Y (px) giữa tâm legend & patch
    prefer_left=True,
    max_cost=None,      # nếu None: nhận mọi cặp hợp lệ; nếu số: bỏ cặp cost > max_cost
):
    """
    legend_boxes: list[(text, (tx, ty, tw, th))]
    patch_rects : list[(x, y, w, h)] từ YOLO

    Trả về:
        mapping: list length = len(legend_boxes)
                 mỗi phần tử là (x, y, w, h) hoặc None nếu không có patch hợp lệ.
    """
    nL = len(legend_boxes)
    nP = len(patch_rects)

    if nL == 0 or nP == 0:
        return [None] * nL

    # Ma trận cost (nLegend x nPatch)
    cost = np.full((nL, nP), BIG_COST, dtype=np.float32)

    for i, (_, (tx, ty, tw, th)) in enumerate(legend_boxes):
        cx_L = tx + tw / 2.0
        cy_L = ty + th / 2.0

        for j, (x, y, w, h) in enumerate(patch_rects):
            cx_P = x + w / 2.0
            cy_P = y + h / 2.0

            dy = abs(cy_L - cy_P)

            # Ràng buộc cùng 'dòng': nếu lệch Y quá xa, coi như không hợp lệ
            if dy > y_tol:
                continue  # giữ cost = BIG_COST

            dx = cx_L - cx_P

            if prefer_left:
                # Mặc định: patch nằm bên trái legend (cx_P < cx_L)
                if dx <= 0:
                    # patch ở bên phải hoặc trùng X với legend → không hợp lệ
                    continue

                # cost: khoảng cách theo X + một chút phạt theo Y
                dist = dx + 0.3 * dy
            else:
                # Nếu không áp ràng buộc trái/phải, dùng đoạn này:
                dist = float(np.hypot(dx, dy))

            cost[i, j] = dist

    # Giải bài toán gán tối ưu (tối thiểu tổng cost)
    row_ind, col_ind = linear_sum_assignment(cost)

    # mapping[i] = patch_rects[j] hoặc None
    mapping = [None] * nL
    for i, j in zip(row_ind, col_ind):
        c = float(cost[i, j])

        # Loại các cặp cost "giả" hoặc quá lớn
        if c >= BIG_COST:
            continue
        if max_cost is not None and c > max_cost:
            continue

        mapping[i] = patch_rects[j]

    return mapping

# ==========================================
# 7. ASSIGN LEGEND FOR BAR
# ==========================================

def shrink_legend_bbox(
    bbox,
    img_size,
    ratio: float = 0.12,
    min_px: int = 2,
    max_px: int = 3,
):
    """
    Shrink bbox legend theo cả 2 chiều, tỉ lệ ~0.1-0.15.
    Mỗi cạnh co vào từ 1 đến 3 pixel (nếu kích thước đủ lớn),
    để tránh dính viền, text.

    bbox: (x1, y1, x2, y2)
    img_size: (W, H) của ảnh gốc
    """
    x1, y1, x2, y2 = bbox
    W, H = img_size
    w = x2 - x1
    h = y2 - y1

    if w <= 2 or h <= 2:
        # quá nhỏ, không shrink
        return x1, y1, x2, y2

    # tính số pixel co vào theo tỉ lệ
    dx = w * ratio
    dy = h * ratio

    # clamp vào [min_px, max_px], nhưng không làm bbox âm
    # đảm bảo còn ít nhất 2px chiều rộng/chiều cao
    max_dx_allowed = max(0, (w - 2) / 2)
    max_dy_allowed = max(0, (h - 2) / 2)

    if max_dx_allowed <= 0 or max_dy_allowed <= 0:
        return x1, y1, x2, y2

    dx = min(max(dx, min_px), max_px, max_dx_allowed)
    dy = min(max(dy, min_px), max_px, max_dy_allowed)

    x1_new = x1 + dx
    x2_new = x2 - dx
    y1_new = y1 + dy
    y2_new = y2 - dy

    # clamp trong image
    x1_new = max(0, min(x1_new, W - 1))
    x2_new = max(0, min(x2_new, W))
    y1_new = max(0, min(y1_new, H - 1))
    y2_new = max(0, min(y2_new, H))

    if x2_new <= x1_new or y2_new <= y1_new:
        # nếu co quá, quay lại bbox gốc
        return x1, y1, x2, y2

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def shrink_bar_bbox_vertical(
    bbox,
    img_size,
    ratio_x: float = 0.12,
    min_px: int = 2,
    max_px: int = 4,
    shrink_y_px: int = 0,
):
    """
    Shrink bbox cho BAR (biểu đồ cột đứng):
    - Chủ yếu co theo trục ngang (x).
    - Trục dọc (y) chỉ co rất ít (mặc định 1px mỗi cạnh),
      để tránh dính trục/tick một chút mà không làm mất thân cột.

    bbox: (x1, y1, x2, y2)
    img_size: (W, H)
    """
    x1, y1, x2, y2 = bbox
    W, H = img_size
    w = x2 - x1
    h = y2 - y1

    if w <= 2 or h <= 2:
        return x1, y1, x2, y2

    # shrink theo x giống legend
    dx = w * ratio_x
    max_dx_allowed = max(0, (w - 2) / 2)
    if max_dx_allowed > 0:
        dx = min(max(dx, min_px), max_px, max_dx_allowed)
    else:
        dx = 0

    # shrink rất ít theo y (mặc định 1px mỗi cạnh)
    dy = min(shrink_y_px, max(0, (h - 2) / 2))

    x1_new = x1 + dx
    x2_new = x2 - dx
    y1_new = y1 + dy
    y2_new = y2 - dy

    # clamp trong image
    x1_new = max(0, min(x1_new, W - 1))
    x2_new = max(0, min(x2_new, W))
    y1_new = max(0, min(y1_new, H - 1))
    y2_new = max(0, min(y2_new, H))

    if x2_new <= x1_new or y2_new <= y1_new:
        return x1, y1, x2, y2

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def central_crop_to_size(patch_pil: Image.Image, target_size):
    """
    Central crop patch PIL về kích thước target_size (w_t, h_t),
    nhưng không vượt quá kích thước patch.

    target_size: (w_target, h_target)
    """
    w_t, h_t = target_size
    w, h = patch_pil.size

    if w_t <= 0 or h_t <= 0:
        return patch_pil

    # không crop vượt quá ảnh
    w_t = min(w_t, w)
    h_t = min(h_t, h)

    left = (w - w_t) // 2
    top = (h - h_t) // 2
    right = left + w_t
    bottom = top + h_t

    return patch_pil.crop((left, top, right, bottom))

# ==========================================
# 6. MAIN WORKFLOW
# ==========================================
def run_inference(img_paths, output_dir, detector, device="auto", batch_size=2, imgsz=640):
    """
    Streamed inference over image list in small batches to avoid high memory.
    Yields (path, result) per image.
    """
    print(f"Running inference on images in: {img_paths if isinstance(img_paths, str) else 'list'}")
    dev_flag = 0 if device and device.lower().startswith("cuda") else ("cpu" if device == "cpu" else None)
    half = dev_flag == 0  # use half on CUDA

    # chunk the list to avoid loading all at once
    paths = list(img_paths) if not isinstance(img_paths, (str, os.PathLike)) else [img_paths]
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        results_gen = detector.predict(
            source=chunk,
            imgsz=imgsz,
            conf=0.25,
            iou=0.5,
            save=True,
            project=output_dir,
            name="predict",
            device=dev_flag,
            stream=True,
            half=half,
        )
        for path, res in zip(chunk, results_gen):
            yield path, res
    print("Inference done!")

# 1. Hàm lấy backbone + transform
class FeatureMapMeanPool(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]          # [B, C, H, W]
        return x.mean(dim=(2, 3))     # [B, C]

def get_backbone(model_type: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None

    # -----------------------------
    # RESNET50 (tầng giữa)
    # -----------------------------
    if model_type == "resnet50":
        base_model = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True,
            out_indices=(1,)   # stage 2 (0-based) => nhạy low/mid hơn layer sâu
        )
        model = FeatureMapMeanPool(base_model)
        data_config = resolve_model_data_config(base_model)

    # -----------------------------
    # EFFICIENTNET (option A: embedding cuối)
    # -----------------------------
    elif model_type == "efficientnet_b0":
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0       # trả về embedding vector (global pooled)
        )
        data_config = resolve_model_data_config(model)

    elif model_type == "efficientnet_b1":
        model = timm.create_model(
            "efficientnet_b1",
            pretrained=True,
            num_classes=0
        )
        data_config = resolve_model_data_config(model)

    # -----------------------------
    # EFFICIENTNET (option B: feature map tầng giữa)
    # Bạn gọi bằng model_type = "efficientnet_b0_mid"
    # -----------------------------
    elif model_type == "efficientnet_b0_mid":
        base_model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=(2,)    # thử (1,) hoặc (2,) tùy bạn muốn nông/sâu
        )
        model = FeatureMapMeanPool(base_model)
        data_config = resolve_model_data_config(base_model)

    elif model_type == "efficientnet_b1_mid":
        base_model = timm.create_model(
            "efficientnet_b1",
            pretrained=True,
            features_only=True,
            out_indices=(2,)
        )
        model = FeatureMapMeanPool(base_model)
        data_config = resolve_model_data_config(base_model)

    # -----------------------------
    # CÁC MODEL KHÁC
    # -----------------------------
    elif model_type == "clip_vitb32":
        model = timm.create_model(
            "vit_base_patch32_clip_224.openai",
            pretrained=True,
            num_classes=0
        )
        data_config = resolve_model_data_config(model)

    elif model_type == "swin_tiny":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        data_config = resolve_model_data_config(model)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    model.to(device)

    transform = create_transform(**data_config, is_training=False)

    return {
        "model": model,
        "transform": transform,
        "device": device,
    }

# 2. Hàm trích embedding cho một patch (legend/bar)
def extract_patch_embedding(
    image,
    bbox,
    backbone,
    kind: str = "generic",
    legend_ref_size: tuple[int, int] | None = None,
):
    """
    Trích embedding cho một patch được xác định bởi bbox, có tiền xử lý riêng
    cho legend/bar trong biểu đồ cột đứng.

    Parameters
    ----------
    image : np.ndarray hoặc PIL.Image
        Ảnh RGB gốc (H, W, 3). Nếu là numpy sẽ được convert sang PIL.Image.
    bbox : tuple (x1, y1, x2, y2)
        Toạ độ bounding box trong hệ toạ độ ảnh gốc (pixel).
    backbone : dict
        Kết quả trả về từ get_backbone().
    kind : str
        "legend"  -> shrink bbox legend (0.1-0.15, 1-3 px mỗi cạnh).
        "bar"     -> shrink bbox theo trục ngang, trục dọc chỉ 1px.
        "generic" -> dùng bbox như cũ, không tiền xử lý.
    legend_ref_size : (w_leg, h_leg) hoặc None
        Nếu kind="bar" và legend_ref_size != None, sẽ central crop bar patch
        về kích thước gần bằng legend trước khi resize (giảm vấn đề scale).

    Returns
    -------
    emb : torch.Tensor
        Vector embedding đã được L2-normalize, shape = (D,).
        (ở CPU để dễ xử lý tiếp).
    """
    model = backbone["model"]
    transform = backbone["transform"]
    device = backbone["device"]

    # Chuyển ảnh sang PIL nếu đang là numpy
    if isinstance(image, np.ndarray):
        # Giả sử image là RGB (H, W, 3)
        image_pil = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("image must be a numpy array or PIL.Image")

    W, H = image_pil.size  # PIL: (width, height)
    x1, y1, x2, y2 = bbox

    # Áp dụng tiền xử lý bbox tùy loại patch
    if kind == "legend":
        x1, y1, x2, y2 = shrink_legend_bbox((x1, y1, x2, y2), (W, H))
    elif kind == "bar":
        x1, y1, x2, y2 = shrink_bar_bbox_vertical((x1, y1, x2, y2), (W, H))
    else:
        # generic: giữ nguyên bbox
        pass

    # Clamp trong phạm vi ảnh (phòng trường hợp shrink vượt biên)
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after preprocessing: {(x1, y1, x2, y2)} for image size {(W, H)}")

    # Crop patch đầu tiên
    patch = image_pil.crop((x1, y1, x2, y2))

    # Nếu là bar và có legend_ref_size -> central crop theo size legend
    if kind == "bar" and legend_ref_size is not None:
        patch = central_crop_to_size(patch, legend_ref_size)

    # Áp dụng transform (resize, normalize, ...), ra tensor CHW
    patch_tensor = transform(patch)  # shape: (C, H, W)
    patch_tensor = patch_tensor.unsqueeze(0).to(device)  # shape: (1, C, H, W)

    # Forward qua model để lấy feature
    with torch.no_grad():
        feat = model(patch_tensor)  # shape: (1, D) hoặc (1, D, ...)
        # Nếu model trả ra nhiều chiều hơn (VD: B, D, 1, 1) -> flatten
        feat = feat.view(feat.size(0), -1)  # (1, D)
        # L2-normalize để dùng cosine similarity
        feat = F.normalize(feat, p=2, dim=1)

    emb = feat.squeeze(0).cpu()  # (D,)
    return emb

# 3. Hàm ghép legend -> bar theo cosine similarity
def match_legend_to_bars(legend_embs: torch.Tensor,
                         bar_embs: torch.Tensor):
    """
    Ghép mỗi legend với bar có độ tương đồng cao nhất.

    Giả định legend_embs và bar_embs đã được L2-normalize.

    Parameters
    ----------
    legend_embs : torch.Tensor
        Tensor shape (L, D), L là số legend patch.
    bar_embs : torch.Tensor
        Tensor shape (B, D), B là số bar patch.

    Returns
    -------
    matches : dict
        {
          "legend_to_bar": list length L,
                           mỗi phần tử là index bar tương ứng (int),
          "scores": list length L,
                    mỗi phần tử là cosine similarity (float),
          "similarity_matrix": torch.Tensor shape (L, B)
                               (có thể dùng để debug/visualize)
        }
    """
    if legend_embs.ndim != 2 or bar_embs.ndim != 2:
        raise ValueError("legend_embs and bar_embs must be 2D tensors")

    if legend_embs.size(1) != bar_embs.size(1):
        raise ValueError("Embedding dim mismatch between legend and bar")

    # Đảm bảo là float32
    legend_embs = legend_embs.float()
    bar_embs = bar_embs.float()

    # Tính ma trận similarity: (L, D) @ (D, B) = (L, B)
    # Vì đã L2-normalize nên dot product = cosine similarity
    sim_matrix = torch.matmul(legend_embs, bar_embs.t())  # (L, B)

    # Lấy bar tốt nhất cho mỗi legend
    scores, indices = torch.max(sim_matrix, dim=1)  # both shape (L,)

    matches = {
        "legend_to_bar": indices.tolist(),
        "scores": scores.tolist(),
        "similarity_matrix": sim_matrix,  # giữ tensor để debug nếu cần
    }
    return matches

"""## draw_debug_image"""

def draw_debug_image(
    base_image_rgb,
    xaxis, yaxis,
    legend_patches,
    legend_text_boxes,
    bar_rects,
    x_label_rects,
    legend_for_bar,
    x_label_for_bar,
    save_path,
    # --- CÁC BIẾN BẬT TẮT ĐƯỜNG NỐI (Mặc định True) ---
    draw_line_legend_text=True, # Nối: Patch màu -> Text chú giải
    draw_line_bar_legend=True,  # Nối: Cột -> Patch màu chú giải
    draw_line_bar_xlabel=False   # Nối: Cột -> X-label dưới trục
):
    """
    Vẽ thông tin debug với tùy chọn bật tắt các đường nối (lines).
    """
    img = cv2.cvtColor(base_image_rgb.copy(), cv2.COLOR_RGB2BGR)

    COLOR_AXIS      = (0, 0, 0)
    COLOR_XLABEL    = (160, 160, 160)
    thickness = 1

    def rect_to_xyxy(rect):
        x, y, w, h = rect
        return int(x), int(y), int(x + w), int(y + h)

    def rect_center(rect):
        x, y, w, h = rect
        return int(x + w / 2.0), int(y + h / 2.0)

    # ===== Palette màu =====
    palette = [
        (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
        (128, 255, 0), (255, 128, 0),
    ]

    def color_for_legend(idx):
        if idx is None or idx < 0:
            return (100, 100, 100)
        return palette[idx % len(palette)]

    # 1) Vẽ trục
    if xaxis is not None:
        x1, y1, x2, y2 = map(int, xaxis)
        cv2.line(img, (x1, y1), (x2, y2), COLOR_AXIS, thickness)
    if yaxis is not None:
        x1, y1, x2, y2 = map(int, yaxis)
        cv2.line(img, (x1, y1), (x2, y2), COLOR_AXIS, thickness)

    # 2) Vẽ nền tất cả x-label (màu xám trước)
    for rect in x_label_rects or []:
        x1, y1, x2, y2 = rect_to_xyxy(rect)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_XLABEL, thickness)

    # ===== Legend =====
    legend_count = min(len(legend_patches or []), len(legend_text_boxes or []))

    # 3) Vẽ Legend
    for legend_idx in range(legend_count):
        c = color_for_legend(legend_idx)
        patch_rect = legend_patches[legend_idx]
        text_rect  = legend_text_boxes[legend_idx]

        # bbox
        p_x1, p_y1, p_x2, p_y2 = rect_to_xyxy(patch_rect)
        cv2.rectangle(img, (p_x1, p_y1), (p_x2, p_y2), c, thickness)

        t_x1, t_y1, t_x2, t_y2 = rect_to_xyxy(text_rect)
        cv2.rectangle(img, (t_x1, t_y1), (t_x2, t_y2), c, thickness)

        # [CHECK FLAG] Line patch -> text
        if draw_line_legend_text:
            c_patch = rect_center(patch_rect)
            c_text  = rect_center(text_rect)
            cv2.line(img, c_patch, c_text, c, thickness)

    # 4) Vẽ Bar & X-Label (re-color)
    n_bars = len(bar_rects or [])
    for i in range(n_bars):
        bar_rect = bar_rects[i]

        leg_idx = None
        if legend_for_bar is not None and i < len(legend_for_bar):
            leg_idx = legend_for_bar[i]

        c = color_for_legend(leg_idx)
        c_bar = rect_center(bar_rect)

        # Vẽ bbox bar
        x1, y1, x2, y2 = rect_to_xyxy(bar_rect)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)

        # [CHECK FLAG] Line bar -> Legend Patch
        if leg_idx is not None and 0 <= leg_idx < legend_count:
            if draw_line_bar_legend:
                patch_rect = legend_patches[leg_idx]
                c_patch = rect_center(patch_rect)
                cv2.line(img, c_bar, c_patch, c, thickness)

        # Xử lý với X-label tương ứng
        if x_label_for_bar is not None and i < len(x_label_for_bar):
            lbl_rect = x_label_for_bar[i]
            if lbl_rect is not None:
                c_lbl = rect_center(lbl_rect)

                # [CHECK FLAG] Line bar -> X-label
                if draw_line_bar_xlabel:
                    cv2.line(img, c_bar, c_lbl, c, thickness)

                # Vẫn luôn vẽ lại màu bbox của X-label (để đồng bộ màu với bar)
                # kể cả khi tắt đường nối, ta vẫn muốn nhìn thấy màu tương ứng.
                lx1, ly1, lx2, ly2 = rect_to_xyxy(lbl_rect)
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), c, thickness)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    return img

images = []
texts = []

def getYVal(IMG_DIR, JSON_DIR, objects_dectector, backbone):
    img_dir = Path(IMG_DIR)
    json_dir = Path(JSON_DIR)
    predict_dir = os.path.join(json_dir, "images/val")

    # Collect image paths
    image_paths = [p for p in img_dir.iterdir()
                   if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    yValueDict = {}
    axes_store = {}
    legend_store = {}

    # 2. Loop xử lý từng ảnh (streamed batches)
    for index, (path, r) in enumerate(
        run_inference(
            image_paths,
            predict_dir,
            objects_dectector,
            TASK4_CONFIG.get("device", "auto"),
            batch_size=TASK4_CONFIG.get("batch_size", 2),
            imgsz=TASK4_CONFIG.get("imgsz", 640),
        )
    ):
        # Chỉ xử lý file ảnh (đã lọc ở trên, check lại cho chắc)
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        # ===== 1) Đường dẫn ảnh =====
        img_path = str(path)

        # ===== 2) Đường dẫn JSON tương ứng =====
        json_path = json_dir / f"{path.stem}.json"

        if not json_path.exists():
            print(f"⚠️ Không tìm thấy JSON cho ảnh {path.name}: {json_path}")
            continue

        # Đọc output từ task 2 3
        p = json_path
        task2_3 = json.loads(p.read_text(encoding="utf-8"))

        # ===== 3) Load ảnh gốc =====
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape

        # path là đường dẫn file ảnh
        actual_image = Image.open(path).convert("RGB")


        # ============= Objects detection ====================
        boxes = r.boxes         # tất cả bbox
        names = r.names         # dict: {0: 'legend', 1: 'bar', 2: 'plot', ...}

        # Khởi tạo dict: mỗi class là 1 list
        detections_by_class = {name: [] for name in names.values()}

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            cls_name = names[cls_id]

            detections_by_class[cls_name].append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "score": conf,
            })

        # Ví dụ: lấy riêng tất cả bbox class 'legend', 'bar', 'plot'
        legend_dets = detections_by_class.get("legend", [])
        bar_dets    = detections_by_class.get("bar", [])
        plot_dets   = detections_by_class.get("plot", [])

        # ====================== DETECT AXES ============================
        plot_bb = {}
        if not plot_dets:
            xaxis = None
            yaxis = None
        else:
            # Chọn bbox plot lớn nhất (thường là vùng biểu đồ chính)
            best_plot = max(
                plot_dets,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
            )

            x1, y1, x2, y2 = best_plot["bbox"]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Giả định:
            # - trục X là cạnh dưới của bbox plot: từ (x1, y2) -> (x2, y2)
            # - trục Y là cạnh trái của bbox plot: từ (x1, y1) -> (x1, y2)
            xaxis = (x1, y2, x2, y2)
            yaxis = (x1, y1, x1, y2)
            plot_bb = {"x0": x1, "y0": y1, "x1": x2, "y1": y2}

        if xaxis is None or yaxis is None:
            print(f"❌ Không detect được plot cho {path.name}, bỏ qua.")
            continue

        # store axes info for task4 output
        axes_store[path.name] = {
            "_plot_bb": plot_bb,
            "axes": {
                "x-axis": [
                    {"id": 0, "tick_pt": {"x": xaxis[0], "y": xaxis[1]}},
                    {"id": 1, "tick_pt": {"x": xaxis[2], "y": xaxis[3]}},
                ],
                "y-axis": [
                    {"id": 0, "tick_pt": {"x": yaxis[0], "y": yaxis[1]}},
                    {"id": 1, "tick_pt": {"x": yaxis[2], "y": yaxis[3]}},
                ],
            },
        }

        # ================== GET ALL LABEL =====================================

        image, x_tick_list, x_title, y_tick_list, y_title, legend_text_boxes, image_text = \
            getProbableLabels(image, task2_3, xaxis, yaxis)

        # ================= GET RATIO ==========================================
        list_text, normalize_ratio, (min_val, min_pixel) = getRatio_optimized(y_tick_list)

        if normalize_ratio is None:
            # Không tính được ratio (thiếu tick Y numeric, v.v.)
            print(f"❌ Không tính được normalize_ratio cho {path.name}, bỏ qua chart này.")
            continue


        print("[{0}] path: {1}, ratio: {2}".format(index, path.name, normalize_ratio), end='\n\n')

        try:
            ## =============== LEGEND ANALYSIS =================================
            NO_LEGEND = False
            legendtexts = []
            legendrects = []
            legend_patch_boxes = []
            legend_text_rects = []

            # ========== Xử lý trường hợp KHÔNG CÓ legend ==========
            if not legend_text_boxes:
                print(f"⚠️ Không tìm thấy legend nào cho {path.name}, dùng 'series_0' làm tên series.")
                legendtexts = ["series_0"]
                NO_LEGEND = True

            for det in legend_dets:
                x1, y1, x2, y2 = det["bbox"]
                legend_patch_boxes.append((x1, y1, x2 - x1, y2 - y1))

            # if len(legend_patch_boxes) != len(legend_text_boxes):
            #     print(f"⚠️ lỗi phân role hoặc detect thiếu legend {path.name}.")
            #     continue

            # legend_text_boxes: list[(text, (tx, ty, tw, th))]
            # legend_patch_boxes  : list[(x, y, w, h)] từ YOLO (ô màu)
            if not NO_LEGEND:
                assignments = assign_legend_patches(
                    legend_boxes=legend_text_boxes,
                    patch_rects=legend_patch_boxes,
                    y_tol=20,         # chỉnh theo ảnh
                    prefer_left=True, # giữ patch nằm bên trái text
                    max_cost=None     # hoặc ví dụ 200, tuỳ bạn
                )

                # Ghép legend text (từ getProbableLabels) với legend patch (từ YOLO)
                for idx, box in enumerate(legend_text_boxes):
                    text, (textx, texty, width, height) = box

                    patch_box = assignments[idx]
                    if patch_box is None:
                        print("  ❌ Không tìm được ô màu nào gần legend này.")
                        continue

                    (lx, ly, lw, lh) = patch_box
                    legendrects.append(patch_box)
                    legendtexts.append(text)
                    legend_text_rects.append((textx, texty, width, height))  # <<-- THÊM
                    print(f"  - Legend '{text}' có bbox {patch_box}")

            if len(x_tick_list) > 0:
                # TRƯỜNG HỢP 1: CÓ X-TICK (Biểu đồ nhóm - Grouped Bar Chart)
                # Cấu trúc: { "Legend A": {"Năm 2020": 0.0, "Năm 2021": 0.0}, ... }
                mode = "matrix"
                data = {
                    legend_text: {x_label: 0.0 for x_label, _ in x_tick_list}
                    for legend_text in legendtexts
                }

            else:
                # TRƯỜNG HỢP 2: KHÔNG CÓ X-TICK (Biểu đồ đơn - Simple Bar Chart)
                # Lúc này Legend chính là định danh cho cột.
                # Cấu trúc phẳng: { "French controls": 0.0, "French controls from general...": 0.0, ... }
                mode = "flat"
                data = {legend_text: 0.0 for legend_text in legendtexts}

            ## =============== Detect Bboxes bar =====================
            bar_rects = []
            for det in bar_dets:
                x1, y1, x2, y2 = det["bbox"]
                bar_rects.append((x1, y1, x2 - x1, y2 - y1))

            # ========== Trường hợp KHÔNG CÓ bar ==========
            if not bar_rects:
                print(f"❌ Không phát hiện bar nào cho {path.name}, bỏ qua chart.")
                continue

            # ================== EMBEDDING LEGEND &BAR ===================
            legend_embs_list = []
            legend_sizes = []
            for (lx, ly, lw, lh) in legendrects:
                # convert (x, y, w, h) -> (x1, y1, x2, y2)
                bbox_xyxy = (lx, ly, lx + lw, ly + lh)
                try:
                    # 1) Ước lượng kích thước legend sau khi shrink (để dùng làm ref cho bar)
                    sx1, sy1, sx2, sy2 = shrink_legend_bbox(bbox_xyxy, (img_width, img_height))
                    legend_sizes.append((sx2 - sx1, sy2 - sy1))

                    # 2) Trích embedding với tiền xử lý dạng legend
                    emb = extract_patch_embedding(
                        actual_image,
                        bbox_xyxy,
                        backbone,
                        kind="legend",          # <<< quan trọng
                        legend_ref_size=None    # legend không cần ref_size
                    )
                    legend_embs_list.append(emb)

                except Exception as e:
                    print(f"⚠ Lỗi extract embedding legend patch {bbox_xyxy} trong {path.name}: {e}")

            legend_ref_size = None
            if legend_sizes:
                avg_w = int(sum(w for w, _ in legend_sizes) / len(legend_sizes))
                avg_h = int(sum(h for _, h in legend_sizes) / len(legend_sizes))
                legend_ref_size = (avg_w, avg_h)


            bar_embs_list = []
            for (bx, by, bw, bh) in bar_rects:
                bbox_xyxy = (bx, by, bx + bw, by + bh)
                try:
                    emb = extract_patch_embedding(
                      actual_image,
                      bbox_xyxy,
                      backbone,
                      kind="bar",              # <<< tiền xử lý kiểu bar
                      legend_ref_size=legend_ref_size  # <<< central crop cho giống legend
                    )
                    bar_embs_list.append(emb)
                except Exception as e:
                    print(f"⚠ Lỗi extract embedding bar patch {bbox_xyxy} trong {path.name}: {e}")

            legend_for_bar = None  # danh sách length B: mỗi phần tử là index legend (0..L-1)


            # Khởi tạo trước để tránh UnboundLocalError ở chỗ khác
            legend_embs = None
            bar_embs = None
            sim_matrix = None

            # Chỉ chạy khi có đủ dữ liệu multi-legend
            if legend_embs_list and bar_embs_list and len(legend_embs_list) == len(legendtexts):
                try:
                    legend_embs = torch.stack(legend_embs_list)  # (L, D)
                    bar_embs    = torch.stack(bar_embs_list)     # (B, D)

                    matches = match_legend_to_bars(legend_embs, bar_embs)
                    sim_matrix = matches["similarity_matrix"]    # (L, B)

                    # Với mỗi bar (cột theo trục 1), chọn legend có similarity lớn nhất
                    # sim_matrix: (L, B) -> argmax theo dim=0 -> (B,)
                    legend_for_bar = torch.argmax(sim_matrix, dim=0).tolist()  # list length B


                except Exception as e:
                    print(f"⚠ Lỗi match_legend_to_bars trong {path.name}: {e}")
                    legend_for_bar = None
                    legend_embs = None
                    bar_embs = None

            # Trường hợp fallback:
            if legend_for_bar is None:
                # Nếu chỉ có 1 legend -> mọi bar thuộc legend 0
                if len(legendtexts) == 1:
                    legend_for_bar = [0] * len(bar_rects)
                else:
                    # Multi-legend nhưng match lỗi -> log và tạm coi mọi bar thuộc legend 0
                    print(
                        f"⚠ {path.name}: match_legend_to_bars không dùng được, "
                        "tạm thời gán tất cả bar cho legend đầu tiên."
                    )
                    legend_for_bar = [0] * len(bar_rects)


            # ======= Gán mỗi bar → tick X gần nhất (labels) ======
            textBoxes = []
            labels = []  # label text cho từng bar_rect

            for rectBox in bar_rects:
                min_distance = sys.maxsize
                closestBox = None
                labeltext = None

                for text, textBox in x_tick_list:
                    d = RectDist(rectBox, textBox)
                    if d < min_distance:
                        min_distance = d
                        closestBox = textBox
                        labeltext = text

                textBoxes.append(closestBox)
                labels.append(labeltext)

            # # Chiều cao từng bar
            list_len = [(rect, float(rect[3])) for rect in bar_rects]

            nd = infer_ndigits_from_ticks(y_tick_list, default=1, cap=3)
            # # y-values = chiều cao * normalize_ratio
            y_val = [(rect, round(min_val + l * normalize_ratio, nd+1)) for rect, l in list_len]


            # ========== Gán y-value cho TỪNG legend ==========
            # 1. Xác định Mode dựa trên x_tick_list
            is_flat_mode = len(x_tick_list) == 0

            # 2. Bắt đầu gán dữ liệu
            for legend_idx, legendtext in enumerate(legendtexts):
                print(f" >> Gán giá trị cho legend '{legendtext}'")

                # ================== TRƯỜNG HỢP FLAT MODE (Không có trục X) ==================
                if is_flat_mode:
                    # Logic: Tìm thanh bar nào thuộc về legend này và gán giá trị trực tiếp
                    found_val = 0.0

                    for idx_bar, item in enumerate(y_val):
                        # Kiểm tra xem bar này có thuộc legend đang xét không?
                        if legend_for_bar[idx_bar] == legend_idx:
                            found_val = item[1] # Lấy giá trị (chiều cao * ratio)
                            break # Tìm thấy rồi thì dừng (vì Flat mode là 1-1)

                    # Gán vào dictionary phẳng
                    data[legendtext] = found_val

                # ================== TRƯỜNG HỢP MATRIX MODE (Có trục X) ==================
                else:
                    # Logic cũ của bạn (đúng cho trường hợp này)
                    for x_label, box in x_tick_list:
                        (x, y, w, h) = box
                        value = 0.0
                        dist = sys.maxsize

                        for idx_bar, item in enumerate(y_val):
                            # 1. Bar phải thuộc legend này (check màu)
                            if legend_for_bar[idx_bar] != legend_idx:
                                continue

                            # 2. Bar phải nằm đúng vị trí cột X (check vị trí)
                            # Lưu ý: cần đảm bảo labels[idx_bar] đã được gán đúng ở đoạn code trên
                            if labels[idx_bar] == x_label:
                                (vx, vy, vw, vh) = item[0]
                                cx_bar = vx + vw / 2.0
                                cx_lbl = x + w / 2.0
                                d = abs(cx_lbl - cx_bar)

                                # Lấy bar gần tâm nhãn X nhất (để tránh nhiễu)
                                if d < dist:
                                    dist = d
                                    value = item[1]

                        # Gán vào dictionary lồng nhau
                        data[legendtext][x_label] = value

            # Legend colors for task5
            legend_pairs = []
            if legendtexts:
                for idx, legend_text in enumerate(legendtexts):
                    color = [0, 0, 0]
                    if idx < len(legendrects):
                        color = mean_color_rgb(image, legendrects[idx])
                    legend_pairs.append({"legend": legend_text, "color": color})
            legend_store[path.name] = legend_pairs

            # ================== VẼ ẢNH DEBUG ==========================
            # List bbox x-label (tất cả tick)
            x_label_rects = [box for (_, box) in x_tick_list]

            # Với mỗi bar, đã có textBoxes là bbox x-label gần nhất
            debug_dir = os.path.join(TASK4_CONFIG["output_json"], "ResultImage")
            debug_path = os.path.join(debug_dir, f"{path.stem}_debug.png")

            # # Lưu ý: actual_image là RGB
            draw_debug_image(
                base_image_rgb=image,
                xaxis=xaxis,
                yaxis=yaxis,
                legend_patches=legendrects,        # ô chú giải
                legend_text_boxes=legend_text_rects,  # bbox text legend
                bar_rects=bar_rects,
                x_label_rects=x_label_rects,
                legend_for_bar=legend_for_bar,
                x_label_for_bar=textBoxes,
                save_path=debug_path
            )

            # =================================================
            yValueDict[path.name] = data


        except Exception as e:
            print(f"❌ Lỗi khi xử lý {path.name}: {e}")
            continue

    return yValueDict, axes_store, legend_store

def main():
    # Load the model
    objects_dectector = YOLO(TASK4_CONFIG["yolo_weight"])

    # Khởi tạo backbone 1 lần
    backbone = get_backbone("resnet50")  # hoặc "clip_vitb32", "swin_tiny"

    yValueDict, axes_store, legend_store = getYVal(TASK4_CONFIG["input_images"], TASK4_CONFIG["input_json"], objects_dectector, backbone)

    rows = []
    for img_name, legends_dict in yValueDict.items():
        for legend, legend_data in legends_dict.items():
            if isinstance(legend_data, dict):
                # Matrix mode: legend_data is a dictionary mapping x_label to value
                for x_label, val in legend_data.items():
                    rows.append({
                        "image": img_name,
                        "legend": legend,
                        "x_label": x_label,
                        "value": float(val)  # ép float cho ổn định
                    })
            else:
                # Flat mode: legend_data is the value itself
                rows.append({
                    "image": img_name,
                    "legend": legend,
                    "x_label": "Value", # Placeholder for flat mode
                    "value": float(legend_data)  # ép float cho ổn định
                })

    df = pd.DataFrame(rows)

    save_results(
        df=df,
        yValueDict=yValueDict,
        axes_store=axes_store,
        legend_store=legend_store,
        excel_path=TASK4_CONFIG["output_excel"],
        json_path=TASK4_CONFIG["output_json"]
    )
