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

# Import Config chung của dự án
import Config

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
# Giả định file Config.py của bạn có hàm returnTestTask4_Config
try:
    TASK4_CONFIG = Config.returnTestTask4_Config()
except AttributeError:
    print("[WARN] Chưa tìm thấy Config.returnTestTask4_Config, dùng cấu hình mặc định.")
    # Fallback paths (bạn sửa lại nếu cần test nhanh không qua Config)
    TASK4_CONFIG = {
        "input_images": "./dataset/images",
        "input_json": "./dataset/json",
        "output_excel": "./output/y_values.xlsx",
        "yolo_weight": "./weights/best.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 2. HÀM HỖ TRỢ HÌNH HỌC (GEOMETRY UTILS)
# ==========================================
def lineIntersectsRectX(candx, rect):
    (x, y, w, h) = rect
    return x <= candx <= x + w

def point_line_distance(px, py, x1, y1, x2, y2):
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)

def RectDist(rectA, rectB):
    (rectAx, rectAy, rectAw, rectAh) = rectA
    (rectBx, rectBy, rectBw, rectBh) = rectB
    return abs((rectAx + rectAw / 2) - (rectBx + rectBw / 2))

def nearbyRectangle(current, candidate, threshold):
    (currx, curry, currw, currh) = current
    (candx, candy, candw, candh) = candidate
    
    currxmax, currymax = currx + currw, curry + currh
    candxmax, candymax = candx + candw, candy + candh

    if candymax <= curry and candymax + threshold >= curry: return True
    if candy >= currymax and currymax + threshold >= candy: return True
    if candymax >= curry and candy <= curry: return True
    if currymax >= candy and curry <= candy: return True
    if (candy >= curry and candy <= currymax and candymax >= curry and candymax <= currymax): return True
    return False

def mergeRects(rects, mode='contours'):
    # Đơn giản hóa logic merge để code gọn hơn
    if not rects: return []
    # Nếu rects là list contour, convert sang rect
    if mode == 'contours':
        rects = [cv2.boundingRect(c) for c in rects]
        
    rects.sort(key=lambda x: x[0])
    accepted = []
    used = [False] * len(rects)
    xThr, yThr = 5, 5

    for i, r in enumerate(rects):
        if used[i]: continue
        curr = list(r)
        used[i] = True
        
        for j in range(i+1, len(rects)):
            if used[j]: continue
            cand = rects[j]
            if cand[0] <= (curr[0] + curr[2]) + xThr:
                if nearbyRectangle(curr, cand, yThr):
                    # Merge logic
                    x_new = curr[0]
                    y_new = min(curr[1], cand[1])
                    w_new = max(curr[0]+curr[2], cand[0]+cand[2]) - x_new
                    h_new = max(curr[1]+curr[3], cand[1]+cand[3]) - y_new
                    curr = [x_new, y_new, w_new, h_new]
                    used[j] = True
            else:
                break
        accepted.append(tuple(curr))
    return accepted

def filterBbox(rects, legendBox):
    text, (textx, texty, width, height) = legendBox
    filtered = []
    for rect in rects:
        (x, y, w, h) = rect
        # Lấy các box nằm ngang hàng (+- 10 pixel)
        if abs(y - texty) <= 10 and abs((y + h) - (texty + height)) <= 10:
            filtered.append(rect)
    
    filtered = mergeRects(filtered, 'rects')
    
    closest = None
    dist = sys.maxsize
    for rect in filtered:
        (x, y, w, h) = rect
        # Tìm box gần nhất (thường là bên trái text)
        d = abs((x + w) - textx)
        if d <= dist:
            dist = d
            closest = rect
    return closest

# ==========================================
# 3. XỬ LÝ ẢNH & TEXT (IMAGE PROCESSING)
# ==========================================
def cleanText(image_text):
    return [(text, (textx, texty, w, h)) for text, (textx, texty, w, h) in image_text if text.strip() != 'I']

def _poly_from_block(block):
    p = block["polygon"]
    # Check nếu polygon là dict hay list để xử lý
    if isinstance(p, dict):
        verts = [[p["x0"], p["y0"]], [p["x1"], p["y1"]], [p["x2"], p["y2"]], [p["x3"], p["y3"]]]
    else:
        verts = np.array(p).reshape(-1, 2).tolist()
    return np.array(verts, dtype=np.int32).reshape((-1, 1, 2))

def _dilate_polygon_on_mask(img_shape, poly, pad):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if pad > 0:
        k = 1 + pad * 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def fill_text_regions(img, chart_json, roles_to_remove=None, pad=1, fill_color=(255, 255, 255)):
    texts = chart_json.get("texts", [])
    if not texts: return img
    
    # Tạo mask để vẽ đè lên
    acc_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Nếu có roles, tạo map
    role_map = {}
    if chart_json.get("roles"):
        role_map = {r["id"]: r["role"] for r in chart_json["roles"]}

    for block in texts:
        bid = block["id"]
        role = role_map.get(bid, None)
        
        if (roles_to_remove is None) or (role in roles_to_remove):
            poly = _poly_from_block(block)
            mask = _dilate_polygon_on_mask(img.shape, poly, pad=pad)
            acc_mask = np.maximum(acc_mask, mask)

    img_out = img.copy()
    img_out[acc_mask > 0] = fill_color
    return img_out

def load_chart_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Hỗ trợ lấy text blocks từ nhiều nguồn (task 3 hoặc task 2)
        try:
            texts = d["task3"]["input"]["task2_output"]["text_blocks"]
            roles = d["task3"]["output"]["text_roles"]
        except KeyError:
            texts = d.get("task2", {}).get("output", {}).get("text_blocks", [])
            roles = []
            
        return {"doc": d, "texts": texts, "roles": roles}
    except Exception as e:
        print(f"[ERR] Load JSON {path} failed: {e}")
        return {"doc": {}, "texts": [], "roles": []}

# ==========================================
# 4. LOGIC BIỂU ĐỒ (CHART LOGIC)
# ==========================================
def getProbableLabels(image, d, xaxis, yaxis):
    try:
        text_blocks = d["task3"]["input"]["task2_output"]["text_blocks"]
    except KeyError:
        try: text_blocks = d["task2"]["output"]["text_blocks"]
        except: return image, [], [], [], [], [], [], [], [], []

    id_to_text, id_to_rect = {}, {}
    raw_image_text = []

    for block in text_blocks:
        bid = block["id"]
        txt = block["text"]
        poly = block["polygon"]
        
        if isinstance(poly, dict):
            xs = [poly["x0"], poly["x1"], poly["x2"], poly["x3"]]
            ys = [poly["y0"], poly["y1"], poly["y2"], poly["y3"]]
        else:
            xs, ys = poly[0::2], poly[1::2]

        x_min, y_min = min(xs), min(ys)
        w, h = max(xs) - x_min, max(ys) - y_min
        rect = (x_min, y_min, w, h)
        
        id_to_text[bid] = txt
        id_to_rect[bid] = rect
        raw_image_text.append((txt, rect))

    image_text = cleanText(raw_image_text)
    
    # Lấy roles
    try:
        text_roles = d["task3"]["output"]["text_roles"]
        id_to_role = {item["id"]: item["role"] for item in text_roles}
    except: id_to_role = {}

    tick_blocks, axis_blocks, legend_blocks = [], [], []
    for bid, role in id_to_role.items():
        if bid not in id_to_text: continue
        item = (id_to_text[bid], id_to_rect[bid])
        if role == "tick_label": tick_blocks.append(item)
        elif role == "axis_title": axis_blocks.append(item)
        elif role == "legend_label": legend_blocks.append(item)

    (x1, y1, x2, y2) = xaxis
    (yx1, yy1, yx2, yy2) = yaxis
    x_labels_list, y_labels_list = [], []
    x_labels, y_labels = [], []

    for text, (tx, ty, w, h) in tick_blocks:
        cx, cy = tx + w/2, ty + h/2
        side_xaxis = np.sign((x2 - x1)*(cy - y1) - (y2 - y1)*(cx - x1))
        side_yaxis = np.sign((yx2 - yx1)*(cy - yy1) - (yy2 - yy1)*(cx - yx1))

        if side_xaxis == -1 and side_yaxis == 1:
            y_labels_list.append((text, (tx, ty, w, h)))
            y_labels.append(text)
        elif side_xaxis == 1 and side_yaxis == -1:
            x_labels_list.append((text, (tx, ty, w, h)))
            x_labels.append(text)

    x_text, y_text_list = [], []
    for text, (tx, ty, w, h) in axis_blocks:
        cx, cy = tx + w/2, ty + h/2
        dist_x = point_line_distance(cx, cy, x1, y1, x2, y2)
        dist_y = point_line_distance(cx, cy, yx1, yy1, yx2, yy2)
        if dist_y < dist_x: y_text_list.append((text, (tx, ty, w, h)))
        else: x_text.append(text)

    legends = [t for t, _ in legend_blocks]
    return image, x_labels, x_labels_list, x_text, y_labels, y_labels_list, y_text_list, legends, legend_blocks, image_text

def reject_outliers(data, m=1):
    if len(data) == 0: return data
    return data[abs(data - np.mean(data)) <= m * np.std(data)]

def getRatio(path, image_text, xaxis, yaxis):
    list_text = []
    
    # Quét để tìm các số nằm trên trục Y
    for text, (textx, texty, w, h) in image_text:
        text = text.strip()
        (x1, y1, x2, y2) = xaxis
        (x11, y11, x22, y22) = yaxis
        
        # Check vị trí hình học (Góc phần tư trên-trái so với giao điểm trục)
        if (np.sign((x2 - x1)*(texty - y1) - (y2 - y1)*(textx - x1)) == -1 and 
            np.sign((x22 - x11)*(texty - y11) - (y22 - y11)*(textx - x11)) == 1):
            
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                # Logic cũ của bạn: lineIntersectsRectX -> nhưng ở đây simplify
                # Lấy số đầu tiên tìm được và tọa độ đáy chữ (y + h)
                list_text.append((float(numbers[0]), texty + h))

    if len(list_text) < 2: return [], 0
    
    list_text.sort(key=lambda x: x[0])
    vals = [x[0] for x in list_text]
    ys   = [x[1] for x in list_text]
    
    val_diff = [vals[i] - vals[i-1] for i in range(1, len(vals))]
    y_diff   = [abs(ys[i] - ys[i-1]) for i in range(1, len(ys))] # Absolute pixel diff

    val_diff = reject_outliers(np.array(val_diff))
    y_diff = reject_outliers(np.array(y_diff))
    
    if len(y_diff) == 0 or np.mean(y_diff) == 0: return vals, 0
    
    normalize_ratio = np.mean(val_diff) / np.mean(y_diff)
    return vals, normalize_ratio

# ==========================================
# 5. DEEP LEARNING (BACKBONE & MATCHING)
# ==========================================
def get_backbone(model_type="resnet50", device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "resnet50": model_name = "resnet50"
    else: model_name = "resnet50" 

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    return {"model": model, "transform": transform, "device": device}

def extract_patch_embedding(image, bbox, backbone):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = map(int, bbox)
    w, h = img_pil.size
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return None

    patch = img_pil.crop((x1, y1, x2, y2))
    tensor = backbone["transform"](patch).unsqueeze(0).to(backbone["device"])
    
    with torch.no_grad():
        emb = backbone["model"](tensor).flatten()
        emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu()

def match_legend_to_bars(legend_embs, bar_embs):
    sim_matrix = torch.matmul(legend_embs, bar_embs.t())
    matches = {"similarity_matrix": sim_matrix}
    return matches

# ==========================================
# 6. MAIN WORKFLOW
# ==========================================
def main():
    print("--- STARTING TASK 4 ---")
    
    # 1. Load Models
    print(" -> Loading YOLO...")
    objects_detector = YOLO(TASK4_CONFIG['yolo_weight'])
    
    print(" -> Loading Backbone...")
    backbone = get_backbone("resnet50", TASK4_CONFIG['device'])
    
    img_dir = Path(TASK4_CONFIG['input_images'])
    json_dir = Path(TASK4_CONFIG['input_json'])
    
    # 2. Run Inference (Nếu muốn chạy batch trước như code cũ)
    # Tuy nhiên, để tối ưu mình sẽ detect từng ảnh trong loop bên dưới
    # Nhưng để tôn trọng flow code của bạn (dùng .predict save result), mình sẽ giữ logic detect
    
    # Lấy danh sách ảnh
    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    print(f" -> Found {len(image_paths)} images.")

    rows = []
    
    for path in tqdm(image_paths, desc="Processing"):
        img_path = str(path)
        json_path = json_dir / f"{path.stem}.json"
        
        if not json_path.exists():
            continue
            
        # --- A. Đọc dữ liệu ---
        image = cv2.imread(img_path)
        actual_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Để extract embedding
        
        # Load JSON Task 2/3
        with open(json_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            
        # --- B. Detect Objects (YOLO) ---
        # Chạy detect trực tiếp trên ảnh này thay vì chạy batch từ trước
        results = objects_detector(image, verbose=False)
        r = results[0]
        
        detections = {"legend": [], "bar": [], "plot": []}
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]
            if name in detections:
                detections[name].append(list(map(int, box.xyxy[0].tolist())))

        # --- C. Detect Axes (Plot Box) ---
        plot_dets = detections["plot"]
        if not plot_dets:
            # print(f"Skipping {path.name}: No plot detected.")
            continue
            
        best_plot = max(plot_dets, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
        x1, y1, x2, y2 = best_plot
        xaxis = (x1, y2, x2, y2)
        yaxis = (x1, y1, x1, y2)

        # --- D. OCR & Labels Analysis ---
        _, x_labels, x_labels_list, _, _, _, _, legends, legend_blocks, image_text = \
            getProbableLabels(image, input_json, xaxis, yaxis)

        # --- E. Calculate Ratio ---
        _, normalize_ratio = getRatio(img_path, image_text, xaxis, yaxis)
        if not normalize_ratio:
            continue

        # --- F. Match Legends (Visual) ---
        legend_rects_patches = []
        final_legend_texts = []
        
        legend_dets_yolo = detections["legend"]
        for l_text, l_box in legend_blocks:
            matched = filterBbox(legend_dets_yolo, (l_text, l_box))
            if matched:
                legend_rects_patches.append(matched)
                final_legend_texts.append(l_text)
        
        if not final_legend_texts:
            final_legend_texts = ["series_0"]
        
        # --- G. Match Bar to Legend ---
        bar_rects = detections["bar"]
        if not bar_rects: continue

        legend_for_bar = [0] * len(bar_rects) # Mặc định

        if len(final_legend_texts) > 1 and legend_rects_patches:
            l_embs, b_embs = [], []
            for r in legend_rects_patches:
                emb = extract_patch_embedding(actual_image, (r[0], r[1], r[0]+r[2], r[1]+r[3]), backbone)
                if emb is not None: l_embs.append(emb)
            
            valid_bar_idx = []
            for i, r in enumerate(bar_rects):
                emb = extract_patch_embedding(actual_image, r, backbone) # xyxy
                if emb is not None: 
                    b_embs.append(emb)
                    valid_bar_idx.append(i)
            
            if len(l_embs) == len(final_legend_texts) and b_embs:
                matches = match_legend_to_bars(torch.stack(l_embs), torch.stack(b_embs))
                sim_matrix = matches["similarity_matrix"]
                assigned = torch.argmax(sim_matrix, dim=0).tolist()
                
                for i, a_idx in enumerate(assigned):
                    legend_for_bar[valid_bar_idx[i]] = a_idx

        # --- H. Calculate Values ---
        # Nếu không có X ticks, tạo tick giả
        if not x_labels_list:
            sorted_bars = sorted(bar_rects, key=lambda r: (r[0]+r[2])/2)
            x_labels_list = [(str(i+1), (r[0], r[3], r[2]-r[0], 10)) for i, r in enumerate(sorted_bars)]

        # Map từng bar vào X-label gần nhất
        for i, bar_box in enumerate(bar_rects):
            bx1, by1, bx2, by2 = bar_box
            bar_height = by2 - by1
            value = round(bar_height * normalize_ratio, 2)
            
            # Tìm label gần nhất
            cx_bar = (bx1 + bx2) / 2
            closest_label = "Unknown"
            min_dist = sys.maxsize
            
            for xl_text, (lx, ly, lw, lh) in x_labels_list:
                cx_lbl = lx + lw/2
                d = abs(cx_bar - cx_lbl)
                if d < min_dist:
                    min_dist = d
                    closest_label = xl_text
            
            # Xác định legend
            l_idx = legend_for_bar[i]
            if l_idx < len(final_legend_texts):
                legend_name = final_legend_texts[l_idx]
                
                rows.append({
                    "image": path.name,
                    "legend": legend_name,
                    "x_label": closest_label,
                    "value": value
                })

    # --- 3. Save Excel ---
    if rows:
        df = pd.DataFrame(rows)
        out_path = TASK4_CONFIG['output_excel']
        # Tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_excel(out_path, index=False)
        print(f"Done! Saved to {out_path}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()