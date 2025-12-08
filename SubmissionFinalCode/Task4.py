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

def getProbableLabels(image, d, xaxis, yaxis):
    """
    image: ảnh RGB (H, W, 3)
    d    : object json đã load bằng json.loads(...)
    xaxis, yaxis: tuple (x1, y1, x2, y2)

    Yêu cầu:
    - d["task3"]["input"]["task2_output"]["text_blocks"]
    - d["task3"]["output"]["text_roles"] với role: tick_label, axis_title, legend_label
    """

    # 1. Lấy text_blocks từ task3.input.task2_output =====
    try:
        text_blocks = d["task3"]["input"]["task2_output"]["text_blocks"]
    except KeyError:
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

    # 2. Lấy text_roles từ task3.output =====
    text_roles = d["task3"]["output"]["text_roles"]
    id_to_role = {item["id"]: item["role"] for item in text_roles}

    # 3. Gom theo role: tick_label, axis_title, legend_label =====
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

    # 4. Chia tick_label thành Y tick và X tick bằng cross product (logic cũ) =====
    (x1,  y1,  x2,  y2)  = xaxis
    (yx1, yy1, yx2, yy2) = yaxis

    x_labels_list = []  # [(text, rect), ...]
    y_labels_list = []  # [(text, rect), ...]
    x_labels = []       # [text, ...]
    y_labels = []       # [text, ...]

    for text, (tx, ty, w, h) in tick_blocks:
        # dùng tâm bbox để ổn định hơn
        cx = tx + w / 2.0
        cy = ty + h / 2.0

        side_xaxis = np.sign((x2  - x1)  * (cy - y1)  - (y2  - y1)  * (cx - x1))
        side_yaxis = np.sign((yx2 - yx1) * (cy - yy1) - (yy2 - yy1) * (cx - yx1))

        # Giữ đúng logic phân vùng như code cũ:

        # To the left of y-axis and top of x-axis -> Y tick
        if side_xaxis == -1 and side_yaxis == 1:
            y_labels_list.append((text, (tx, ty, w, h)))
            y_labels.append(text)

        # To the right of y-axis and bottom of x-axis -> X tick
        elif side_xaxis == 1 and side_yaxis == -1:
            x_labels_list.append((text, (tx, ty, w, h)))
            x_labels.append(text)

        # các trường hợp còn lại (side_xaxis/side_yaxis = 0 hoặc vùng khác) bỏ qua

    # 5. Chia axis_title thành x_text (tiêu đề trục X) và y_text_list (tiêu đề/mô tả trục Y) =====
    x_text = []      # list string
    y_text_list = [] # list (text, rect)

    for text, (tx, ty, w, h) in axis_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0

        dist_to_x = point_line_distance(cx, cy, x1, y1, x2, y2)
        dist_to_y = point_line_distance(cx, cy, yx1, yy1, yx2, yy2)

        # Nếu gần trục Y hơn -> coi là y_title (bạn vẫn trả về dưới dạng (text, rect))
        if dist_to_y < dist_to_x:
            y_text_list.append((text, (tx, ty, w, h)))
        else:
            # Gần trục X hơn -> x_title
            x_text.append(text)

    # 6. Legend: dùng trực tiếp legend_label =====
    legends = [text for (text, rect) in legend_blocks]
    maxList = legend_blocks[:]  # giữ dạng [(text, rect), ...] cho tương thích output cũ

    # 7. Trả về đúng format cũ =====
    return (
        image,
        x_labels,
        x_labels_list,
        x_text,
        y_labels,
        y_labels_list,
        y_text_list,
        legends,
        maxList,
        image_text,
    )

# ==========================================
# 3. HÀM TÍNH RATIO
# ==========================================
def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]

def getRatio(path, image_text, xaxis, yaxis):
    list_text = []
    list_ticks = []

    # filepath = img_dir + "/" + path.name
    filepath = path
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape

    for text, (textx, texty, w, h) in image_text:
        text = text.strip()

        (x1, y1, x2, y2) = xaxis
        (x11, y11, x22, y22) = yaxis

        # To the left of y-axis and top of x-axis
        if (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
            np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == 1):

            # Consider numeric only for ticks on y-axis
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if bool(numbers):
                list_text.append((numbers[0], (textx, texty, w, h)))

    # Get the y-labels by finding the maximum
    # intersections with the sweeping line
    maxIntersection = 0
    maxList = []
    for i in range(x11):
        count = 0
        current = []
        for index, (text, rect) in enumerate(list_text):
            if lineIntersectsRectX(i, rect):
                count += 1
                current.append(list_text[index])

        if count > maxIntersection:
            maxIntersection = count
            maxList = current

    # Get list of text and ticks
    list_text = []
    for text, (textx, texty, w, h) in maxList:
        list_text.append(float(text))
        list_ticks.append(float(texty + h))

    text_sorted = (sorted(list_text))
    ticks_sorted  = (sorted(list_ticks))

    ticks_diff = ([ticks_sorted[i] - ticks_sorted[i-1] for i in range(1, len(ticks_sorted))])
    text_diff = ([text_sorted[i] - text_sorted[i-1] for i in range(1, len(text_sorted))])
    print("[get text-to-tick ratio] ticks_diff: {0}, text_diff: {1}".format(ticks_diff, text_diff))

    # Detected text may not be perfect! Remove the outliers.
    ticks_diff = reject_outliers(np.array(ticks_diff), m=1)
    text_diff = reject_outliers(np.array(text_diff), m=1)
    print("[reject_outliers] ticks_diff: {0}, text_diff: {1}".format(ticks_diff, text_diff))

    normalize_ratio = np.array(text_diff).mean() / np.array(ticks_diff).mean()

    return text_sorted, normalize_ratio

# ==========================================
# 4. IGNORE TEXT
# ==========================================
def load_chart_json(path):
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    texts  = d["task3"]["input"]["task2_output"]["text_blocks"]
    roles = d["task3"]["output"]["text_roles"]
    # legends = d["task5"]["output"]["legend_pairs"]
    return {"doc": d, "texts": texts , "roles": roles}

def _poly_from_block(block):
    """Chuyển polygon {x0..x3, y0..y3} → np.ndarray shape (N,1,2) cho OpenCV."""
    p = block["polygon"]
    # Giữ thứ tự 4 đỉnh như trong JSON (đã là pixel tuyệt đối)
    verts = np.array(
        [[p["x0"], p["y0"]],
         [p["x1"], p["y1"]],
         [p["x2"], p["y2"]],
         [p["x3"], p["y3"]]], dtype=np.int32
    ).reshape((-1, 1, 2))
    return verts

def _dilate_polygon_on_mask(img_shape, poly, pad):
    """
    Nở polygon bằng cách vẽ lên mask rồi dilate.
    Trả về mask nhị phân (uint8) cùng kích thước ảnh.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if pad > 0:
        k = 1 + pad * 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def fill_text_regions(
    img, chart_json,
    roles_to_remove=None,  # ví dụ: {"tick_label","axis_title","legend_label"}
    pad=1,                 # số pixel nở biên vùng text
    fill_color=(255, 255, 255)
):
    """
    Tô trắng các vùng text theo polygon trong input JSON của bạn.
    - roles_to_remove=None → xoá TẤT CẢ text_blocks
    - roles_to_remove=set(...) → chỉ xoá những block có role thuộc tập đó
    """
    texts = chart_json["texts"]
    role_map = {r["id"]: r["role"] for r in chart_json["roles"]}

    # Gom tất cả mask text cần xoá để vẽ 1 lần (tối ưu hiệu năng)
    acc_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for block in texts:
        bid = block["id"]
        role = role_map.get(bid, None)

        if (roles_to_remove is None) or (role in roles_to_remove):
            poly = _poly_from_block(block)
            mask = _dilate_polygon_on_mask(img.shape, poly, pad=pad)
            acc_mask = np.maximum(acc_mask, mask)

    # Áp dụng mask: vùng trắng hoá
    img_out = img.copy()
    img_out[acc_mask > 0] = fill_color
    return img_out

# ==========================================
# 5. HELPER FUNCTION
# ==========================================
def nearbyRectangle(current, candidate, threshold):
    (currx, curry, currw, currh) = current
    (candx, candy, candw, candh) = candidate

    currxmin = currx
    currymin = curry
    currxmax = currx + currw
    currymax = curry + currh

    candxmin = candx
    candymin = candy
    candxmax = candx + candw
    candymax = candy + candh

    # If candidate is on top, and is close
    if candymax <= currymin and candymax + threshold >= currymin:
        return True

    # If candidate is on bottom and is close
    if candymin >= currymax and currymax + threshold >= candymin:
        return True

    # If intersecting at the top, merge it
    if candymax >= currymin and candymin <= currymin:
        return True

    # If intersecting at the bottom, merge it
    if currymax >= candymin and currymin <= candymin:
        return True

    # If intersecting on the sides or is inside, merge it
    if (candymin >= currymin and
        candymin <= currymax and
        candymax >= currymin and
        candymax <= currymax):
        return True

    return False

def mergeRects(contours, mode='contours'):
    rects = []
    rectsUsed = []

    # Just initialize bounding rects and set all bools to false
    for cnt in contours:
        if mode == 'contours':
            rects.append(cv2.boundingRect(cnt))
        elif mode == 'rects':
            rects.append(cnt)

        rectsUsed.append(False)

    # Sort bounding rects by x coordinate
    def getXFromRect(item):
        return item[0]

    rects.sort(key = getXFromRect)

    # Array of accepted rects
    acceptedRects = []

    # Merge threshold for x coordinate distance
    xThr = 5
    yThr = 5

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):

            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (candxMin <= currxMax + xThr):

                    if not nearbyRectangle((candxMin, candyMin, candxMax - candxMin, candyMax - candyMin), 
                                           (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin), yThr):
                        break

                    # Reset coordinates of current rect
                    currxMax = candxMax
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)

                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                else:
                    break

            # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

    return acceptedRects

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

def filterBbox(rects, legendBox):
    text, (textx, texty, width, height) = legendBox

    filtered = []
    for rect in rects:
        (x, y, w, h) = rect
        if abs(y - texty) <= 10 and abs(y - texty + h - height) <= 10:
            filtered.append(rect)

    filtered = mergeRects(filtered, 'rects')

    closest = None
    dist = sys.maxsize
    for rect in filtered:
        (x, y, w, h) = rect
        if abs(x + w - textx) <= dist:
            dist = abs(x + w - textx)
            closest = rect

    return closest

# ==========================================
# 6. MAIN WORKFLOW
# ==========================================

# 1. Hàm lấy backbone + transform
def get_backbone(model_type: str, device: str | None = None):
    """
    Khởi tạo backbone vision + transform tiền xử lý.

    Parameters
    ----------
    model_type : str
        Một trong các giá trị:
        - "resnet50"
        - "clip_vitb32"
        - "swin_tiny"
    device : str or None
        "cuda" / "cpu". Nếu None, sẽ tự chọn "cuda" nếu có GPU.

    Returns
    -------
    backbone : dict
        {
            "model": torch.nn.Module (đã pretrained, trả về feature vector),
            "transform": callable (PIL.Image -> tensor chuẩn hóa),
            "device": str
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map tên "model_type" sang "model_name" của timm
    if model_type == "resnet50":
        model_name = "resnet50"
    elif model_type == "clip_vitb32":
        # CLIP ViT-B/32 trong timm
        model_name = "vit_base_patch32_clip_224.openai"
    elif model_type == "swin_tiny":
        model_name = "swin_tiny_patch4_window7_224"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # num_classes=0 -> timm trả về feature vector thay vì logits
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0  # rất quan trọng để dùng như feature extractor
    )
    model.eval()
    model.to(device)

    # Tạo transform chuẩn dựa trên config của model
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    backbone = {
        "model": model,
        "transform": transform,
        "device": device,
    }
    return backbone

# 2. Hàm trích embedding cho một patch (legend/bar)
def extract_patch_embedding(image, bbox, backbone):
    """
    Trích embedding cho một patch được xác định bởi bbox.

    Parameters
    ----------
    image : np.ndarray hoặc PIL.Image
        Ảnh RGB gốc (H, W, 3). Nếu là numpy sẽ được convert sang PIL.Image.
    bbox : tuple (x1, y1, x2, y2)
        Toạ độ bounding box trong hệ toạ độ ảnh gốc (pixel).
    backbone : dict
        Kết quả trả về từ get_backbone().

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

    # Lấy bbox và clamp trong phạm vi ảnh
    x1, y1, x2, y2 = bbox
    w, h = image_pil.size  # PIL: (width, height)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox} for image size {(w, h)}")

    # Crop patch
    patch = image_pil.crop((x1, y1, x2, y2))

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

def run_inference(img_dir, output_dir, detector):
    print(f"Running inference on images in: {img_dir}")
    results = detector.predict(
        source=img_dir,
        imgsz=640,
        conf=0.4,
        save=True,
        project=output_dir,
        name="predict",
        device=0
    )
    print("Inference done!")
    return results

images = []
texts = []

def getYVal(IMG_DIR, JSON_DIR, objects_dectector, backbone):
    img_dir = Path(IMG_DIR)
    json_dir = Path(JSON_DIR)
    predict_dir = TASK4_CONFIG["output_json"]

    # Run inference
    image_paths = [p for p in img_dir.iterdir()
                   if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    # results = run_inference(image_paths, predict_dir, objects_dectector)

    yValueDict = {}

    # 2. Loop xử lý từng ảnh
    for index, path in enumerate(image_paths):
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
            
        # ===== THÊM ĐOẠN NÀY: Chạy predict cho TỪNG ẢNH =====
        current_results = objects_dectector.predict(
            source=str(path),
            imgsz=640,
            conf=0.4,
            save=True,                  # Nếu muốn save ảnh predict
            project=str(predict_dir),   # Sửa lại đường dẫn cho đúng format string
            name="predict",
            device=0,                   # Dùng GPU
            verbose=False               # Tắt log cho đỡ rối
        )
        r = current_results[0]

        # ===== 1) Đường dẫn ảnh =====
        img_path = str(path)

        # ===== 2) Đường dẫn JSON tương ứng =====
        json_path = json_dir / f"{path.stem}.json"

        if not json_path.exists():
            print(f"⚠️ Không tìm thấy JSON cho ảnh {path.name}: {json_path}")
            continue

        # Đọc output từ task 2 3
        p = json_path
        input = json.loads(p.read_text(encoding="utf-8"))

        # ===== 3) Load ảnh gốc =====
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape
        actual_image = image.copy()

        # ============= Objects detection ====================
        # r = results[index]      # Results của ảnh index
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

        if xaxis is None or yaxis is None:
            print(f"❌ Không detect được plot cho {path.name}, bỏ qua.")
            continue

        # ================== GET ALL LABEL =======================
        image, x_labels, x_labels_list, x_text, y_labels, y_labels_list, y_text_list, legends, legend_boxes, image_text = \
            getProbableLabels(image, input, xaxis, yaxis)

        actual_image = image.copy()

        # ================= GET RATIO ===========================
        # Dùng str(path) để chắc chắn cv2.imread trong getRatio không lỗi
        list_text, normalize_ratio = getRatio(str(path), image_text, xaxis, yaxis)

        if not normalize_ratio:
            # Không tính được ratio (thiếu tick Y numeric, v.v.)
            print(f"❌ Không tính được normalize_ratio cho {path.name}, bỏ qua chart này.")
            continue

        print("[getYVal] legends: {0}".format(legends))
        print("[{0}] path: {1}, ratio: {2}".format(index, path.name, normalize_ratio), end='\n\n')

        ## =============== Ignore Text =====================
        chart_json = load_chart_json(json_path)
        img_clean = fill_text_regions(image, chart_json, roles_to_remove=None, pad=2)

        try:
            ## =============== LEGEND ANALYSIS =====================
            legendtexts = []
            legendrects = []
            legendboxes = []

            # bbox nhỏ nằm gần legend_text (YOLO class 'legend')
            for det in legend_dets:
                x1, y1, x2, y2 = det["bbox"]
                legendboxes.append((x1, y1, x2 - x1, y2 - y1))

            # Ghép legend text (từ getProbableLabels) với legend patch (từ YOLO)
            for idx, box in enumerate(legend_boxes):
                text, (textx, texty, width, height) = box

                bboxes = filterBbox(legendboxes, box)
                if bboxes is None:
                    print("  ❌ Không tìm được ô màu nào gần legend này.")
                    continue

                (lx, ly, lw, lh) = bboxes
                legendrects.append(bboxes)
                legendtexts.append(text)
                print(f"  - Legend '{text}' có bbox {bboxes}")


            # ========== Xử lý trường hợp KHÔNG CÓ legend ==========
            if not legendtexts:
                # Không detect được legend -> fallback single-series
                print(f"⚠️ Không tìm thấy legend nào cho {path.name}, dùng 'series_0' làm tên series.")
                legendtexts = ["series_0"]

            # Khởi tạo data: mỗi legendtext là 1 series
            data = {
                legend_text: {x_label: 0.0 for x_label, _ in x_labels_list}
                for legend_text in legendtexts
            }

            ## =============== Detect Bboxes bar =====================
            bar_rects = []
            for det in bar_dets:
                x1, y1, x2, y2 = det["bbox"]
                bar_rects.append((x1, y1, x2 - x1, y2 - y1))

            # ========== Trường hợp KHÔNG CÓ bar ==========
            if not bar_rects:
                print(f"❌ Không phát hiện bar nào cho {path.name}, bỏ qua chart.")
                continue

            # ================== EMBEDDING LEGEND & BAR ==========================
            legend_embs_list = []
            for (lx, ly, lw, lh) in legendrects:
                # convert (x, y, w, h) -> (x1, y1, x2, y2)
                bbox_xyxy = (lx, ly, lx + lw, ly + lh)
                try:
                    emb = extract_patch_embedding(actual_image, bbox_xyxy, backbone)
                    legend_embs_list.append(emb)
                except Exception as e:
                    print(f"⚠ Lỗi extract embedding legend patch {bbox_xyxy} trong {path.name}: {e}")

            bar_embs_list = []
            for (bx, by, bw, bh) in bar_rects:
                bbox_xyxy = (bx, by, bx + bw, by + bh)
                try:
                    emb = extract_patch_embedding(actual_image, bbox_xyxy, backbone)
                    bar_embs_list.append(emb)
                except Exception as e:
                    print(f"⚠ Lỗi extract embedding bar patch {bbox_xyxy} trong {path.name}: {e}")

            legend_for_bar = None  # danh sách length B: mỗi phần tử là index legend (0..L-1)

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

                    print(f"  >> {path.name}: Gán {len(bar_rects)} bar cho {len(legendtexts)} legend bằng similarity.")
                except Exception as e:
                    print(f"⚠ Lỗi match_legend_to_bars trong {path.name}: {e}")
                    legend_for_bar = None

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

            # ========== Trường hợp KHÔNG CÓ tick X ==========
            if not x_labels_list:
                # Tạo nhãn giả 1..N dựa trên vị trí bar
                print(f"⚠️ Không phát hiện tick X cho {path.name}, tạo nhãn 1..N theo vị trí bar.")
                sorted_bars = sorted(bar_rects, key=lambda r: r[0] + r[2] / 2.0)
                x_labels_list = [(str(i + 1), rect) for i, rect in enumerate(sorted_bars)]

                # Khởi tạo lại data với x_labels_list mới
                data = {
                    legend_text: {x_label: 0.0 for x_label, _ in x_labels_list}
                    for legend_text in legendtexts
                }

            # ======= Gán mỗi bar → tick X gần nhất (labels) =======
            textBoxes = []
            labels = []  # label text cho từng bar_rect

            for rectBox in bar_rects:
                min_distance = sys.maxsize
                closestBox = None
                labeltext = None

                for text, textBox in x_labels_list:
                    d = RectDist(rectBox, textBox)
                    if d < min_distance:
                        min_distance = d
                        closestBox = textBox
                        labeltext = text

                textBoxes.append(closestBox)
                labels.append(labeltext)

            # Chiều cao từng bar
            list_len = [(rect, float(rect[3])) for rect in bar_rects]

            # y-values = chiều cao * normalize_ratio
            y_val = [(rect, round(l * normalize_ratio, 1)) for rect, l in list_len]

            # ========== Gán y-value cho TỪNG legend ==========
            for legend_idx, legendtext in enumerate(legendtexts):
                print(f"  >> Gán giá trị cho legend '{legendtext}'")

                for x_label, box in x_labels_list:
                    (x, y, w, h) = box
                    value = 0.0
                    closest = None
                    dist = sys.maxsize

                    for idx_bar, item in enumerate(y_val):
                        # BỎ QUA những bar không thuộc legend này
                        if legend_for_bar[idx_bar] != legend_idx:
                            continue

                        if labels[idx_bar] == x_label:
                            (vx, vy, vw, vh) = item[0]
                            cx_bar = vx + vw / 2.0
                            cx_lbl = x + w / 2.0
                            d = abs(cx_lbl - cx_bar)
                            if d < dist:
                                dist = d
                                closest = item[0]
                                value = item[1]

                    data[legendtext][x_label] = value

            yValueDict[path.name] = data

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {path.name}: {e}")
            continue

    return yValueDict

def main():
    # Load the model
    objects_dectector = YOLO(TASK4_CONFIG["yolo_weight"])

    # Khởi tạo backbone 1 lần
    backbone = get_backbone("resnet50")  # hoặc "clip_vitb32", "swin_tiny"

    yValueDict = getYVal(TASK4_CONFIG["input_images"], TASK4_CONFIG["input_json"], objects_dectector, backbone)
    rows = []

    for img_name, legends_dict in yValueDict.items():
        for legend, xdict in legends_dict.items():
            for x_label, val in xdict.items():
                rows.append({
                    "image": img_name,
                    "legend": legend,
                    "x_label": x_label,
                    "value": val
                })

    # Tạo DataFrame
    df = pd.DataFrame(rows)

    # Đường dẫn file Excel (bạn đổi lại cho đúng folder trên Drive của bạn)
    excel_path = TASK4_CONFIG["output_excel"]

    # Lưu ra Excel
    df.to_csv(excel_path, index=False, encoding='utf-8-sig')

    print("Đã lưu kết quả vào:", excel_path)