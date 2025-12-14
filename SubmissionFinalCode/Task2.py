# import torch
# import os
# import json
# import logging
# import paddle
# from paddleocr import PaddleOCR
# import cv2
# import numpy as np
# import Config

# # ==========================================
# # 1. CẤU HÌNH ĐƯỜNG DẪN
# # ==========================================

# Task_2_config = Config.returnTestTask2_Config()

# INPUT_IMG_DIR = Task_2_config["input_images"] 
# INPUT_JSON_DIR = Task_2_config["input_json"] 
# OUTPUT_DIR = Task_2_config["output"] 

# # Các đuôi ảnh hỗ trợ
# VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# # Tắt log rác của Paddle
# logging.getLogger("ppocr").setLevel(logging.WARNING)

# # ==========================================
# # 2. CÁC HÀM TIỆN ÍCH (CẮT & XOAY ẢNH)
# # ==========================================

# def parse_polygon_from_dict(poly_dict):
#     """Chuyển đổi dict {x0, y0...} sang numpy array"""
#     return np.array([
#         [poly_dict["x0"], poly_dict["y0"]],
#         [poly_dict["x1"], poly_dict["y1"]],
#         [poly_dict["x2"], poly_dict["y2"]],
#         [poly_dict["x3"], poly_dict["y3"]]
#     ], dtype=np.float32)

# def sorted_boxes(dt_boxes):
#     """
#     [FIX] Hàm này bị thiếu trong code của bạn.
#     Sắp xếp lại thứ tự 4 điểm: TL, TR, BR, BL
#     """
#     num_points = dt_boxes.shape[0]
#     sorted_points = sorted(dt_boxes, key=lambda x: x[0])
#     left_points = sorted_points[:2]
#     right_points = sorted_points[2:]

#     if left_points[0][1] < left_points[1][1]:
#         tl = left_points[0]
#         bl = left_points[1]
#     else:
#         tl = left_points[1]
#         bl = left_points[0]

#     if right_points[0][1] < right_points[1][1]:
#         tr = right_points[0]
#         br = right_points[1]
#     else:
#         tr = right_points[1]
#         br = right_points[0]
    
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# def get_rotate_crop_image(img, points):
#     """Cắt và xoay ảnh theo 4 điểm (fix nghiêng)"""
#     # Tính chiều rộng và cao của box mới
#     width_top = np.linalg.norm(points[0] - points[1])
#     width_bottom = np.linalg.norm(points[2] - points[3])
#     max_width = int(max(width_top, width_bottom))

#     height_left = np.linalg.norm(points[0] - points[3])
#     height_right = np.linalg.norm(points[1] - points[2])
#     max_height = int(max(height_left, height_right))

#     # Điểm đích
#     dst_pts = np.array([
#         [0, 0],
#         [max_width - 1, 0],
#         [max_width - 1, max_height - 1],
#         [0, max_height - 1]
#     ], dtype=np.float32)

#     # Biến đổi và cắt
#     M = cv2.getPerspectiveTransform(points, dst_pts)
#     dst_img = cv2.warpPerspective(img, M, (max_width, max_height))
#     return dst_img

# def read_image_windows(path):
#     """Đọc ảnh hỗ trợ đường dẫn tiếng Việt/Unicode"""
#     if not os.path.exists(path):
#         print(f"  [ERR] File không tồn tại: {path}")
#         return None
#     try:
#         stream = np.fromfile(path, dtype=np.uint8)
#         img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
#         if img_bgr is None: return None
#         return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     except Exception as e:
#         print(f"  [ERR] Lỗi đọc ảnh: {e}")
#         return None

# def expand_polygon(points, img_height, img_width, scale_ratio=1.1):
#     """Nới rộng polygon từ tâm ra các phía."""
#     center = np.mean(points, axis=0)
#     vectors = points - center
#     expanded_points = center + vectors * scale_ratio
    
#     # Clip tọa độ
#     expanded_points[:, 0] = np.clip(expanded_points[:, 0], 0, img_width - 1)
#     expanded_points[:, 1] = np.clip(expanded_points[:, 1], 0, img_height - 1)
    
#     return expanded_points.astype(np.float32)

# # ==========================================
# # 3. KHỞI TẠO MODEL
# # ==========================================
# def init_model():
#     print("--- Đang khởi tạo Model PaddleOCR ---")
#     try:
#         if paddle.is_compiled_with_cuda():
#             paddle.device.set_device("gpu")
#             print(" -> [OK] Đã kích hoạt chế độ GPU.")
#         else:
#             paddle.device.set_device("cpu")
#             print(" -> [WARN] Chạy trên CPU.")
#     except Exception:
#         pass

#     model = PaddleOCR(
#         lang="en",
#         use_textline_orientation=True,
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#     )
#     return model

# # ==========================================
# # 4. XỬ LÝ CHÍNH
# # ==========================================
# def process_single_image_yolo(ocr_model, img_path, json_path):
#     img = read_image_windows(img_path)
#     if img is None: return [], None

#     if not os.path.exists(json_path):
#         return [], img
    
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception:
#         return [], img

#     if "task2" not in data or "output" not in data["task2"]:
#         return [], img
    
#     input_blocks = data["task2"]["output"]["text_blocks"]
#     final_blocks = []

#     img_h, img_w = img.shape[:2]

#     # Tạo folder debug để kiểm tra xem model nhìn thấy gì (Quan trọng)
#     debug_dir = "debug_crops"
#     if not os.path.exists(debug_dir): os.makedirs(debug_dir)

#     for idx, block in enumerate(input_blocks):
#         try:
#             poly_dict = block["polygon"]
#             poly_points = parse_polygon_from_dict(poly_dict)

#             # ---------------------------------------------------------
#             # CHECK 1: Kiểm tra xem tọa độ có bị chuẩn hóa (0.0 - 1.0) không?
#             # Nếu tọa độ toàn < 1.0 nghĩa là đang sai hệ quy chiếu
#             if np.max(poly_points) <= 1.5: 
#                 print(f"[WARN] Tọa độ có vẻ là Normalized. Đang scale lại theo ảnh {img_w}x{img_h}")
#                 poly_points[:, 0] *= img_w
#                 poly_points[:, 1] *= img_h
#             # ---------------------------------------------------------

#             # 1. Nới rộng box (Box Dilation) - Giữ mức 1.1 là an toàn
#             poly_points = expand_polygon(poly_points, img_h, img_w, scale_ratio=1.1)
            
#             # 2. Sắp xếp điểm
#             poly_points = sorted_boxes(poly_points)
            
#             # 3. Cắt ảnh
#             crop_img = get_rotate_crop_image(img, poly_points)

#             if crop_img is None or crop_img.shape[0] < 4 or crop_img.shape[1] < 4:
#                 final_blocks.append(block)
#                 continue

#             # C. Lưu ảnh ra để debug (Xóa dòng này khi đã chạy ngon)
#             cv2.imwrite(f"{debug_dir}/crop_{idx}.jpg", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

#             # 4. Gọi OCR
#             crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
#             rec_results = ocr_model.ocr(crop_bgr)

#             rec_text = ""
#             rec_score = 0.0

#             if rec_results and isinstance(rec_results, list) and len(rec_results) > 0:
#                 # Lấy khối kết quả đầu tiên (là một list chứa các kết quả từng dòng)
#                 first_result = rec_results[0]
                
#                 # PaddleOCR trả về một list chứa các cặp [tọa độ, (text, score)]
#                 if isinstance(first_result, list):
                    
#                     all_text = []
#                     max_score = 0.0
                    
#                     for item in first_result:
#                         # item là một list, ví dụ: [ [1, 1], [1, 8], ('9', 0.4278) ]
#                         # Lấy phần tử cuối cùng, là tuple ('Text', Score)
#                         if isinstance(item, list) and len(item) > 1 and isinstance(item[-1], tuple):
#                             text, score = item[-1]
                            
#                             # Gom text lại
#                             all_text.append(text)
                            
#                             # Cập nhật score cao nhất
#                             score = float(score)
#                             if score > max_score:
#                                 max_score = score
                                
#                     # Gán kết quả cuối cùng
#                     rec_text = " ".join(all_text)
#                     rec_score = max_score

#             # print(rec_results)

#             # In ra màn hình để xem nó có đọc được không
#             # print(f" - Block {idx}: '{rec_text}' ({rec_score:.2f})")

#             new_block = block.copy()
#             new_block["text"] = rec_text
#             new_block["score"] = round(float(rec_score), 4)
#             final_blocks.append(new_block)

#         except Exception as e:
#             print(f"Lỗi block {idx}: {e}")
#             final_blocks.append(block)

#     return final_blocks, img

# # ==========================================
# # 5. VISUALIZATION & SAVE
# # ==========================================

# def visualize_and_save(img_rgb, blocks, filename, output_dir):
#     if img_rgb is None or not blocks: return
#     vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#     for b in blocks:
#         poly = b["polygon"]
#         pts = np.array([
#             [poly["x0"], poly["y0"]], [poly["x1"], poly["y1"]],
#             [poly["x2"], poly["y2"]], [poly["x3"], poly["y3"]]
#         ], np.int32).reshape((-1, 1, 2))
        
#         cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
#         if b.get("text"):
#             # Lưu ý: cv2.putText không hỗ trợ tiếng Việt có dấu
#             cv2.putText(vis_img, b["text"], (pts[0][0][0], pts[0][0][1] - 5), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

#     save_path = os.path.join(output_dir, filename)
#     cv2.imwrite(save_path, vis_img)

# def save_json(data, output_path):
#     final_output = {
#         "task2": {
#             "input": {"task1_output": {"chart_type": "vertical bar"}},
#             "name": "Text Detection and Recognition",
#             "output": {"text_blocks": data},
#         }
#     }
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, ensure_ascii=False, indent=4)

# # ==========================================
# # 6. MAIN
# # ==========================================
# def main():
#     if not os.path.exists(INPUT_IMG_DIR):
#         print(f"[LỖI] Thư mục ảnh không tồn tại: {INPUT_IMG_DIR}")
#         return
#     if not os.path.exists(INPUT_JSON_DIR):
#         print(f"[LỖI] Thư mục JSON YOLO không tồn tại: {INPUT_JSON_DIR}")
#         return

#     vis_output_dir = os.path.join(OUTPUT_DIR, "visualize_images")
#     if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
#     if not os.path.exists(vis_output_dir): os.makedirs(vis_output_dir)

#     ocr = init_model()

#     files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
#     total = len(files)

#     print(f"\nTìm thấy {total} ảnh.")
#     print("-" * 50)

#     for idx, filename in enumerate(files):
#         img_path = os.path.join(INPUT_IMG_DIR, filename)
        
#         json_name = os.path.splitext(filename)[0] + ".json"
#         json_input_path = os.path.join(INPUT_JSON_DIR, json_name)
#         json_output_path = os.path.join(OUTPUT_DIR, json_name)

#         print(f"[{idx + 1}/{total}] Đang xử lý: {filename}")

#         blocks, img_rgb = process_single_image_yolo(ocr, img_path, json_input_path)
        
#         if blocks:
#             save_json(blocks, json_output_path)

#         if img_rgb is not None and len(blocks) > 0:
#             visualize_and_save(img_rgb, blocks, filename, vis_output_dir)

#     print("-" * 50)
#     print("Hoàn tất.")
#     print(f"File JSON kết quả tại: {OUTPUT_DIR}")
#     print(f"Ảnh Visualize tại: {vis_output_dir}")

# -*- coding: utf-8 -*-
"""
Task2 (YOLO -> PaddleOCR 3.x Recognition-only)

- Input: Ảnh + JSON từ YOLO (task2/output/text_blocks với polygon x0..y3)
- Output: JSON bổ sung text + score cho từng block + ảnh visualize + debug crops

Lưu ý quan trọng (PaddleOCR 3.x):
- PaddleOCR.ocr() không còn tham số det/rec. Nếu bạn muốn recognition-only sau YOLO,
  hãy dùng module TextRecognition (và TextLineOrientationClassification nếu cần).
"""

import os
import json
import logging

import cv2
import numpy as np
import paddle

import Config
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
Task2_Config = Config.returnTestTask2_Config()
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
YOLO_TEXT_WEIGHT = "./weights/best_det.pt"
PAD_EXPAND_PX = 4  # expand detector boxes a bit for recognition

# Tắt log rác của PaddleOCR
logging.getLogger("ppocr").setLevel(logging.WARNING)



# ==========================================
# 2. THAM SỐ TIỀN XỬ LÝ CROP (khuyến nghị bật)
# ==========================================

def init_model():
    """Load YOLO OBB detector + PaddleOCR recognizer-only."""
    print("--- Khởi tạo YOLOv8-OBB text detector + PaddleOCR recognizer ---")
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(" -> [OK] Paddle GPU.")
        else:
            paddle.device.set_device("cpu")
            print(" -> [WARN] Paddle CPU.")
    except Exception:
        pass

    detector = YOLO(YOLO_TEXT_WEIGHT)
    recognizer = PaddleOCR(
        lang="en",
        use_angle_cls=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # show_log=False,
    )

    print(" -> [OK] Khởi tạo model thành công.")
    return {"detector": detector, "recognizer": recognizer}


# ==========================================
# 3. ĐỌC ẢNH
# ==========================================

def read_image_windows(path):
    if not os.path.exists(path):
        print(f"  [ERR] File không tồn tại: {path}")
        return None

    if os.path.getsize(path) == 0:
        print(f"  [ERR] File rỗng (0 KB): {path}")
        return None

    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"  [ERR] Lỗi đọc ảnh: {e}")
        return None


# ==========================================
# 4. HỖ TRỢ GỘP DÒNG MULTI-LINE
# ==========================================

def _bbox_from_polygon(poly: dict):
    """Convert polygon dict (x0..x3, y0..y3) -> (x_min, y_min, x_max, y_max)."""
    xs = [poly[f"x{i}"] for i in range(4)]
    ys = [poly[f"y{i}"] for i in range(4)]
    # return min(xs), min(ys), max(xs), max(ys)
    return poly


def merge_multiline_text_blocks(text_blocks, img_width=None, x_tol_ratio=0.03, y_gap_ratio=0.14):
    """Gộp các text block nằm cùng cột và rất sát theo trục dọc thành 1 block.

    Mục tiêu: xử lý các legend nhiều dòng như
        "French controls from" + "general population"
    để đưa vào LayoutLMv3 như một text duy nhất.

    Heuristic:
      - Cùng cột: |x_min1 - x_min2| < x_tol_px
      - Khoảng cách dọc nhỏ: 0 <= dy <= avg_height * y_gap_ratio
      - Chỉ áp dụng cho vùng legend (nửa phải ảnh) để tránh ảnh hưởng trục Y.
    """
    if not text_blocks:
        return []

    # Ngưỡng lệch cột tính theo % chiều rộng ảnh
    if img_width is not None and img_width > 0:
        x_tol_px = max(5, int(img_width * x_tol_ratio))
    else:
        x_tol_px = 15

    # Chuẩn bị bbox & height
    aug_blocks = []
    heights = []
    for b in text_blocks:
        x_min, y_min, x_max, y_max = _bbox_from_polygon(b["polygon"])
        h = max(1, y_max - y_min)
        heights.append(h)
        aug = dict(b)
        aug["_bbox"] = (x_min, y_min, x_max, y_max)
        aug["_height"] = h
        aug_blocks.append(aug)

    avg_h = sum(heights) / len(heights) if heights else 0
    max_gap = avg_h * y_gap_ratio if avg_h > 0 else 0

    # Sắp xếp theo thứ tự đọc: y tăng, rồi x tăng
    aug_blocks.sort(key=lambda b: (b["_bbox"][1], b["_bbox"][0]))

    merged = []
    used = set()

    for i, cur in enumerate(aug_blocks):
        if i in used:
            continue

        x_min, y_min, x_max, y_max = cur["_bbox"]
        text = cur["text"]
        score = cur.get("score", 1.0)

        # Chỉ merge trong vùng legend (phần bên phải ảnh)
        if img_width is not None and x_min < img_width * 0.5:
            # Không merge, giữ nguyên block
            new_poly = {
                "x0": x_min,
                "y0": y_min,
                "x1": x_max,
                "y1": y_min,
                "x2": x_max,
                "y2": y_max,
                "x3": x_min,
                "y3": y_max,
            }
            merged.append({
                "polygon": new_poly,
                "text": text,
                "score": round(score, 4),
            })
            continue

        # Thử gộp với các dòng phía dưới cùng cột
        for j in range(i + 1, len(aug_blocks)):
            if j in used:
                continue

            cand = aug_blocks[j]
            x2_min, y2_min, x2_max, y2_max = cand["_bbox"]

            # Nếu đã xa hơn nhiều dòng thì dừng sớm
            if avg_h > 0 and (y2_min - y_max) > avg_h * 1.5:
                break

            same_col = abs(x2_min - x_min) <= x_tol_px
            dy = y2_min - y_max
            small_gap = 0 <= dy <= max_gap

            if same_col and small_gap:
                # Gộp text và bbox
                text = text + " " + cand["text"]
                x_min = min(x_min, x2_min)
                y_min = min(y_min, y2_min)
                x_max = max(x_max, x2_max)
                y_max = max(y_max, y2_max)
                score = max(score, cand.get("score", 1.0))
                used.add(j)

        new_poly = {
            "x0": x_min,
            "y0": y_min,
            "x1": x_max,
            "y1": y_min,
            "x2": x_max,
            "y2": y_max,
            "x3": x_min,
            "y3": y_max,
        }
        merged.append({
            "polygon": new_poly,
            "text": text,
            "score": round(score, 4),
        })

    # Đánh lại id liên tục từ 0..n-1
    final_blocks = []
    for idx, b in enumerate(merged):
        final_blocks.append({
            "id": idx,
            "polygon": b["polygon"],
            "text": b["text"],
            "score": b.get("score", 1.0),
        })

    return final_blocks


# ==========================================
# 5. PREPROCESS IMAGE FOR SMALL TEXT
# ==========================================
def preprocess_for_ocr(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Light preprocessing to boost tiny text:
      - Upscale (1.3–2x up to ~1800 px long side).
      - CLAHE for local contrast.
      - Light denoise + unsharp mask.
      - Adaptive threshold if contrast is still low.
    Returns RGB 3-channel for PaddleOCR; falls back to original on errors.
    """
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        long_side = max(h, w)
        target = 1800
        scale = min(2.0, max(1.3, target / long_side)) if long_side < target else 1.0
        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        if np.std(sharp) < 50:
            sharp = cv2.adaptiveThreshold(
                sharp,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                17,
                8,
            )

        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB), scale
    except Exception:
        return img_rgb, 1.0


# Helper: upscale + light enhance for recognition crops
def prep_crop_for_rec(crop_rgb: np.ndarray) -> np.ndarray:
    try:
        h, w = crop_rgb.shape[:2]
        if h == 0 or w == 0:
            return crop_rgb
        long_side = max(h, w)
        target = 512
        scale = min(2.5, max(1.5, target / long_side))
        if scale > 1.0:
            crop_rgb = cv2.resize(
                crop_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        # slight sharpen
        blur = cv2.GaussianBlur(crop_rgb, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(crop_rgb, 1.3, blur, -0.3, 0)
        return sharp
    except Exception:
        return crop_rgb


# Create a rectangular crop by masking the polygon area and padding outside with white.
def mask_crop_from_polygon(img: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    try:
        poly = np.asarray(poly, dtype=np.int32)
        if poly.shape[0] < 4:
            return None
        x_min = max(int(np.min(poly[:, 0])), 0)
        x_max = min(int(np.max(poly[:, 0])), img.shape[1] - 1)
        y_min = max(int(np.min(poly[:, 1])), 0)
        y_max = min(int(np.max(poly[:, 1])), img.shape[0] - 1)
        if x_max <= x_min or y_max <= y_min:
            return None
        crop = img[y_min:y_max, x_min:x_max]
        # Shift polygon to crop coordinates
        shifted = poly - np.array([x_min, y_min])
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], 255)
        bg = np.full_like(crop, 255)
        # copy polygon pixels, keep white elsewhere
        crop_masked = bg.copy()
        crop_masked[mask == 255] = crop[mask == 255]
        return crop_masked
    except Exception:
        return None


# Expand a rotated polygon radially from its centroid to preserve orientation.
def expand_polygon(pts: np.ndarray, pad: float, max_w: int, max_h: int) -> np.ndarray:
    try:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 4:
            return pts
        cx, cy = pts.mean(axis=0)
        expanded = []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx, dy)
            if dist == 0:
                nx, ny = x, y
            else:
                scale = (dist + pad) / dist
                nx = cx + dx * scale
                ny = cy + dy * scale
            nx = min(max(nx, 0), max_w - 1)
            ny = min(max(ny, 0), max_h - 1)
            expanded.append([nx, ny])
        return np.asarray(expanded, dtype=np.int32)
    except Exception:
        return np.asarray(pts, dtype=np.int32)


# ==========================================
# 6. XỬ LÝ DỮ LIỆU
# ==========================================

def process_single_image(ocr_model, img_path):
    # 1. Đọc ảnh
    img = read_image_windows(img_path)
    if img is None:
        print("  -> [SKIP] Bỏ qua do lỗi đọc ảnh.")
        return []

    orig_h, orig_w = img.shape[:2]

    # 1.1 Làm rõ ảnh trước khi OCR
    img, scale = preprocess_for_ocr(img)

    detector = ocr_model["detector"]
    recognizer = ocr_model["recognizer"]

    try:
        results = detector.predict(img, verbose=False)
    except Exception as e:
        print(f"  [!] Lỗi detector: {e}")
        return []

    text_blocks = []

    for res in results:
        polys = []
        scores = []

        obb = getattr(res, "obb", None)
        if obb is not None and hasattr(obb, "xyxyxyxy"):
            polys = obb.xyxyxyxy.cpu().numpy()
            scores = obb.conf.cpu().numpy()
        else:
            boxes = res.boxes
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                polys.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                scores.append(float(b.conf[0]))

        for poly_idx, (poly, det_score) in enumerate(zip(polys, scores)):
            try:
                poly_arr = np.array(poly)
                # Rescale coords back to original image
                pts = [[int(round(p[0] / scale)), int(round(p[1] / scale))] for p in poly_arr]
                if len(pts) < 4:
                    continue

                # Expand polygon in processed-image space and crop with masking (keeps orientation, pads outside white)
                exp_proc = expand_polygon(poly_arr, PAD_EXPAND_PX, img.shape[1], img.shape[0])
                crop = mask_crop_from_polygon(img, exp_proc)
                if crop is None or crop.size == 0:
                    continue
                crop = prep_crop_for_rec(crop)

                rec_text = ""
                rec_score = 0.0
                try:
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    rec_res = recognizer.ocr(crop_bgr)
                    if rec_res:
                        # PaddleOCR >=3.x returns list of dicts, each dict can hold multiple lines.
                        if isinstance(rec_res[0], dict):
                            texts_all = []
                            scores_all = []
                            for entry in rec_res:
                                texts_all.extend(entry.get("rec_texts") or [])
                                scores_all.extend(entry.get("rec_scores") or [])
                            if texts_all:
                                rec_text = " ".join(texts_all).strip()
                            if scores_all:
                                rec_score = float(max(scores_all))

                        # Older PaddleOCR returns [[poly, (text, score)], ...]
                        elif isinstance(rec_res[0], list):
                            texts_all = []
                            scores_all = []
                            for cand in rec_res:
                                if isinstance(cand, list) and len(cand) >= 2:
                                    txt_part = cand[1]
                                    if isinstance(txt_part, tuple) and len(txt_part) >= 2:
                                        texts_all.append(txt_part[0])
                                        scores_all.append(float(txt_part[1]))
                                    elif isinstance(txt_part, list) and len(txt_part) >= 2:
                                        texts_all.append(txt_part[0])
                                        scores_all.append(float(txt_part[1]))
                            if texts_all:
                                rec_text = " ".join(texts_all).strip()
                            if scores_all:
                                rec_score = float(max(scores_all))
                except Exception:
                    rec_text = ""

                # Preserve oriented polygon for output (expanded slightly)
                expanded_pts = expand_polygon(pts, PAD_EXPAND_PX, orig_w, orig_h)
                polygon = {
                    "x0": int(expanded_pts[0][0]),
                    "x1": int(expanded_pts[1][0]),
                    "x2": int(expanded_pts[2][0]),
                    "x3": int(expanded_pts[3][0]),
                    "y0": int(expanded_pts[0][1]),
                    "y1": int(expanded_pts[1][1]),
                    "y2": int(expanded_pts[2][1]),
                    "y3": int(expanded_pts[3][1]),
                }

                text_blocks.append(
                    {
                        "id": len(text_blocks),
                        "polygon": polygon,
                        "text": rec_text,
                        "score": round(float(det_score), 4),
                        "rec_score": round(float(rec_score), 4),
                    }
                )
            except Exception as e:
                print(f"  [WARN] Lỗi xử lý box {poly_idx}: {e}")
                continue

    # Gộp các box thẳng hàng dọc (legend nhiều dòng) sau khi đã đưa về toạ độ gốc
    # merged = merge_multiline_text_blocks(text_blocks, img_width=orig_w)
    # return merged
    return text_blocks


# Hàm phụ trợ phòng hờ (giữ lại logic cũ)

def _process_legacy_format(lines):
    blocks = []
    for i, line in enumerate(lines):
        try:
            box = line[0]
            text = line[1][0]
            pts = [[int(p[0]), int(p[1])] for p in box]
            polygon = {
                "x0": pts[0][0],
                "x1": pts[1][0],
                "x2": pts[2][0],
                "x3": pts[3][0],
                "y0": pts[0][1],
                "y1": pts[1][1],
                "y2": pts[2][1],
                "y3": pts[3][1],
            }
            blocks.append({"id": i, "polygon": polygon, "text": text, "score": 1.0})
        except Exception:
            continue
    return blocks


def save_json(data, output_path):
    final_output = {
        "task2": {
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "name": "Text Detection and Recognition",
            "output": {"text_blocks": data},
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)



# ==========================================
# 6. MAIN (Dùng để test độc lập nếu cần)
# ==========================================


def main():
    if not os.path.exists(Task2_Config["output"]):
        os.makedirs(Task2_Config["output"])

    ocr = init_model()

    files = [
        f
        for f in os.listdir(Task2_Config["input"])
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    total_files = len(files)

    print(f"\nTìm thấy {total_files} ảnh.")
    print("-" * 50)

    for idx, filename in enumerate(files):
        img_full_path = os.path.join(Task2_Config["input"], filename)
        json_full_path = os.path.join(
            Task2_Config["output"], os.path.splitext(filename)[0] + ".json"
        )

        print(f"[{idx + 1}/{total_files}] Processing: {filename} ... ", end="")

        blocks = process_single_image(ocr, img_full_path)
        save_json(blocks, json_full_path)
        print(f"-> Xong! ({len(blocks)} texts)")

    print("-" * 50)
    print(f"Hoàn tất. Kiểm tra tại: {Task2_Config['output']}")


if __name__ == "__main__":
    main()

