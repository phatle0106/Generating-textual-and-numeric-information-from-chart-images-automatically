import os
import json
import logging
import paddle
from paddleocr import PaddleOCR
import Config
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH
# ==========================================
Task2_Config = Config.returnTestTask2_Config()
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
YOLO_TEXT_WEIGHT = "./weights/best_det.pt"
PAD_EXPAND_PX = 4  # expand detector boxes a bit for recognition

logging.getLogger("ppocr").setLevel(logging.WARNING)


# ==========================================
# 2. ĐỊNH NGHĨA MODEL
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
            print(f"  [ERR] OpenCV không giải mã được file (Lỗi định dạng ảnh): {path}")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"  [ERR] Ngoại lệ khi đọc ảnh: {e}")
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
