import os
import json
import logging
import paddle
from paddleocr import PaddleOCR
import Config
import cv2
import numpy as np

# ==========================================
# 1. CẤU HÌNH
# ==========================================
Task2_Config = Config.returnTestTask2_Config()
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

logging.getLogger("ppocr").setLevel(logging.WARNING)


# ==========================================
# 2. ĐỊNH NGHĨA MODEL
# ==========================================

def init_model():
    print("--- Đang khởi tạo Model PaddleOCR (Final Version) ---")
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(" -> [OK] Đã kích hoạt chế độ GPU.")
        else:
            paddle.device.set_device("cpu")
            print(" -> [WARN] Chạy trên CPU.")
    except Exception:
        # Nếu có lỗi khi set device thì bỏ qua, PaddleOCR sẽ tự xử lý
        pass

    # Cấu hình "chiến thắng" (giống tmp.py)
    model = PaddleOCR(
        lang="en",
        # ocr_version="PP-OCRv5",
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # show_log=False,
    )
    print(" -> [OK] Khởi tạo model thành công.")
    return model


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
    return min(xs), min(ys), max(xs), max(ys)


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
# 5. XỬ LÝ DỮ LIỆU
# ==========================================

def process_single_image(ocr_model, img_path):
    # 1. Đọc ảnh
    img = read_image_windows(img_path)
    if img is None:
        print("  -> [SKIP] Bỏ qua do lỗi đọc ảnh.")
        return []

    try:
        # 2. Chạy model
        result = ocr_model.ocr(img)
    except Exception as e:
        print(f"  [!] Lỗi model: {e}")
        return []

    # Kiểm tra kết quả rỗng
    if result is None or len(result) == 0:
        return []

    ocr_res = result[0]

    # Trường hợp trả về dạng dict mới
    if isinstance(ocr_res, dict) and "rec_texts" in ocr_res and "dt_polys" in ocr_res:
        texts = ocr_res["rec_texts"]
        scores = ocr_res["rec_scores"]
        boxes = ocr_res["dt_polys"]

        if texts is None or len(texts) == 0:
            return []

        text_blocks = []
        for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
            try:
                pts = [[int(p[0]), int(p[1])] for p in box]
                if len(pts) < 4:
                    continue

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

                text_blocks.append(
                    {
                        "id": i,
                        "polygon": polygon,
                        "text": text,
                        "score": round(score, 4),
                    }
                )
            except Exception as e:
                print(f"  [WARN] Lỗi xử lý item thứ {i}: {e}")
                continue

        # === BƯỚC QUAN TRỌNG: GỘP CÁC DÒNG LEGEND NHIỀU DÒNG ===
        img_h, img_w = img.shape[:2]
        merged_blocks = merge_multiline_text_blocks(text_blocks, img_width=img_w)
        return merged_blocks

    # Fallback cho format cũ (list lồng nhau)
    if isinstance(ocr_res, list):
        legacy_blocks = _process_legacy_format(ocr_res)
        img_h, img_w = img.shape[:2]
        merged_blocks = merge_multiline_text_blocks(legacy_blocks, img_width=img_w)
        return merged_blocks

    print("  [WARN] Định dạng trả về không hỗ trợ.")
    return []


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
