import torch
import os
import json
import logging
import paddle
from paddleocr import PaddleOCR
import cv2
import numpy as np

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (BẠN CHỈNH Ở ĐÂY)
# ==========================================

# 1. Thư mục chứa ảnh gốc
INPUT_IMG_DIR = r"E:\Hcmut material\Project_1\Dataset_Test\images"  

# 2. Thư mục chứa file JSON (Kết quả bbox từ YOLO)
INPUT_JSON_DIR = r"E:\Hcmut material\Project_1\Dataset_Test\json" 

# 3. Thư mục muốn lưu kết quả cuối cùng (JSON + Ảnh Vis)
OUTPUT_DIR = r"E:\Hcmut material\Project_1\output"

# Các đuôi ảnh hỗ trợ
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Tắt log rác của Paddle
logging.getLogger("ppocr").setLevel(logging.WARNING)

# ==========================================
# 2. CÁC HÀM TIỆN ÍCH (CẮT & XOAY ẢNH)
# ==========================================
def parse_polygon_from_dict(poly_dict):
    """Chuyển đổi dict {x0, y0...} sang numpy array"""
    return np.array([
        [poly_dict["x0"], poly_dict["y0"]],
        [poly_dict["x1"], poly_dict["y1"]],
        [poly_dict["x2"], poly_dict["y2"]],
        [poly_dict["x3"], poly_dict["y3"]]
    ], dtype=np.float32)

def get_rotate_crop_image(img, points):
    """Cắt và xoay ảnh theo 4 điểm (fix nghiêng)"""
    # Tính chiều rộng và cao của box mới
    width_top = np.linalg.norm(points[0] - points[1])
    width_bottom = np.linalg.norm(points[2] - points[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(points[0] - points[3])
    height_right = np.linalg.norm(points[1] - points[2])
    max_height = int(max(height_left, height_right))

    # Điểm đích
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Biến đổi và cắt
    M = cv2.getPerspectiveTransform(points, dst_pts)
    dst_img = cv2.warpPerspective(img, M, (max_width, max_height))
    return dst_img

def read_image_windows(path):
    """Đọc ảnh hỗ trợ đường dẫn tiếng Việt/Unicode"""
    if not os.path.exists(path):
        print(f"  [ERR] File không tồn tại: {path}")
        return None
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_bgr is None: return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  [ERR] Lỗi đọc ảnh: {e}")
        return None

# ==========================================
# 3. KHỞI TẠO MODEL
# ==========================================
def init_model():
    print("--- Đang khởi tạo Model PaddleOCR ---")
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(" -> [OK] Đã kích hoạt chế độ GPU.")
        else:
            paddle.device.set_device("cpu")
            print(" -> [WARN] Chạy trên CPU.")
    except Exception:
        pass

    model = PaddleOCR(
        lang="en",
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # show_log=False 
    )
    return model

# ==========================================
# 4. XỬ LÝ CHÍNH
# ==========================================
def process_single_image_yolo(ocr_model, img_path, json_path):
    # Đọc ảnh
    img = read_image_windows(img_path)
    if img is None: return [], None

    # Đọc JSON YOLO
    if not os.path.exists(json_path):
        print("  -> [SKIP] Không thấy file JSON YOLO tương ứng.")
        return [], img
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], img

    if "task2" not in data or "output" not in data["task2"]:
        return [], img
    
    input_blocks = data["task2"]["output"]["text_blocks"]
    final_blocks = []

    # Duyệt từng box YOLO
    for idx, block in enumerate(input_blocks):
        try:
            poly_dict = block["polygon"]
            poly_points = parse_polygon_from_dict(poly_dict)

            # Cắt ảnh
            crop_img = get_rotate_crop_image(img, poly_points)

            if crop_img is None or crop_img.shape[0] < 2 or crop_img.shape[1] < 2:
                final_blocks.append(block)
                continue

            # --- GỌI TRỰC TIẾP RECOGNIZER (FIX LỖI DET) ---
            rec_results = ocr_model.text_recognizer([crop_img])

            rec_text = ""
            rec_score = 0.0

            if rec_results and len(rec_results) > 0:
                res_item = rec_results[0]
                if isinstance(res_item, (list, tuple)):
                    rec_text = res_item[0]
                    rec_score = res_item[1]

            # Cập nhật kết quả
            new_block = block.copy()
            new_block["text"] = rec_text
            new_block["score"] = round(float(rec_score), 4)
            final_blocks.append(new_block)

        except Exception as e:
            final_blocks.append(block)

    return final_blocks, img

# ==========================================
# 5. VISUALIZATION & SAVE
# ==========================================
def visualize_and_save(img_rgb, blocks, filename, output_dir):
    if img_rgb is None or not blocks: return
    vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for b in blocks:
        poly = b["polygon"]
        pts = np.array([
            [poly["x0"], poly["y0"]], [poly["x1"], poly["y1"]],
            [poly["x2"], poly["y2"]], [poly["x3"], poly["y3"]]
        ], np.int32).reshape((-1, 1, 2))
        
        # Vẽ box xanh
        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Vẽ text đỏ
        if b.get("text"):
            cv2.putText(vis_img, b["text"], (pts[0][0][0], pts[0][0][1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, vis_img)

def save_json(data, output_path):
    final_output = {
        "task2": {
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "name": "YOLO-OBB + PaddleRec",
            "output": {"text_blocks": data},
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

# ==========================================
# 6. MAIN
# ==========================================
def main():
    # Kiểm tra đường dẫn input
    if not os.path.exists(INPUT_IMG_DIR):
        print(f"[LỖI] Thư mục ảnh không tồn tại: {INPUT_IMG_DIR}")
        return
    if not os.path.exists(INPUT_JSON_DIR):
        print(f"[LỖI] Thư mục JSON YOLO không tồn tại: {INPUT_JSON_DIR}")
        return

    # Tạo các thư mục output
    vis_output_dir = os.path.join(OUTPUT_DIR, "visualize_images")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(vis_output_dir): os.makedirs(vis_output_dir)

    # Init model
    ocr = init_model()

    # Lấy danh sách ảnh
    files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    total = len(files)

    print(f"\nTìm thấy {total} ảnh.")
    print("-" * 50)

    for idx, filename in enumerate(files):
        img_path = os.path.join(INPUT_IMG_DIR, filename)
        
        # Tạo đường dẫn JSON tương ứng (giả định cùng tên file)
        # Ví dụ: anh1.jpg -> tìm file anh1.json trong thư mục JSON
        json_name = os.path.splitext(filename)[0] + ".json"
        json_input_path = os.path.join(INPUT_JSON_DIR, json_name)
        
        # Đường dẫn lưu kết quả
        json_output_path = os.path.join(OUTPUT_DIR, json_name)

        print(f"[{idx + 1}/{total}] Đang xử lý: {filename}")

        # Chạy pipeline
        blocks, img_rgb = process_single_image_yolo(ocr, img_path, json_input_path)
        
        # Lưu kết quả
        if blocks:
            save_json(blocks, json_output_path)

        # Lưu ảnh visualize
        if img_rgb is not None and len(blocks) > 0:
            visualize_and_save(img_rgb, blocks, filename, vis_output_dir)

    print("-" * 50)
    print("Hoàn tất.")
    print(f"File JSON kết quả tại: {OUTPUT_DIR}")
    print(f"Ảnh Visualize tại: {vis_output_dir}")

if __name__ == "__main__":
    main()