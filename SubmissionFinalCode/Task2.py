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
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

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
    except:
        pass

    # Cấu hình "chiến thắng" (giống tmp.py)
    model = PaddleOCR(
        lang='en',
        ocr_version='PP-OCRv4',             
        use_textline_orientation=True,      
        use_doc_orientation_classify=False, 
        # show_log=False
    )
    print(" -> [OK] Khởi tạo model thành công.")
    return model

# ==========================================
# 3. HÀM ĐỌC ẢNH (CÓ DEBUG CHI TIẾT)
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
# 4. XỬ LÝ DỮ LIỆU
# ==========================================
def process_single_image(ocr_model, img_path):
    import numpy as np
    
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

    # Lấy đối tượng kết quả đầu tiên
    ocr_res = result[0]

    # Lấy dữ liệu từ các Key riêng biệt
    # Lưu ý: Các key này là dạng List song song nhau
    # 'rec_texts': ['Text A', 'Text B', ...]
    # 'rec_scores': [0.99, 0.98, ...]
    # 'dt_polys': [numpy_array_box_A, numpy_array_box_B, ...]
    
    # Kiểm tra xem có key chứa text không
    if 'rec_texts' not in ocr_res or 'dt_polys' not in ocr_res:
        # Trường hợp trả về kiểu cũ (list lồng nhau) - Fallback
        if isinstance(ocr_res, list):
             # Logic cũ nếu nhỡ đâu model trả về kiểu cũ
             return _process_legacy_format(ocr_res)
        print("  [WARN] Không tìm thấy key 'rec_texts' hoặc 'dt_polys'.")
        return []

    texts = ocr_res['rec_texts']
    scores = ocr_res['rec_scores']
    boxes = ocr_res['dt_polys']

    # Nếu không tìm thấy chữ nào
    if texts is None or len(texts) == 0:
        return []

    text_blocks = []
    
    # Duyệt qua các list song song
    for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        try:
            pts = []
            for p in box:
                pts.append([int(p[0]), int(p[1])])

            if len(pts) < 4: continue

            polygon = {
                "x0": pts[0][0], "x1": pts[1][0], "x2": pts[2][0], "x3": pts[3][0],
                "y0": pts[0][1], "y1": pts[1][1], "y2": pts[2][1], "y3": pts[3][1]
            }

            text_blocks.append({
                "id": i,
                "polygon": polygon,
                "text": text,
                "score": round(score, 4)
            })
            
        except Exception as e:
            print(f"  [WARN] Lỗi xử lý item thứ {i}: {e}")
            continue
            
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
                "x0": pts[0][0], "x1": pts[1][0], "x2": pts[2][0], "x3": pts[3][0],
                "y0": pts[0][1], "y1": pts[1][1], "y2": pts[2][1], "y3": pts[3][1]
            }
            blocks.append({"id": i, "polygon": polygon, "text": text})
        except: pass
    return blocks

def save_json(data, output_path):
    final_output = {
        "task2": {
            "input": {
                "task1_output": {
                    "chart_type": "vertical bar" 
                }
            },
            "name": "Text Detection and Recognition",
            "output": {
                "text_blocks": data
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

# ==========================================
# 5. MAIN (Dùng để test độc lập nếu cần)
# ==========================================
def main():
    if not os.path.exists(Task2_Config["output"]):
        os.makedirs(Task2_Config["output"])
    
    ocr = init_model()

    files = [f for f in os.listdir(Task2_Config["input"]) if f.lower().endswith(VALID_EXTENSIONS)]
    total_files = len(files)
    
    print(f"\nTìm thấy {total_files} ảnh.")
    print("-" * 50)

    for idx, filename in enumerate(files):
        img_full_path = os.path.join(Task2_Config["input"], filename)
        json_full_path = os.path.join(Task2_Config["output"], os.path.splitext(filename)[0] + ".json")

        print(f"[{idx+1}/{total_files}] Processing: {filename} ... ", end="")

        blocks = process_single_image(ocr, img_full_path)
        
        save_json(blocks, json_full_path)
        print(f"-> Xong! ({len(blocks)} texts)")

    print("-" * 50)
    print(f"Hoàn tất. Kiểm tra tại: {Task2_Config['output']}")

if __name__ == "__main__":
    main()