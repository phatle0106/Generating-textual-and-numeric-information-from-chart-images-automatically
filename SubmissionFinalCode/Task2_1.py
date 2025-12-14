import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import Config
import torch # Import để chắc chắn về CUDA

# ==========================================
# 1. CẤU HÌNH
# ==========================================
Task2_1Config = Config.returnTestTask2_1_Config()

MODEL_PATH = Task2_1Config["weight"]
INPUT_IMG_DIR = Task2_1Config["input"]

# Kiểm tra key output trong config của bạn
OUTPUT_DIR = Task2_1Config.get("output", Task2_1Config.get("ouput"))
OUTPUT_JSON_DIR = OUTPUT_DIR # Nơi lưu file JSON

# --- CẤU HÌNH VISUALIZE MỚI ---
OUTPUT_VIS_DIR = os.path.join(OUTPUT_DIR, "visualized_images") # Nơi lưu ảnh vẽ box
ENABLE_VISUALIZE = True # Bật/Tắt chức năng visualize

# Tạo các thư mục output nếu chưa có
if not os.path.exists(OUTPUT_JSON_DIR): os.makedirs(OUTPUT_JSON_DIR)
if ENABLE_VISUALIZE and not os.path.exists(OUTPUT_VIS_DIR): os.makedirs(OUTPUT_VIS_DIR)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# ==========================================
# 2. HÀM HỖ TRỢ (ĐỌC/GHI ẢNH VÀ CONVERT)
# ==========================================
# Hàm đọc ảnh hỗ trợ đường dẫn tiếng Việt/Windows
def read_image_windows(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# Hàm lưu ảnh hỗ trợ đường dẫn tiếng Việt/Windows
def save_image_windows(path, img):
    try:
        ext = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            encoded_img.tofile(path)
            return True
    except Exception:
        pass
    return False

def convert_obb_to_json_structure(filename, obb_results):
    """Chuyển đổi kết quả OBB của YOLO sang format JSON của bài toán và mở rộng OBB."""
    text_blocks = []
    
    # Giá trị mở rộng (expansion)
    expansion = 2 
    
    if obb_results is not None:
        # Lấy tọa độ OBB (x0, y0, x1, y1, x2, y2, x3, y3)
        boxes = obb_results.xyxyxyxy.cpu().numpy()
        
        for idx, box in enumerate(boxes):
            # box là mảng 4x2 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            
            # 1. Tính toán tâm (Center) của OBB
            # Tâm là trung bình cộng của tất cả 4 đỉnh
            center_x = np.mean(box[:, 0])
            center_y = np.mean(box[:, 1])
            
            expanded_points = []
            for point in box:
                x_old, y_old = point[0], point[1]
                
                # 2. Tính toán vector từ tâm đến đỉnh hiện tại (dx, dy)
                dx = x_old - center_x
                dy = y_old - center_y
                
                # 3. Tính toán độ dài của vector (khoảng cách từ tâm)
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 1e-6: # Tránh chia cho 0 nếu box là một điểm (điều kiện hiếm gặp)
                    # 4. Tính toán hệ số mở rộng
                    # Đỉnh mới sẽ nằm trên đường thẳng đi qua tâm và đỉnh cũ, 
                    # nhưng xa tâm hơn một khoảng 'expansion'
                    
                    # Hệ số tỷ lệ mới: (Khoảng cách cũ + expansion) / Khoảng cách cũ
                    scale_factor = (distance + expansion) / distance
                    
                    # 5. Tính toán tọa độ mới (mở rộng)
                    x_new = center_x + dx * scale_factor
                    y_new = center_y + dy * scale_factor
                else:
                    # Nếu box là 1 điểm, không thể mở rộng theo cách này, giữ nguyên (hoặc xử lý khác tùy ý)
                    x_new, y_old = x_old, y_old 

                expanded_points.append([x_new, y_new])

            # Chuyển list các điểm mở rộng thành cấu trúc JSON
            p = expanded_points
            polygon = {
                "x0": float(p[0][0]), "y0": float(p[0][1]),
                "x1": float(p[1][0]), "y1": float(p[1][1]),
                "x2": float(p[2][0]), "y2": float(p[2][1]),
                "x3": float(p[3][0]), "y3": float(p[3][1])
            }
            
            block = {"id": idx, "polygon": polygon}
            text_blocks.append(block)

    final_json = {
        "task1": {
            "input": {}, "name": "Chart Classification", "output": {"chart_type": "vertical bar"}
        },
        "task2": {
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "name": "Text Detection and Recognition",
            "output": {"text_blocks": text_blocks}
        }
    }
    return final_json

# --- HÀM VISUALIZE MỚI ---
def visualize_obb(img_path, json_data, output_dir):
    """Vẽ bounding box từ JSON lên ảnh gốc"""
    img = read_image_windows(img_path)
    if img is None: return

    blocks = json_data["task2"]["output"]["text_blocks"]
    for block in blocks:
        poly = block["polygon"]
        # Lấy 4 điểm của đa giác
        pts = np.array([
            [poly["x0"], poly["y0"]],
            [poly["x1"], poly["y1"]],
            [poly["x2"], poly["y2"]],
            [poly["x3"], poly["y3"]]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Vẽ đa giác màu xanh lá (Green), độ dày 2
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Lưu ảnh
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    save_image_windows(save_path, img)

# ==========================================
# 3. MAIN INFERENCE
# ==========================================
def main():
    print(f"--- Đang load model YOLO từ: {MODEL_PATH} ---")
    # Kiểm tra CUDA cho RTX 3050
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Inference device: {device} ({torch.cuda.get_device_name(0) if device == 0 else 'CPU'})")
    
    model = YOLO(MODEL_PATH)
    
    files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    total = len(files)
    print(f"Tìm thấy {total} ảnh. Bắt đầu xử lý...")
    
    for idx, filename in enumerate(files):
        img_path = os.path.join(INPUT_IMG_DIR, filename)
        json_name = os.path.splitext(filename)[0] + ".json"
        save_json_path = os.path.join(OUTPUT_JSON_DIR, json_name)
        
        print(f"[{idx+1}/{total}] Detecting: {filename} ...", end="\r")
        
        try:
            # 1. Run Inference (trên GPU, imgsz lớn cho text nhỏ)
            results = model.predict(img_path, save=False, conf=0.25, verbose=False, device=device, imgsz=1024)
            result = results[0]
            
            # 2. Tạo dữ liệu JSON
            if result.obb is not None:
                json_data = convert_obb_to_json_structure(filename, result.obb)
            else:
                json_data = convert_obb_to_json_structure(filename, None)
            
            # 3. Lưu file JSON
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
                
            # 4. VISUALIZE KẾT QUẢ (MỚI)
            if ENABLE_VISUALIZE:
                visualize_obb(img_path, json_data, OUTPUT_VIS_DIR)
                
        except Exception as e:
            print(f"\n[ERR] Lỗi khi xử lý ảnh {filename}: {e}")

    print(f"\n\n[DONE] Hoàn tất!")
    print(f"- File JSON lưu tại: {OUTPUT_JSON_DIR}")
    if ENABLE_VISUALIZE:
        print(f"- Ảnh Visualize lưu tại: {OUTPUT_VIS_DIR}")
