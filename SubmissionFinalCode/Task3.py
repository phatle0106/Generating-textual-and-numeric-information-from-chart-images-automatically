import torch
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification, LayoutLMv3Model
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import pytesseract
import Config


# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
TEST_CONFIG = Config.returnTestTask3_Config()

# ==========================================
# 2. ĐỊNH NGHĨA MODEL
# ==========================================
class LayoutLMv3ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x); 
        x = self.dense(x); 
        x = torch.tanh(x); 
        x = self.dropout(x); 
        x = self.out_proj(x)
        return x

class CustomLayoutLMv3(LayoutLMv3ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = LayoutLMv3ClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids=None, bbox=None, attention_mask=None, labels=None, pixel_values=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        outputs = self.layoutlmv3(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        
        if input_ids is not None: 
            sequence_output = outputs[0][:, :input_ids.size(1), :]
        else: 
            sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits

# ==========================================
# 3. HÀM LOAD DATA
# ==========================================
def normalize_bbox(box, w, h):
    return [
        int(1000 * box[0] / w), 
        int(1000 * box[1] / h), 
        int(1000 * box[2] / w), 
        int(1000 * box[3] / h)
    ]

def load_icpr_bar_charts_flat(data_dir_images, data_dir_json, target_labels):
    dataset_dicts = []
    label2id = {label: i for i, label in enumerate(target_labels)}
    
    ann_dir = data_dir_json
    img_dir = data_dir_images

    if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
        print("Error: Kiểm tra lại cấu trúc thư mục annotations_JSON và images")
        return []

    files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    print(f"Tìm thấy {len(files)} file dữ liệu.")

    for file in tqdm(files, desc="Loading Data"):
        try:
            json_path = os.path.join(ann_dir, file)
            with open(json_path, "r", encoding="utf8") as f: data = json.load(f)
            
            # task3 = data.get("task3") # Phiên bản cũ
            task_data = data.get("task2")

            if not task_data: 
                continue
            
            chart_type = "vertical bar" # Mặc định
            try:
                chart_type = task_data["input"]["task1_output"]["chart_type"]
            except: pass

            img_name_base = os.path.splitext(file)[0]
            png_path = os.path.join(img_dir, img_name_base + ".png")
            jpg_path = os.path.join(img_dir, img_name_base + ".jpg")
            
            if os.path.exists(png_path): 
                image_path = png_path
            elif os.path.exists(jpg_path): 
                image_path = jpg_path
            else: 
                continue
            
            with Image.open(image_path) as img: w, h = img.size
            
            # text_blocks = task3["input"]["task2_output"]["text_blocks"]   # Phiên bản cũ
            text_blocks = task_data["output"]["text_blocks"]
            
            words, bboxes, labels = [], [], []
            
            original_blocks_cleaned = [] 

            for block in text_blocks:
                text = block.get("text", "").strip()
                if not text: 
                    continue
                
                role_str = "OTHER" 
                
                poly = block.get("polygon")
                if isinstance(poly, dict):
                    poly_list = [poly["x0"], poly["y0"], poly["x1"], poly["y1"], 
                                 poly["x2"], poly["y2"], poly["x3"], poly["y3"]]
                elif isinstance(poly, list):
                    poly_list = poly
                else: 
                    continue

                x_c, y_c = poly_list[0::2], poly_list[1::2]
                box = [min(x_c), min(y_c), max(x_c), max(y_c)]
                
                words.append(text)
                bboxes.append(normalize_bbox(box, w, h))
                labels.append(label2id.get(role_str, 0))
                
                original_blocks_cleaned.append(block)
            
            if words:
                dataset_dicts.append({
                    "id": file, 
                    "image_path": image_path, 
                    "words": words, 
                    "bboxes": bboxes, 
                    "labels": labels,
                    "original_blocks": original_blocks_cleaned,
                    "chart_type": chart_type
                })
        except: 
            continue
            
    return dataset_dicts

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print(f"Loading Model from {TEST_CONFIG['model_path']}...")
    try: 
        model = CustomLayoutLMv3.from_pretrained(TEST_CONFIG["model_path"])
    except: 
        model = LayoutLMv3ForTokenClassification.from_pretrained(TEST_CONFIG["model_path"])
    
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    model.to(TEST_CONFIG["device"])
    model.eval()

    # 2. Tạo thư mục output
    if not os.path.exists(TEST_CONFIG["output_dir"]):
        os.makedirs(TEST_CONFIG["output_dir"])
        print(f"Đã tạo thư mục output: {TEST_CONFIG['output_dir']}")

    # 3. Load Data
    test_data = load_icpr_bar_charts_flat(TEST_CONFIG["data_dir_images"], TEST_CONFIG["data_dir_json"], TEST_CONFIG["labels"])
    
    if len(test_data) == 0: 
        return

    id2label = {i: label for i, label in enumerate(TEST_CONFIG["labels"])}
    
    print(f"Bắt đầu xử lý và lưu file JSON riêng lẻ vào {TEST_CONFIG['output_dir']}...")
    
    for item in tqdm(test_data, desc="Processing"):
        image = Image.open(item["image_path"]).convert("RGB")
        clamped_bboxes = [[max(0, min(1000, b)) for b in box] for box in item["bboxes"]]
        
        # Processor
        encoding = processor(
            image, item["words"], boxes=clamped_bboxes, word_labels=item["labels"],
            truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )

        inputs = {k: v.to(TEST_CONFIG["device"]) for k, v in encoding.items()}
        
        with torch.no_grad():
            logits = model(**inputs)
            if logits.shape[1] > inputs["labels"].shape[1]: 
                logits = logits[:, :inputs["labels"].shape[1], :]
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        word_ids = encoding.word_ids()
        predicted_roles_map = {}
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in predicted_roles_map:
                    pred_id = preds[idx]
                    predicted_roles_map[word_idx] = id2label[pred_id]
        
        text_roles_output = []
        original_blocks = item["original_blocks"]
        
        for i, block in enumerate(original_blocks):
            role = predicted_roles_map.get(i, "other").lower()
            
            text_roles_output.append({
                "id": block["id"],
                "role": role
            })

        final_json = {
            "task3": {
                "input": {
                    "task1_output": {
                        "chart_type": item["chart_type"]
                    },
                    "task2_output": {
                        "text_blocks": original_blocks
                    }
                },
                "output": {
                    "text_roles": text_roles_output
                }
            }
        }

        output_path = os.path.join(TEST_CONFIG["output_dir"], item["id"])
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)

    print("Hoàn tất! Kiểm tra thư mục output_results.")