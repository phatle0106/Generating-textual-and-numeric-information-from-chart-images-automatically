from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def draw_ocr_boxes(image_path: Path, json_path: Path) -> Image.Image:
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_blocks = []
        if isinstance(data, dict):
            if "task2" in data:
                text_blocks = data["task2"].get("output", {}).get("text_blocks", [])
            else:
                text_blocks = data.get("text_blocks", [])
        elif isinstance(data, list):
            text_blocks = data

        for item in text_blocks:
            poly = item.get("polygon")
            if not poly:
                continue
            points = [
                (poly["x0"], poly["y0"]),
                (poly["x1"], poly["y1"]),
                (poly["x2"], poly["y2"]),
                (poly["x3"], poly["y3"]),
            ]
            draw.polygon(points, outline="red", width=2)
        return image
    except Exception:
        return Image.open(image_path)

