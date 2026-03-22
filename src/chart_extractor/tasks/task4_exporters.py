from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd


def build_task_outputs(yValueDict: dict, axes_store: dict, legend_store: dict):
    task4_list, task5_list, task6_list = [], [], []
    for img_name, data in yValueDict.items():
        ax_entry = axes_store.get(img_name, {})
        task4_list.append(
            {"image": img_name, "task4": {"output": {"_plot_bb": ax_entry.get("_plot_bb", {}), "axes": ax_entry.get("axes", {})}}}
        )
        lg_entry = legend_store.get(img_name, [])
        task5_list.append({"image": img_name, "task5": {"output": {"legend_pairs": lg_entry}}})

        series = []
        for legend, legend_data in data.items():
            if isinstance(legend_data, dict):
                for x_label, val in legend_data.items():
                    series.append({"legend_label": legend, "x_label": x_label, "value": float(val)})
            else:
                series.append({"legend_label": legend, "x_label": "Value", "value": float(legend_data)})
        task6_list.append({"image": img_name, "task6": {"output": {"data series": series, "visual elements": []}}})
    return {"task4_outputs": task4_list, "task5_outputs": task5_list, "task6_outputs": task6_list}


def save_results(df: pd.DataFrame, yValueDict: dict, axes_store: dict, legend_store: dict, excel_path: str | Path, json_path: str | Path):
    excel_path = Path(excel_path)
    json_path = Path(json_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(excel_path, index=False)
    payload = build_task_outputs(yValueDict=yValueDict, axes_store=axes_store, legend_store=legend_store)
    with open(json_path / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    individual_dir = excel_path.parent / "individual_results"
    individual_dir.mkdir(parents=True, exist_ok=True)
    if "image" in df.columns:
        for image_name, sub in df.groupby("image"):
            stem = Path(str(image_name)).stem
            sub.to_csv(individual_dir / f"{stem}.csv", index=False)


def draw_debug_image(
    base_image_rgb,
    xaxis,
    yaxis,
    legend_patches,
    legend_text_boxes,
    bar_rects,
    x_label_rects,
    legend_for_bar,
    x_label_for_bar,
    save_path,
):
    img = base_image_rgb.copy()
    x1, y1, x2, y2 = map(int, xaxis)
    yx1, yy1, yx2, yy2 = map(int, yaxis)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(img, (yx1, yy1), (yx2, yy2), (255, 0, 0), 2)

    for (x, y, w, h) in legend_patches:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
    for (x, y, w, h) in bar_rects:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    for (x, y, w, h) in x_label_rects:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 0), 1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

