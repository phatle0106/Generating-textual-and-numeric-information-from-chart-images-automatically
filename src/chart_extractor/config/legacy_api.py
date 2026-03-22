"""Compatibility config API for legacy task modules."""

from __future__ import annotations

from chart_extractor.config.settings import get_settings

_settings = get_settings()

# Legacy variable names kept for backward compatibility.
Dataset_Image = _settings.paths.dataset_image
Output_Json_Task_2 = _settings.paths.output_json_task2
Output_Json_Task_3 = _settings.paths.output_json_task3
Output_Json_Task_4 = _settings.paths.output_json_task4
Output_Excel_Task_4 = _settings.paths.output_excel_task4
model_path_layoutlmv3 = _settings.models.model_path_layoutlmv3
model_path_yolo = _settings.models.model_path_yolo

Task2Config = {
    "input": Dataset_Image,
    "output": Output_Json_Task_2,
}

Task3Config = {
    "model_path": model_path_layoutlmv3,
    "data_dir_images": Dataset_Image,
    "data_dir_json": Output_Json_Task_2,
    "labels": list(_settings.task3_labels),
    "device": _settings.device,
    "output_dir": Output_Json_Task_3,
}

Task4Config = {
    "input_images": Dataset_Image,
    "input_json": Output_Json_Task_3,
    "output_excel": Output_Excel_Task_4,
    "output_json": Output_Json_Task_4,
    "yolo_weight": model_path_yolo,
    "device": _settings.device,
}


def returnTestTask2_Config():
    return Task2Config


def returnTestTask3_Config():
    return Task3Config


def returnTestTask4_Config():
    return Task4Config

