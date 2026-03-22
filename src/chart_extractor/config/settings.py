from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PathSettings:
    dataset_image: str = "./DatasetPredict/Input_model/Images/dataset/images"
    output_json_task2: str = "./DatasetPredict/Task2_output"
    output_json_task3: str = "./DatasetPredict/Task3_output"
    output_json_task4: str = "./DatasetPredict/Task4_output"
    output_excel_task4: str = "./DatasetPredict/Task4_output/result.csv"
    run_workspace: str = "./runs"


@dataclass(frozen=True)
class ModelSettings:
    model_path_layoutlmv3: str = "./weights/checkpoint-10000"
    model_path_yolo: str = "./weights/best.pt"
    model_path_yolo_text_det: str = "./weights/best_det.pt"


@dataclass(frozen=True)
class ProjectSettings:
    paths: PathSettings = PathSettings()
    models: ModelSettings = ModelSettings()
    task3_labels: tuple[str, ...] = (
        "CHART_TITLE",
        "LEGEND_TITLE",
        "LEGEND_LABEL",
        "AXIS_TITLE",
        "TICK_LABEL",
        "TICK_GROUPING",
        "MARK_LABEL",
        "VALUE_LABEL",
        "OTHER",
    )
    device: str = "cuda"

    def task2_config(self) -> dict:
        return {
            "input": self.paths.dataset_image,
            "output": self.paths.output_json_task2,
        }

    def task3_config(self) -> dict:
        return {
            "model_path": self.models.model_path_layoutlmv3,
            "data_dir_images": self.paths.dataset_image,
            "data_dir_json": self.paths.output_json_task2,
            "labels": list(self.task3_labels),
            "device": self.device,
            "output_dir": self.paths.output_json_task3,
        }

    def task4_config(self) -> dict:
        return {
            "input_images": self.paths.dataset_image,
            "input_json": self.paths.output_json_task3,
            "output_excel": self.paths.output_excel_task4,
            "output_json": self.paths.output_json_task4,
            "yolo_weight": self.models.model_path_yolo,
            "device": self.device,
        }

    def resolve(self, rel_or_abs: str) -> Path:
        path = Path(rel_or_abs)
        if path.is_absolute():
            return path
        return (get_project_root() / path).resolve()


def get_settings() -> ProjectSettings:
    return ProjectSettings()

