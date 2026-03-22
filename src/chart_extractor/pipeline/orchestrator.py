from __future__ import annotations

from pathlib import Path

from chart_extractor.config import legacy_api as Config
from chart_extractor.config.settings import get_project_root, get_settings


def _sync_task_configs(
    input_dir: Path | None = None,
    output_task2: Path | None = None,
    output_task3: Path | None = None,
    output_csv_task4: Path | None = None,
) -> dict[str, Path]:
    settings = get_settings()
    root = get_project_root()

    in_dir = (input_dir or settings.resolve(Config.Dataset_Image)).resolve()
    t2_out = (output_task2 or settings.resolve(Config.Output_Json_Task_2)).resolve()
    t3_out = (output_task3 or settings.resolve(Config.Output_Json_Task_3)).resolve()
    t4_csv = (output_csv_task4 or settings.resolve(Config.Output_Excel_Task_4)).resolve()

    from chart_extractor.tasks import task2_ocr as Task2
    from chart_extractor.tasks import task3_roles as Task3
    from chart_extractor.tasks import task4_values as Task4

    Task2.Task2_Config = Config.returnTestTask2_Config()
    Task3.TEST_CONFIG = Config.returnTestTask3_Config()
    Task4.TASK4_CONFIG = Config.returnTestTask4_Config()

    Task2.Task2_Config["input"] = str(in_dir)
    Task2.Task2_Config["output"] = str(t2_out)
    Task3.TEST_CONFIG["data_dir_images"] = str(in_dir)
    Task3.TEST_CONFIG["data_dir_json"] = str(t2_out)
    Task3.TEST_CONFIG["output_dir"] = str(t3_out)
    Task4.TASK4_CONFIG["input_images"] = str(in_dir)
    Task4.TASK4_CONFIG["input_json"] = str(t3_out)
    Task4.TASK4_CONFIG["output_excel"] = str(t4_csv)

    # Keep legacy config globals in sync for wrappers and downstream modules.
    Config.Dataset_Image = str(in_dir.relative_to(root)) if str(in_dir).startswith(str(root)) else str(in_dir)
    Config.Output_Json_Task_2 = str(t2_out.relative_to(root)) if str(t2_out).startswith(str(root)) else str(t2_out)
    Config.Output_Json_Task_3 = str(t3_out.relative_to(root)) if str(t3_out).startswith(str(root)) else str(t3_out)
    Config.Output_Excel_Task_4 = str(t4_csv.relative_to(root)) if str(t4_csv).startswith(str(root)) else str(t4_csv)

    return {"input_dir": in_dir, "task2_dir": t2_out, "task3_dir": t3_out, "task4_csv": t4_csv}


def run_pipeline(input_dir: Path | None = None) -> dict[str, Path]:
    from chart_extractor.tasks import task2_ocr as Task2
    from chart_extractor.tasks import task3_roles as Task3
    from chart_extractor.tasks import task4_values as Task4

    paths = _sync_task_configs(input_dir=input_dir)
    paths["task2_dir"].mkdir(parents=True, exist_ok=True)
    paths["task3_dir"].mkdir(parents=True, exist_ok=True)
    paths["task4_csv"].parent.mkdir(parents=True, exist_ok=True)

    Task2.main()
    Task3.main()
    Task4.main()
    return paths


def run_task2(input_dir: Path | None = None) -> dict[str, Path]:
    from chart_extractor.tasks import task2_ocr as Task2

    paths = _sync_task_configs(input_dir=input_dir)
    paths["task2_dir"].mkdir(parents=True, exist_ok=True)
    Task2.main()
    return paths


def run_task3(input_dir: Path | None = None) -> dict[str, Path]:
    from chart_extractor.tasks import task3_roles as Task3

    paths = _sync_task_configs(input_dir=input_dir)
    paths["task3_dir"].mkdir(parents=True, exist_ok=True)
    Task3.main()
    return paths


def run_task4(input_dir: Path | None = None) -> dict[str, Path]:
    from chart_extractor.tasks import task4_values as Task4

    paths = _sync_task_configs(input_dir=input_dir)
    paths["task4_csv"].parent.mkdir(parents=True, exist_ok=True)
    Task4.main()
    return paths
