import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = BASE_DIR / "SubmissionFinalCode"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.append(str(SUBMISSION_DIR))
os.chdir(BASE_DIR)

import Config  # noqa: E402

TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
IMAGE_DIR = (BASE_DIR / Config.Dataset_Image).resolve()
OUTPUT_JSON_TASK2 = (BASE_DIR / Config.Output_Json_Task_2).resolve()
OUTPUT_JSON_TASK3 = (BASE_DIR / Config.Output_Json_Task_3).resolve()
OUTPUT_JSON_TASK4 = (BASE_DIR / Config.Output_Json_Task_4).resolve()
RESULT_CSV = (BASE_DIR / Config.Output_Excel_Task_4).resolve()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def ensure_folders() -> None:
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_TASK2.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_TASK3.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_TASK4.mkdir(parents=True, exist_ok=True)


def clear_temp() -> None:
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR, ignore_errors=True)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploads(files: List[gr.File]) -> List[Path]:
    ensure_folders()
    saved_paths: List[Path] = []
    for file in files or []:
        ext = Path(file.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        dest = TEMP_UPLOAD_DIR / Path(file.name).name
        shutil.copy(file.name, dest)
        saved_paths.append(dest)
    return saved_paths


def load_results(filter_images: List[str]) -> pd.DataFrame:
    if not RESULT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULT_CSV)
    if filter_images:
        df = df[df["image"].isin(filter_images)]
    df["value"] = df["value"].fillna("")
    return df


def load_pipeline_modules():
    import Task2 as t2  # noqa: WPS433
    import Task3 as t3  # noqa: WPS433
    import Task4 as t4  # noqa: WPS433

    if not hasattr(t2, "_app_cached_ocr"):
        t2._app_cached_ocr = None

    def init_model_cached():
        if t2._app_cached_ocr is None:
            t2._app_cached_ocr = t2.init_model()
        return t2._app_cached_ocr

    t2.init_model = init_model_cached  # type: ignore[assignment]
    return t2, t3, t4


def prepare_configs():
    Config.Dataset_Image = str(TEMP_UPLOAD_DIR)
    Config.Task2Config["input"] = str(TEMP_UPLOAD_DIR)
    Config.Task3Config["data_dir_images"] = str(TEMP_UPLOAD_DIR)
    Config.Task4Config["input_images"] = str(TEMP_UPLOAD_DIR)


def run_pipeline(files: List[gr.File]) -> Tuple[pd.DataFrame, List[str], str]:
    if not files:
        return pd.DataFrame(), [], "Upload at least one image."

    try:
        clear_temp()
        saved_paths = save_uploads(files)
        if not saved_paths:
            return pd.DataFrame(), [], "No supported images were uploaded."

        prepare_configs()
        Task2, Task3, Task4 = load_pipeline_modules()

        Task2.Task2_Config = Config.returnTestTask2_Config()
        Task3.TEST_CONFIG = Config.returnTestTask3_Config()
        Task4.TASK4_CONFIG = Config.returnTestTask4_Config()

        Task2.Task2_Config["input"] = str(TEMP_UPLOAD_DIR)
        Task3.TEST_CONFIG["data_dir_images"] = str(TEMP_UPLOAD_DIR)
        Task4.TASK4_CONFIG["input_images"] = str(TEMP_UPLOAD_DIR)

        Task2.main()
        Task3.main()
        Task4.main()

        image_names = [p.name for p in saved_paths]
        df = load_results(image_names)
        status = f"Completed. Processed {len(saved_paths)} image(s)."
        return df, [str(p) for p in saved_paths], status
    except Exception as exc:  # pragma: no cover - run-time safety
        return pd.DataFrame(), [], f"Something went wrong: {exc}"


def handle_upload(files: List[gr.File]) -> Tuple[List[str], str]:
    clear_temp()
    saved_paths = save_uploads(files)
    if not saved_paths:
        return [], "No images saved. Check file types."
    return [str(p) for p in saved_paths], f"Ready to run ({len(saved_paths)} image(s))."


def build_interface() -> gr.Blocks:
    css = """
    .gradio-container {background: radial-gradient(circle at 10% 20%, #eef3ff 0, #f8fbff 30%, #fefefe 70%);}
    .panel {border-radius: 14px; border: 1px solid #e2e8f0; background: rgba(255,255,255,0.8); box-shadow: 0 18px 38px rgba(15,23,42,0.08);}
    """
    with gr.Blocks(title="Bar Chart Extraction", css=css) as demo:
        gr.Markdown(
            """
            <div style="padding: 16px 18px; border-radius: 18px;
                        background: linear-gradient(120deg, #0f172a, #1d4ed8);
                        color: white; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);">
                <h2 style="margin:0;">Bar Chart Information Extraction</h2>
                <p style="margin:4px 0 0 0; opacity:0.9;">Upload chart images, run the OCR + parsing pipeline, and view extracted values.</p>
            </div>
            """,
            elem_classes=["panel"],
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                upload = gr.File(
                    label="Upload bar/column charts",
                    file_types=["image"],
                    file_count="multiple",
                )
                run_btn = gr.Button("Run extraction", variant="primary")
                clear_btn = gr.Button("Clear uploads")
                status = gr.Textbox(label="Status", lines=3, interactive=False)
            with gr.Column(scale=1, elem_classes=["panel"]):
                gallery = gr.Gallery(
                    label="Ready images",
                    show_label=True,
                    height=320,
                    columns=3,
                    allow_preview=True,
                )

        with gr.Row():
            table = gr.Dataframe(
                headers=["image", "legend", "x_label", "value"],
                datatype=["str", "str", "str", "str"],
                row_count=(0, "dynamic"),
                col_count=(4, "fixed"),
                interactive=False,
                label="Extracted values",
                elem_classes=["panel"],
            )

        upload.upload(handle_upload, inputs=upload, outputs=[gallery, status])
        clear_btn.click(fn=lambda: (clear_temp(), [], "Uploads cleared")[1:], outputs=[gallery, status])
        run_btn.click(run_pipeline, inputs=upload, outputs=[table, gallery, status])

    return demo


if __name__ == "__main__":
    ensure_folders()
    app = build_interface()
    app.queue().launch(server_name="0.0.0.0", share=True, show_api=False)
