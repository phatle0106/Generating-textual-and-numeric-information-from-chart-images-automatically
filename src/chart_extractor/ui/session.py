from __future__ import annotations

import shutil
import time
from pathlib import Path

import streamlit as st

from chart_extractor.config import legacy_api as Config
from chart_extractor.config.settings import get_project_root, get_settings
from chart_extractor.utils.io import ensure_dir, list_images

BASE_DIR = get_project_root()
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def ensure_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(time.time())
        clear_session_data()


def clear_session_data() -> None:
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR)
    ensure_dir(TEMP_UPLOAD_DIR)

    result_dir = (BASE_DIR / Config.Output_Excel_Task_4).resolve().parent
    if result_dir.exists():
        individual_dir = result_dir / "individual_results"
        if individual_dir.exists():
            shutil.rmtree(individual_dir)
            ensure_dir(individual_dir)
        result_csv = result_dir / "result.csv"
        if result_csv.exists():
            try:
                result_csv.unlink()
            except Exception:
                pass


def ensure_folders() -> None:
    settings = get_settings()
    ensure_dir(TEMP_UPLOAD_DIR)
    ensure_dir(settings.resolve(Config.Dataset_Image))
    ensure_dir(settings.resolve(Config.Output_Json_Task_2))
    ensure_dir(settings.resolve(Config.Output_Json_Task_3))
    ensure_dir(settings.resolve(Config.Output_Json_Task_4))
    ensure_dir(settings.resolve(settings.paths.run_workspace))


def save_uploaded_files(uploaded_files) -> int:
    ensure_folders()
    saved_count = 0
    for file in uploaded_files:
        dest_path = TEMP_UPLOAD_DIR / file.name
        with open(dest_path, "wb") as f:
            f.write(file.getbuffer())
        saved_count += 1
    return saved_count


def list_uploaded_images() -> list[Path]:
    return list_images(TEMP_UPLOAD_DIR, ALLOWED_EXTENSIONS)

