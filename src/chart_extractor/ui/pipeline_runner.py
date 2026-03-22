from __future__ import annotations

from pathlib import Path

import streamlit as st

from chart_extractor.config import legacy_api as Config
from chart_extractor.config.settings import get_project_root, get_settings
from chart_extractor.pipeline.orchestrator import run_pipeline
from chart_extractor.ui.session import TEMP_UPLOAD_DIR, ensure_folders

BASE_DIR = get_project_root()


def run_extraction_pipeline() -> None:
    ensure_folders()
    settings = get_settings()
    result_csv = settings.resolve(Config.Output_Excel_Task_4)
    t2_output = settings.resolve(Config.Output_Json_Task_2)
    t3_output = settings.resolve(Config.Output_Json_Task_3)

    with st.expander("Debug: Configuration Paths"):
        st.write(f"**Input Images:** `{TEMP_UPLOAD_DIR}`")
        st.write(f"**Task 2 Output:** `{t2_output}`")
        st.write(f"**Task 3 Output:** `{t3_output}`")
        st.write(f"**Task 4 CSV:** `{result_csv}`")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Running Task 2 -> Task 3 -> Task 4...")
        run_pipeline(input_dir=TEMP_UPLOAD_DIR)
        progress_bar.progress(100)
        if not result_csv.exists():
            st.error("Pipeline finished but result.csv was not created.")
            return
        status_text.success("Pipeline completed successfully.")
    except Exception as e:
        st.error(f"Error during pipeline execution: {e}")

