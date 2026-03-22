from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from chart_extractor.config import legacy_api as Config
from chart_extractor.config.settings import get_project_root, get_settings
from chart_extractor.ui.pipeline_runner import run_extraction_pipeline
from chart_extractor.ui.session import (
    ALLOWED_EXTENSIONS,
    clear_session_data,
    ensure_session_state,
    list_uploaded_images,
    save_uploaded_files,
)
from chart_extractor.ui.styles import inject_custom_css
from chart_extractor.utils.visualization import draw_ocr_boxes

BASE_DIR = get_project_root()
RESULT_CSV = (BASE_DIR / Config.Output_Excel_Task_4).resolve()


def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/404/404621.png", width=64)
        st.title("Settings")
        st.markdown("---")
        st.write("### Instructions")
        st.info("1. Upload images.\n2. Click Run Extraction.\n3. View and download results.")
        if st.button("Clear Session & Uploads", type="secondary"):
            clear_session_data()
            st.rerun()
        st.caption(f"v2.0.0 | Session: {st.session_state.get('session_id', 'N/A')[:8]}...")


def _render_result_fallback() -> None:
    try:
        df = pd.read_csv(RESULT_CSV)
        if not df.empty:
            st.warning("Found combined results only. Re-run extraction for separated per-image results.")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data found.")
    except Exception as e:
        st.error(f"Error loading results: {e}")


def _render_result_per_image() -> None:
    individual_dir = RESULT_CSV.parent / "individual_results"
    csv_files = sorted(list(individual_dir.glob("*.csv")))
    uploaded_imgs = list_uploaded_images()
    img_map = {p.stem: p for p in uploaded_imgs}

    for csv_path in csv_files:
        stem = csv_path.stem
        with st.expander(f"Result: {stem}", expanded=True):
            col_img, col_data = st.columns([1, 2])
            with col_img:
                if stem in img_map:
                    original_img_path = img_map[stem]
                    task2_json_dir = (BASE_DIR / Config.Output_Json_Task_2).resolve()
                    json_candidate = task2_json_dir / f"{original_img_path.name}.json"
                    if not json_candidate.exists():
                        json_candidate = task2_json_dir / f"{original_img_path.stem}.json"
                    if json_candidate.exists():
                        st.image(
                            draw_ocr_boxes(original_img_path, json_candidate),
                            caption=f"{original_img_path.name} (OCR Visualization)",
                            use_container_width=True,
                        )
                    else:
                        st.image(str(original_img_path), caption=original_img_path.name, use_container_width=True)
                else:
                    st.warning(f"Image source not found in uploads: {stem}")

            with col_data:
                try:
                    sub_df = pd.read_csv(csv_path)
                    st.dataframe(sub_df, use_container_width=True, height=300)
                    csv_data = sub_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download {stem}.csv",
                        data=csv_data,
                        file_name=f"{stem}.csv",
                        mime="text/csv",
                        key=f"dl_{stem}",
                    )
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")


def render_main_content() -> None:
    st.title("Bar Chart Information Extraction")
    st.markdown("### Automated data extraction from bar chart images")

    uploaded_files = st.file_uploader(
        "Upload Images (PNG, JPG, JPEG)",
        type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploaded_files:
        save_uploaded_files(uploaded_files)
        st.subheader(f"Uploaded Images ({len(uploaded_files)})")
        cols = st.columns(4)
        for idx, file_path in enumerate(list_uploaded_images()):
            with cols[idx % 4]:
                st.image(Image.open(file_path), caption=file_path.name, use_container_width=True)

        st.markdown("---")
        if st.button("Run Extraction", type="primary", use_container_width=True):
            with st.spinner("Processing images..."):
                run_extraction_pipeline()
                st.info("Run finished. Check logs and results below.")

    if RESULT_CSV.exists():
        st.markdown("---")
        st.subheader("Extraction Results")
        individual_dir = RESULT_CSV.parent / "individual_results"
        if not individual_dir.exists() or not any(individual_dir.glob("*.csv")):
            _render_result_fallback()
        else:
            _render_result_per_image()


def main() -> None:
    settings = get_settings()
    st.set_page_config(
        page_title="Bar Chart Extraction",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ = settings  # explicit keep for future dynamic settings controls
    ensure_session_state()
    inject_custom_css()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()

