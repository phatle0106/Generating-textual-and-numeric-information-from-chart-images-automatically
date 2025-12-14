import torch
import os
import sys
import shutil
import time
import traceback
from pathlib import Path
from typing import List
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import json

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = BASE_DIR / "SubmissionFinalCode"

# Add SubmissionFinalCode to sys.path so we can import Task modules
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.append(str(SUBMISSION_DIR))

# Ensure we are working from the base directory
os.chdir(BASE_DIR)

# Import Config after setting up paths
import Config  # noqa: E402

# --- GLOBAL CONFIGURATION ---
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
RESULT_CSV = (BASE_DIR / Config.Output_Excel_Task_4).resolve()
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Bar Chart Extraction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Modern Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* Light Theme - Black Text on White */
        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: 'Inter', sans-serif;
            color: #0F172A !important;
        }
        
        p, div, label, li, span, h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
        }

        /* Gradient Background for Main Area - Light */
        [data-testid="stAppViewContainer"] {
            background: #FFFFFF;
        }

        /* Sidebar Styling - Light Gray */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }

        /* Headers */
        h1 {
            background: linear-gradient(to right, #2563EB, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent !important;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        }
        
        /* Dataframes */
        .stDataFrame {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
        }
        
        /* Alerts */
        [data-testid="stAlert"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0;
        }
        
        input {
            color: #0F172A !important; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- SESSION MANAGEMENT ---
def ensure_session_state():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(time.time())
        # Clear temp uploads on new session start
        clear_session_data()

def clear_session_data():
    # 1. Clear Uploads
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Clear Results
    result_dir = (BASE_DIR / Config.Output_Excel_Task_4).resolve().parent
    if result_dir.exists():
         # Remove individual results
         individual_dir = result_dir / "individual_results"
         if individual_dir.exists():
             shutil.rmtree(individual_dir)
             individual_dir.mkdir(parents=True, exist_ok=True)
         
         # Remove main csv
         result_csv = result_dir / "result.csv"
         if result_csv.exists():
             try:
                 result_csv.unlink()
             except Exception:
                 pass
    
def ensure_folders():
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / Config.Dataset_Image).resolve().mkdir(parents=True, exist_ok=True)
    (BASE_DIR / Config.Output_Json_Task_2).resolve().mkdir(parents=True, exist_ok=True)
    (BASE_DIR / Config.Output_Json_Task_3).resolve().mkdir(parents=True, exist_ok=True)
    (BASE_DIR / Config.Output_Json_Task_4).resolve().mkdir(parents=True, exist_ok=True)

# --- PIPELINE LOGIC ---
# --- HELPER FUNCTIONS ---
def draw_ocr_boxes(image_path: Path, json_path: Path) -> Image.Image:
    """Draws bounding boxes and text from Task 2 JSON onto the image."""
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Load JSON data
        # Load JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Navigate to text_blocks (root -> task2 -> output -> text_blocks)
        text_blocks = []
        if isinstance(data, dict):
            if "task2" in data:
                text_blocks = data["task2"].get("output", {}).get("text_blocks", [])
            else:
                # Try finding list in root or other common keys
                text_blocks = data.get("text_blocks", [])
        elif isinstance(data, list):
             text_blocks = data

        # Draw each polygon
        for item in text_blocks:
            poly = item.get("polygon")
            text = item.get("text", "")
            if poly:
                # Polygon points: (x0,y0), (x1,y1), ...
                points = [
                    (poly["x0"], poly["y0"]),
                    (poly["x1"], poly["y1"]),
                    (poly["x2"], poly["y2"]),
                    (poly["x3"], poly["y3"])
                ]
                
                # Draw Box
                draw.polygon(points, outline="red", width=2)
                
                # Draw Text (optional, can be messy if small)
                # For now just box is cleaner, or maybe draw text above
                # x_min = min(p[0] for p in points)
                # y_min = min(p[1] for p in points)
                # draw.text((x_min, y_min - 10), text, fill="red")
                
        return image
    except Exception as e:
        print(f"Error drawing OCR boxes: {e}")
        return Image.open(image_path) # Fallback to original

@st.cache_resource
def load_pipeline_modules():
    """Load modules once to avoid reloading PyTorch models repeatedly."""


    try:
        import Task2 as t2
        import Task3 as t3
        import Task4 as t4
        
        # Initialize OCR once if needed
        # Initialize OCR once if needed
        # Store original init_model to avoid recursion on reload
        if not hasattr(t2, "original_init_model"):
            t2.original_init_model = t2.init_model

        if not hasattr(t2, "_app_cached_ocr"):
             t2._app_cached_ocr = None
             
        def init_model_cached():
             if t2._app_cached_ocr is None:
                 t2._app_cached_ocr = t2.original_init_model()
             return t2._app_cached_ocr
             
        t2.init_model = init_model_cached
        return t2, t3, t4
    except Exception as e:
        error_str = str(e)
        st.error(f"Failed to load pipeline modules: {e}")
        
        if "dll" in error_str.lower() or "procedure could not be found" in error_str.lower():
            st.warning("⚠️ This is a common PyTorch environment issue on Windows.")
            st.markdown("### Suggested Fix")
            st.markdown("Run the following command in your terminal (Active Environment: `dath`) to reinstall a stable PyTorch version:")
            st.code("pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", language="bash")
            st.markdown("After reinstalling, run `streamlit run app.py` again.")
            
        return None, None, None

def run_extraction_pipeline():
    ensure_folders()
    Task2, Task3, Task4 = load_pipeline_modules()
    
    if not Task2:
        st.error("Pipeline modules failed to load.")
        return

    # Configure pipeline to use temp directory
    Config.Dataset_Image = str(TEMP_UPLOAD_DIR)
    
    # RELOAD configs to ensure fresh state if modified
    Task2.Task2_Config = Config.returnTestTask2_Config()
    Task3.TEST_CONFIG = Config.returnTestTask3_Config()
    Task4.TASK4_CONFIG = Config.returnTestTask4_Config()
    
    # Explicitly set input/output paths
    Task2.Task2_Config["input"] = str(TEMP_UPLOAD_DIR)
    t2_output = (BASE_DIR / Config.Output_Json_Task_2).resolve()
    Task2.Task2_Config["output"] = str(t2_output)
    
    Task3.TEST_CONFIG["data_dir_images"] = str(TEMP_UPLOAD_DIR)
    Task3.TEST_CONFIG["data_dir_json"] = str(t2_output)
    t3_output = (BASE_DIR / Config.Output_Json_Task_3).resolve()
    Task3.TEST_CONFIG["output_dir"] = str(t3_output)
    
    Task4.TASK4_CONFIG["input_images"] = str(TEMP_UPLOAD_DIR)
    Task4.TASK4_CONFIG["input_json"] = str(t3_output)
    t4_output_csv = (BASE_DIR / Config.Output_Excel_Task_4).resolve()
    Task4.TASK4_CONFIG["output_excel"] = str(t4_output_csv)

    # Debug info
    with st.expander("Debug: Configuration Paths"):
        st.write(f"**Input Images:** `{Task2.Task2_Config['input']}`")
        st.write(f"**Task 2 Output:** `{Task2.Task2_Config['output']}`")
        st.write(f"**Task 3 Input:** `{Task3.TEST_CONFIG['data_dir_json']}`")
        st.write(f"**Task 3 Output:** `{Task3.TEST_CONFIG['output_dir']}`")
        st.write(f"**Task 4 CSV:** `{Task4.TASK4_CONFIG['output_excel']}`")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # TASK 2
        status_text.text("Running Task 2: Text Detection & Recognition...")
        Task2.main()
        
        # Verify Task 2
        t2_files = list(t2_output.glob("*.json"))
        if not t2_files:
            st.error(f"Task 2 finished but produced no JSON files in `{t2_output}`. Check if images are valid.")
            return
        st.toast(f"Task 2 complete: {len(t2_files)} files generated.")
        progress_bar.progress(33)

        # TASK 3
        status_text.text("Running Task 3: Text Role Classification...")
        Task3.main()
        
        # Verify Task 3
        t3_files = list(t3_output.glob("*.json"))
        if not t3_files:
            st.error(f"Task 3 finished but produced no JSON files in `{t3_output}`. Pipeline stopped.")
            return
        st.toast(f"Task 3 complete: {len(t3_files)} files generated.")
        progress_bar.progress(66)

        # TASK 4
        status_text.text("Running Task 4: Chart Value Extraction...")
        Task4.main()
        
        if not t4_output_csv.exists():
             st.error("Task 4 finished but CSV result file was not created.")
             return
             
        progress_bar.progress(100)
        status_text.success("Pipeline completed successfully!")
        # Keep success message and progress bar visible
        
    except Exception as e:
        st.error(f"Error during pipeline execution: {e}")
        # print(traceback.format_exc())

# --- UI COMPONENTS ---
def render_sidebar():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/404/404621.png", width=64)
        st.title("Settings")
        st.markdown("---")
        
        st.write("### Instructions")
        st.info(
            """
            1. **Upload** your chart images.
            2. Click **Run Extraction**.
            3. View and **Download** results.
            """
        )
        
        if st.button("Clear Session & Uploads", type="secondary"):
            clear_session_data()
            st.rerun()

        st.caption(f"v1.0.0 | Session: {st.session_state.get('session_id', 'N/A')[:8]}...")

def render_main_content():
    st.title("Bar Chart Information Extraction")
    st.markdown("### Automated data extraction from bar chart images")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Images (PNG, JPG, JPEG)", 
        type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        save_uploaded_files(uploaded_files)
        
        # Display uploaded images in a grid
        st.subheader(f"Uploaded Images ({len(uploaded_files)})")
        cols = st.columns(4)
        for idx, file_path in enumerate(list_uploaded_images()):
            col = cols[idx % 4]
            with col:
                 image = Image.open(file_path)
                 st.image(image, caption=file_path.name, use_container_width=True)

        # Run Button
        st.markdown("---")
        if st.button("Run Extraction", type="primary", use_container_width=True):
            with st.spinner("Processing images... This may take a moment."):
                run_extraction_pipeline()
                # Debug tips
                st.info("Run finished. Check above for any errors or below for results.")

    # Results Section
    if RESULT_CSV.exists():
        st.markdown("---")
        st.subheader("Extraction Results")
        
        # Check for individual results folder
        individual_dir = RESULT_CSV.parent / "individual_results"
        
        # 1. Fallback if no individual files found: Show old combined view/message
        if not individual_dir.exists() or not any(individual_dir.glob("*.csv")):
             try:
                df = pd.read_csv(RESULT_CSV)
                if not df.empty:
                    st.warning("Found combined results only (older run). Please re-run extraction for separated results.")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No data found.")
             except Exception as e:
                 st.error(f"Error loading results: {e}")
                 if "No columns to parse" in str(e) or "EmptyDataError" in str(e):
                     st.warning("No data extracted from images. Logic in Task 4 yielded no results.")
                 else:
                     st.error(f"Error loading results: {e}")
                 
        else:
            # 2. Modern View: Per-Image Results
            csv_files = sorted(list(individual_dir.glob("*.csv")))
            
            # Map input images to CSVs by stem
            # Note: We rely on file stem matching. 
            # Img: "chart1.png" -> stem "chart1"
            # CSV: "chart1.csv"
            
            uploaded_imgs = list_uploaded_images()
            img_map = {p.stem: p for p in uploaded_imgs}
            
            for csv_path in csv_files:
                stem = csv_path.stem
                
                with st.expander(f"Result: {stem}", expanded=True):
                    col_img, col_data = st.columns([1, 2])
                    
                    # Show Image with OCR Boxes
                    with col_img:
                        if stem in img_map:
                            original_img_path = img_map[stem]
                            
                            # Locate the Task 2 JSON file
                            # Task 2 output is in Config.Output_Json_Task_2
                            # Filename usually matches stem + extension or just stem + .json
                            # Based on Task2.py, it likely uses original filename + .json
                            # Let's try to find it.
                            task2_json_dir = (BASE_DIR / Config.Output_Json_Task_2).resolve()
                            # Try exact match 
                            # Possible naming: "image.png.json" or "image.json"
                            # Task2 main loop: path.name -> save path / path.name (but with .json appended or replaced?)
                            # Let's check Task2 save logic if needed, but usually it appends .json or replaces suffix.
                            # Usually simple is: task2_json_dir / (original_img_path.name + ".json")
                            
                            json_candidate = task2_json_dir / (original_img_path.name + ".json")
                            if not json_candidate.exists():
                                 # Try replace suffix
                                 json_candidate = task2_json_dir / (original_img_path.stem + ".json")
                            
                            if json_candidate.exists():
                                annotated_image = draw_ocr_boxes(original_img_path, json_candidate)
                                st.image(annotated_image, caption=f"{original_img_path.name} (OCR Visualization)", use_container_width=True)
                            else:
                                # Fallback to original if JSON missing
                                st.image(str(original_img_path), caption=str(original_img_path.name), use_container_width=True)
                        else:
                            st.warning(f"Image source not found in uploads: {stem}")
                            
                    # Show Data
                    with col_data:
                        try:
                            sub_df = pd.read_csv(csv_path)
                            st.dataframe(sub_df, use_container_width=True, height=300)
                            
                            csv_data = sub_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"Download {stem}.csv",
                                data=csv_data,
                                file_name=f"{stem}.csv",
                                mime="text/csv",
                                key=f"dl_{stem}"
                            )
                        except Exception as e:
                             if "No columns to parse" in str(e):
                                 st.info("Empty value file (no chart data detected).")
                             else:
                                 st.error(f"Error loading CSV: {e}")
                            
            # Option to download ALL as ZIP could be added here later
    
def save_uploaded_files(uploaded_files):
    ensure_folders()
    # Check if we need to clear previous uploads if they aren't in the current upload list
    # Actually, simplistic approach: Just save them. `clear_session` handles full wipe.
    
    saved_count = 0
    for file in uploaded_files:
        dest_path = TEMP_UPLOAD_DIR / file.name
        # Only write if size/timestamp differs usually, but simplest is overwrite
        with open(dest_path, "wb") as f:
            f.write(file.getbuffer())
        saved_count += 1
    return saved_count

def list_uploaded_images() -> List[Path]:
    if not TEMP_UPLOAD_DIR.exists():
        return []
    return sorted(
        [p for p in TEMP_UPLOAD_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS],
        key=lambda p: p.name
    )

def main():
    ensure_session_state()
    inject_custom_css()
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()
