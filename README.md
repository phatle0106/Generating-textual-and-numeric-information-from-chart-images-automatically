# Generating Textual and Numeric Information From Chart Images

This project extracts structured information from bar chart images with a 3-stage pipeline:

1. Task 2: text detection + OCR
2. Task 3: text role classification (title, tick label, legend label, etc.)
3. Task 4: chart value extraction and export (CSV/JSON + debug visualizations)

The repository was restructured into a package-first layout while preserving legacy entrypoints.

## Project Structure

```text
.
|- app.py                          # Root Streamlit entrypoint (kept stable)
|- pyproject.toml                  # Packaging config (src layout)
|- requirements.txt
|- scripts/
|  |- run_pipeline.py
|  |- run_task2.py
|  |- run_task3.py
|  `- run_task4.py
|- src/chart_extractor/
|  |- cli.py                       # Canonical CLI entrypoint
|  |- config/                      # Typed settings + legacy compatibility API
|  |- pipeline/                    # Task orchestration
|  |- tasks/                       # Task implementations
|  |- ui/                          # Streamlit UI modules
|  |- utils/                       # Shared helpers
|  `- schemas/                     # Dataclass schemas
`- SubmissionFinalCode/            # Legacy compatibility wrappers
```

## Legacy Compatibility

The following are still supported:

- `streamlit run app.py`
- `python SubmissionFinalCode/Main.py`
- `from SubmissionFinalCode import Task2, Task3, Task4` and calling `main()`

`SubmissionFinalCode/*` now acts as a compatibility layer over `src/chart_extractor/*`.

## Environment Setup

### 1. Python version

Use Python 3.10+ (3.10/3.11 recommended).

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Notes:
- `torch` is intentionally not pinned in `requirements.txt`; install the correct build for your CUDA/CPU setup.
- `paddlepaddle_gpu` should match your CUDA runtime.

## Model and Weights

Expected default locations:

- LayoutLMv3 checkpoint: `weights/checkpoint-10000/`
- YOLO text/bar weights: `weights/best.pt`, `weights/best_det.pt`

Default paths are configured in:
- `src/chart_extractor/config/settings.py`
- `src/chart_extractor/config/legacy_api.py` (legacy compatibility)

## Run Instructions

### Streamlit UI (primary)

```bash
streamlit run app.py
```

### Canonical CLI

```bash
python -m chart_extractor.cli run-pipeline
python -m chart_extractor.cli run-task2
python -m chart_extractor.cli run-task3
python -m chart_extractor.cli run-task4
```

Optional input override:

```bash
python -m chart_extractor.cli run-pipeline --input-dir ./temp_uploads
```

### Script shortcuts

```bash
python scripts/run_pipeline.py
python scripts/run_task2.py
python scripts/run_task3.py
python scripts/run_task4.py
```

## Input and Output

Default input image directory:

- `./DatasetPredict/Input_model/Images/dataset/images`

Default outputs:

- Task 2 JSON: `./DatasetPredict/Task2_output`
- Task 3 JSON: `./DatasetPredict/Task3_output`
- Task 4 JSON: `./DatasetPredict/Task4_output`
- Task 4 CSV: `./DatasetPredict/Task4_output/result.csv`
- Per-image CSV: `./DatasetPredict/Task4_output/individual_results/*.csv`
- Task 4 debug images: `./DatasetPredict/Task4_output/ResultImage/*.png`

Additional runtime workspace (new):

- `./runs/`

## Troubleshooting

1. CUDA / PyTorch / Paddle mismatch
- Reinstall torch with the wheel matching your CUDA version.
- Ensure `paddlepaddle_gpu` is compatible with the same CUDA toolkit.

2. Windows path or import errors
- Run from repository root.
- Ensure editable install is done: `pip install -e .`

3. Empty Task 4 outputs
- Verify Task 2 and Task 3 JSON files are generated first.
- Check that input images are valid bar charts and model weights exist.

4. Streamlit starts but pipeline fails
- Check that `weights/` files exist.
- Validate config paths in `settings.py`.

## Development Notes

- Keep dataset directories unchanged (out of scope for refactor).
- Main package code lives under `src/chart_extractor`.
- Use wrappers in `SubmissionFinalCode` only for backward compatibility.
- Suggested checks before commit:

```bash
python -m chart_extractor.cli run-task2 --input-dir ./temp_uploads
python -m chart_extractor.cli run-pipeline --input-dir ./temp_uploads
```

