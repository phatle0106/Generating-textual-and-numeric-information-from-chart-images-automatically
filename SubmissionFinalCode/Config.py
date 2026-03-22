from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chart_extractor.config.legacy_api import (  # noqa: F401,F403
    Dataset_Image,
    Output_Excel_Task_4,
    Output_Json_Task_2,
    Output_Json_Task_3,
    Output_Json_Task_4,
    Task2Config,
    Task3Config,
    Task4Config,
    model_path_layoutlmv3,
    model_path_yolo,
    returnTestTask2_Config,
    returnTestTask3_Config,
    returnTestTask4_Config,
)

