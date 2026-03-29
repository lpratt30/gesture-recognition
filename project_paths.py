from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
GESTURE_DATA_DIR = PROJECT_ROOT / "gesture_data"

DATASET_V0_DIR = GESTURE_DATA_DIR / "dataset_v0"
DATASET_V1_DIR = GESTURE_DATA_DIR / "dataset_v1"

RAW_EVENT_STREAMS_DIR = GESTURE_DATA_DIR / "raw_event_streams"
EVENT_CAPTURE_DIR = GESTURE_DATA_DIR / "event_capture"
RAW_CAPTURE_DIR = GESTURE_DATA_DIR / "raw_capture"
EVENT_VISUALIZATIONS_DIR = GESTURE_DATA_DIR / "event_visualizations"
MODEL_AUDITS_DIR = GESTURE_DATA_DIR / "model_audits"

RF_MODEL_PATH = PROJECT_ROOT / "peace_rf_model.joblib"
XGB_MODEL_PATH = PROJECT_ROOT / "peace_xgb_model.joblib"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
