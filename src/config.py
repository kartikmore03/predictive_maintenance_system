import os

# Project root directory (one level above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Data and model directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Raw dataset path
RAW_DATA_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

# Saved artifacts
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_logreg.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")

# Target column used for training
TARGET_COL = "machine_failure"

# Feature definitions
CATEGORICAL_COLS = [
    "product_id",
    "type",
]

NUMERIC_COLS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
