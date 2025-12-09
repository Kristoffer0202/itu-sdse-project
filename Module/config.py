from pathlib import Path
import datetime
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
ARTIFACTS_DIR = PROJ_ROOT / "notebooks/artifacts"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_DATA_FILE = RAW_DATA_DIR / "raw_data.csv"
DATE_LIMITS_FILE = ARTIFACTS_DIR / "date_limits.json"
DATA_FILTERED_FILE = INTERIM_DATA_DIR / "data_filtered.csv"
COLUMNS_DRIFT_FILE = ARTIFACTS_DIR / "columns_drift.json"
SCALE_FILE = ARTIFACTS_DIR / "scaler.pkl"
OUTLIER_SUMMARY_FILE = ARTIFACTS_DIR / "outlier_summary.json"
CAT_MISSING_IMPUTE_FILE = ARTIFACTS_DIR / "cat_missing_impute.csv"
TRAINING_DATA_FILE = PROCESSED_DATA_DIR / "training_data.csv"
TRAINING_GOLD_DATA_FILE = PROCESSED_DATA_DIR / "train_data_gold.csv"

MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"
DATA_VERSION = "00000"
CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
EXPERIMENT_NAME = CURRENT_DATE

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
