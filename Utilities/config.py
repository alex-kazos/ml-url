"""
Central configuration module for loading environment variables.
This module provides a single source of truth for all configuration values.
"""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load environment variables from .env file when python-dotenv is installed.
if load_dotenv is not None:
    load_dotenv()


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default

# Get project root directory (parent of Utilities folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / os.getenv('DATA_ROOT', 'Data')
RAW_DATA_PATH = PROJECT_ROOT / os.getenv('RAW_DATA_PATH', 'Data/raw')
PROCESSED_DATA_PATH = PROJECT_ROOT / os.getenv('PROCESSED_DATA_PATH', 'Data/processed')
MODELS_PATH = PROJECT_ROOT / os.getenv('MODELS_PATH', 'Models')

# MLflow configuration.
# Defaults to a project-local SQLite database: recent MLflow releases refuse the
# old ./mlruns file store, and the model registry needs a database backend anyway.
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_TRACKING_URI = _get_env('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
MLFLOW_REGISTRY_URI = _get_env('MLFLOW_REGISTRY_URI', MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT_NAME = _get_env('MLFLOW_EXPERIMENT_NAME', 'Phishing URL Detection')
MLFLOW_MODEL_NAME = _get_env('MLFLOW_MODEL_NAME', 'ml-url')

# Model promotion (champion / challenger) settings
PROMOTION_METRIC = _get_env('PROMOTION_METRIC', 'F1-Score')
PROMOTION_MIN_IMPROVEMENT = float(_get_env('PROMOTION_MIN_IMPROVEMENT', '0.002'))
PROMOTION_RECALL_TOLERANCE = float(_get_env('PROMOTION_RECALL_TOLERANCE', '0.005'))
# Precision floor for the challenger, and a health check for the champion: a
# model that calls almost everything phishing has high recall and must not be
# able to block a better one through the recall gate.
PROMOTION_MIN_PRECISION = float(_get_env('PROMOTION_MIN_PRECISION', '0.80'))
# Render's free tier has 512 MB of RAM, so keep the served bundle well below that.
MAX_BUNDLE_MB = float(_get_env('MAX_BUNDLE_MB', '50'))
EVAL_SAMPLE_SIZE = int(_get_env('EVAL_SAMPLE_SIZE', '2000'))

# Golden set: hand-picked URLs every served model must mostly get right
GOLDEN_SET_PATH = PROJECT_ROOT / os.getenv('GOLDEN_SET_PATH', 'Models/golden_set.csv')
GOLDEN_SET_MIN_ACCURACY = float(_get_env('GOLDEN_SET_MIN_ACCURACY', '0.75'))

# Holdout set: fixed, labelled URLs (url,label) that are removed from every
# training run and used to compare champion and challenger, so neither model
# is scored on URLs it trained on. Frozen from the v9 run's evaluation set
# (2,000 URLs from its test split). No '#' comments: some URLs contain '#'.
HOLDOUT_SET_PATH = PROJECT_ROOT / os.getenv('HOLDOUT_SET_PATH', 'Models/holdout_set.csv')

# Lookalike rule: typosquat candidates come only from the top N reference domains
# (by rank). Exact and leetspeak matches still use the whole reference list, so a
# known site is never flagged, but an unknown one is not matched against obscure
# domains nobody would impersonate.
LOOKALIKE_TARGET_LIMIT = int(_get_env('LOOKALIKE_TARGET_LIMIT', '5000'))

# Name of the exported champion bundle in Models/ (served when API_MODEL_NAME=champion)
CHAMPION_FILE_STEM = _get_env('CHAMPION_FILE_STEM', 'champion')

# File names
UCI_PHISHING_FILE = os.getenv('UCI_PHISHING_FILE', 'phishing_url_uci.pkl')
KAGGLE_PHISHING_FILE = os.getenv('KAGGLE_PHISHING_FILE', 'phishing_site_urls.csv')
KAGGLE_TOP_SEARCHES_FILE = os.getenv('KAGGLE_TOP_SEARCHES_FILE', 'top-1m.csv')

# Full file paths
UCI_PHISHING_FILE_PATH = RAW_DATA_PATH / UCI_PHISHING_FILE
KAGGLE_PHISHING_FILE_PATH = RAW_DATA_PATH / KAGGLE_PHISHING_FILE
KAGGLE_TOP_SEARCHES_FILE_PATH = RAW_DATA_PATH / KAGGLE_TOP_SEARCHES_FILE

# UCI Repository Configuration
UCI_PHISHING_REPO_ID = int(os.getenv('UCI_PHISHING_REPO_ID', '967'))

# Kaggle Datasets
KAGGLE_PHISHING_SITE_URLS = os.getenv('KAGGLE_PHISHING_SITE_URLS', 'taruntiwarihp/phishing-site-urls')
KAGGLE_TOP_1M = os.getenv('KAGGLE_TOP_1M', 'cheedcheed/top1m')

# Default Kaggle datasets list
KAGGLE_DATASETS = [
    KAGGLE_PHISHING_SITE_URLS,
    KAGGLE_TOP_1M
]
