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

# Get project root directory (parent of Utilities folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / os.getenv('DATA_ROOT', 'Data')
RAW_DATA_PATH = PROJECT_ROOT / os.getenv('RAW_DATA_PATH', 'Data/raw')
PROCESSED_DATA_PATH = PROJECT_ROOT / os.getenv('PROCESSED_DATA_PATH', 'Data/processed')
MODELS_PATH = PROJECT_ROOT / os.getenv('MODELS_PATH', 'Models')

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
