from pathlib import Path
import shutil

import kagglehub
import pandas as pd
from ucimlrepo import fetch_ucirepo

from Utilities.config import RAW_DATA_PATH, UCI_PHISHING_FILE


def clean_dir(path: Path) -> None:
    """Clean a directory by removing all files and subdirectories."""
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def extract_uci_data(uci_repo: int = None, data_path: Path = None) -> pd.DataFrame:
    """Extract and save the configured UCI phishing dataset."""
    from Utilities.config import UCI_PHISHING_REPO_ID

    if uci_repo is None:
        uci_repo = UCI_PHISHING_REPO_ID
    if data_path is None:
        data_path = RAW_DATA_PATH

    save_file = data_path / UCI_PHISHING_FILE
    uci_repo_url = fetch_ucirepo(id=uci_repo)
    df = uci_repo_url.data.original

    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_file)
    return df


def extract_kaggle_data(data_list=None, data_path: Path = None) -> list[Path]:
    """Download Kaggle datasets and copy their files into the raw data folder.

    kagglehub stores downloads in a shared cache. Copying avoids mutating that
    cache, which makes repeated runs more predictable.
    """
    from Utilities.config import KAGGLE_DATASETS

    if data_list is None:
        data_list = KAGGLE_DATASETS
    if data_path is None:
        data_path = RAW_DATA_PATH

    data_path.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []

    for dataset in data_list:
        kaggle_dataset_dir = Path(kagglehub.dataset_download(dataset))
        dataset_files = [item for item in kaggle_dataset_dir.iterdir() if item.is_file()]
        if not dataset_files:
            raise FileNotFoundError(f"No files found in Kaggle dataset: {dataset}")

        for item in dataset_files:
            dest_path = data_path / item.name
            shutil.copy2(item, dest_path)
            copied_files.append(dest_path)
            print(f"Copied {item.name} -> {dest_path}")

    return copied_files
