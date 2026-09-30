from datetime import datetime
from pathlib import Path
import shutil

import kagglehub
import pandas as pd
from ucimlrepo import fetch_ucirepo
from ucimlrepo.fetch import DatasetNotFoundError

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
    """Download and save the UCI phishing dataset, or reuse the saved copy.

    The UCI dataset is a fixed snapshot, so when the UCI site cannot be
    reached (it has served an expired SSL certificate before) the copy saved
    by an earlier run is just as good. Only fail when there is no usable copy.
    """
    from Utilities.config import UCI_PHISHING_REPO_ID

    if uci_repo is None:
        uci_repo = UCI_PHISHING_REPO_ID
    if data_path is None:
        data_path = RAW_DATA_PATH

    save_file = data_path / UCI_PHISHING_FILE
    try:
        df = fetch_ucirepo(id=uci_repo).data.original
    except (ConnectionError, DatasetNotFoundError) as exc:
        return _load_saved_uci_copy(save_file, uci_repo, exc)

    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_file)
    return df


def _load_saved_uci_copy(save_file: Path, uci_repo: int, download_error: Exception) -> pd.DataFrame:
    if not save_file.exists():
        raise RuntimeError(
            f"Could not download UCI dataset {uci_repo} and there is no saved copy at "
            f"{save_file}. Retry later, or copy phishing_url_uci.pkl there from a previous run."
        ) from download_error

    try:
        df = pd.read_pickle(save_file)
    except Exception as read_error:
        raise RuntimeError(
            f"Could not download UCI dataset {uci_repo}, and the saved copy at {save_file} "
            f"cannot be read ({read_error!r}). It may have been written by a different "
            "pandas version."
        ) from download_error

    saved_on = datetime.fromtimestamp(save_file.stat().st_mtime).date()
    print(
        f"WARNING: could not download UCI dataset {uci_repo} ({download_error}). "
        f"Using the copy saved on {saved_on}: {save_file}"
    )
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
