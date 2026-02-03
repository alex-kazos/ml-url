from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import shutil

import pandas as pd
from ucimlrepo import fetch_ucirepo
import kagglehub

# Import configuration
from Utilities.config import (
    RAW_DATA_PATH,
    UCI_PHISHING_FILE
)


def clean_dir(path: Path):
    ''' Clean directory by removing all files and subdirectories

    Parameters
    ----------
    path : Path

    Returns
    -------
    None

    '''
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


# Extract Data from UCI repository
def extract_uci_data(uci_repo: int = None, data_path: Path = None):
    ''' Extract & save data from UCI repository

    Parameters
    ----------
    uci_repo : int, optional
        UCI repository ID. If None, uses UCI_PHISHING_REPO_ID from config.
    data_path : Path, optional
        Path to save data. If None, uses RAW_DATA_PATH from config.

    Returns
    -------
    df : pd.DataFrame or None

    '''
    from Utilities.config import UCI_PHISHING_REPO_ID
    
    if uci_repo is None:
        uci_repo = UCI_PHISHING_REPO_ID
    
    if data_path is None:
        data_path = RAW_DATA_PATH

    save_file = data_path / UCI_PHISHING_FILE
    # fetch dataset
    uci_repo_url = fetch_ucirepo(id=uci_repo)
    # parse to dataframe
    df = uci_repo_url.data.original

    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_file)
    
    return df


# Extract Data from Kaggle repository
def extract_kaggle_data(data_list=None, data_path: Path = None):
    ''' Extract & save data from Kaggle repository
    
    Parameters
    ----------
    data_list : list, optional
        List of Kaggle dataset names. If None, uses KAGGLE_DATASETS from config.
    data_path : Path, optional
        Path to save data. If None, uses RAW_DATA_PATH from config.
    '''
    from Utilities.config import KAGGLE_DATASETS
    
    if data_list is None:
        data_list = KAGGLE_DATASETS
    
    if data_path is None:
        data_path = RAW_DATA_PATH

    with ThreadPoolExecutor(max_workers=2) as executor:
        for dataset in data_list:
            # Download latest version
            path = kagglehub.dataset_download(dataset)
            # Move data to project data folder
            kaggle_dataset_dir = Path(path)
            # kaggle_dataset_dir = Path(path).parent.parent.parent.parent # this move the whole folder to Data/raw

            for item in kaggle_dataset_dir.iterdir():
                print(item)
                dest_path = data_path / item.name
                print(dest_path)
                shutil.move(str(item), str(dest_path))