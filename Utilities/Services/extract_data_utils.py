from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import shutil

import pandas as pd
from ucimlrepo import fetch_ucirepo
import kagglehub


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
def extract_uci_data(uci_repo: int = 967,data_path: Path = None):
    ''' Extract & save data from UCI repository

    Parameters
    ----------
    uci_repo : int
    data_path : Path or None

    Returns
    -------
    df : pd.DataFrame or None

    '''

    save_file = data_path / 'phishing_url_uci.pkl'
    # fetch dataset
    uci_repo_url = fetch_ucirepo(id=967)
    # parse to dataframe
    df = uci_repo_url.data.original

    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_file)


# Extract Data from Kaggle repository
def extract_kaggle_data(data_list= type[list]):
    ''' Extract & save data from Kaggle repository
    '''

    # Define data folder
    data_folder = Path.cwd().parent / "Data" / "raw"

    with ThreadPoolExecutor(max_workers=2) as executor:
        for dataset in data_list:
            # Download latest version
            path = kagglehub.dataset_download(dataset)
            # Move data to project data folder
            # kaggle_dataset_dir = Path(path)
            kaggle_dataset_dir = Path(path).parent.parent.parent.parent

            for item in kaggle_dataset_dir.iterdir():
                print(item)
                dest_path = data_folder / item.name
                print(dest_path)
                shutil.move(str(item), str(dest_path))