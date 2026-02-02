from pathlib import Path

# warnings == success
import warnings
warnings.filterwarnings("ignore")

# Import utility functions
from Utilities.Services.extract_data_utils import (
    extract_uci_data,
    extract_kaggle_data,
    clean_dir
    )


RAW_DATA_PATH = Path('..') / 'Data' / 'raw'

def extract_data_service(data_path: Path):
    ''' Service to extract data from various sources
    '''

    # UCI repository ID for Phishing URL dataset
    phishing_uci_repo = 967

    # Extract UCI data
    # extract_uci_data(uci_repo=phishing_uci_repo, data_path=data_path)


    data_list = [
            # phising site urls -- addition to UCI
            "taruntiwarihp/phishing-site-urls",
             # top searches
             "cheedcheed/top1m"
    ]

    # Extract Kaggle data
    extract_kaggle_data(data_list=data_list)


## fetch data when this script is run directly
if __name__ == '__main__':

    # clean_dir(RAW_DATA_PATH)

    extract_data_service(data_path=RAW_DATA_PATH)