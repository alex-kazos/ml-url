# warnings == success
import warnings
warnings.filterwarnings("ignore")

# Import configuration
from Utilities.config import (
    RAW_DATA_PATH,
    UCI_PHISHING_REPO_ID,
    KAGGLE_DATASETS
)

# Import utility functions
from Utilities.Services.extract_data_utils import (
    extract_uci_data,
    extract_kaggle_data,
    clean_dir
    )


def extract_data_service(data_path=None):
    ''' Service to extract data from various sources
    
    Parameters
    ----------
    data_path : Path, optional
        Path to raw data directory. If None, uses RAW_DATA_PATH from config.
    '''

    if data_path is None:
        data_path = RAW_DATA_PATH

    # Extract UCI data
    # extract_uci_data(uci_repo=UCI_PHISHING_REPO_ID, data_path=data_path)

    # Extract Kaggle data
    extract_kaggle_data(data_list=KAGGLE_DATASETS)


## fetch data when this script is run directly
if __name__ == '__main__':

    clean_dir(RAW_DATA_PATH)

    extract_data_service(data_path=RAW_DATA_PATH)