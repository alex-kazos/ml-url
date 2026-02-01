from pathlib import Path


# Import utility functions
from Utilities.Services.extract_data_utils import (
    extract_uci_data,
    extract_kaggle_data
    )

# warnings == success
import warnings
warnings.filterwarnings("ignore")

def extract_data_service():
    ''' Service to extract data from various sources
    '''

    save_file = Path('..') / 'Data' / 'phishing_url_uci.pkl'
    phishing_uci_repo = 967

    # Extract UCI data
    extract_uci_data(uci_repo=phishing_uci_repo, save_file=save_file)


    data_list = ["simaanjali/tes-upload",
                 "simaanjali/phising-detection-dataset",
                 "nitsey/dataset-phising-website",
                 "eswarchandt/phishing-website-detector"
                 # phising site urls
                "taruntiwarihp/phishing-site-urls",
                 # top searches
                 "cheedcheed/top1m"
                     ]

    # Extract Kaggle data
    extract_kaggle_data(data_list=data_list)

