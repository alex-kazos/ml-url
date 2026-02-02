from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path('..') / 'Data' / 'raw'

def read_uci_phishing_data():
    ''' Read phishing URL data from pickle file
    '''

    df = pd.read_pickle(RAW_DATA_PATH / 'phishing_url_uci.pkl')

    return df

def read_kaggle_phishing_data():
    ''' Read Kaggle phishing data from CSV file
    '''

    file_path = RAW_DATA_PATH / 'phishing_site_urls.csv'

    df = pd.read_csv(file_path)

    return df



def read_kaggle_top_searches():
    ''' Read Kaggle top searches data from CSV file
    '''

    file_path = RAW_DATA_PATH / 'top-1m.csv'

    df = pd.read_csv(file_path)

    return df
