import pandas as pd

# Import configuration
from Utilities.config import (
    UCI_PHISHING_FILE_PATH,
    KAGGLE_PHISHING_FILE_PATH,
    KAGGLE_TOP_SEARCHES_FILE_PATH
)


def read_uci_phishing_data():
    ''' Read phishing URL data from pickle file
    
    Returns
    -------
    df : pd.DataFrame
        Phishing URL data from UCI repository
    '''

    df = pd.read_pickle(UCI_PHISHING_FILE_PATH)

    return df


def read_kaggle_phishing_data():
    ''' Read Kaggle phishing data from CSV file
    
    Returns
    -------
    df : pd.DataFrame
        Phishing site URLs data from Kaggle
    '''

    df = pd.read_csv(KAGGLE_PHISHING_FILE_PATH)

    return df


def read_kaggle_top_searches():
    ''' Read Kaggle top searches data from CSV file
    
    Returns
    -------
    df : pd.DataFrame
        Top 1M websites data from Kaggle
    '''

    df = pd.read_csv(KAGGLE_TOP_SEARCHES_FILE_PATH)

    return df
