import time
from Services.extract_data import extract_data_service
from Services.merge_data import merge_data_service
from Services.preprocess_data import preprocess_data_service
from Services.feature_engineering import feature_engineering_service
from Services.model_training import model_training_service

# Import configuration
from Utilities.config import RAW_DATA_PATH

# Import utility functions
from Utilities.Services.extract_data_utils import clean_dir

if __name__ == "__main__":
    # Step 0: Clean raw data directory before extraction
    clean_dir(RAW_DATA_PATH)

    start_extract_time = time.time()
    # Step 1: Extract data
    extract_data_service(RAW_DATA_PATH)
    end_extract_time = time.time()
    print(f"Data extraction completed in {end_extract_time - start_extract_time:.2f} seconds.")

    # Step 2: Preprocess data
    dataToMerge = preprocess_data_service()
    end_preprocess_time= time.time()
    print(f"Data preprocessing completed in {end_preprocess_time - end_extract_time:.2f} seconds.")

    # Step 3: Merge data
    merged_df = merge_data_service(dataToMerge)
    end_merge_time = time.time()
    print(f"Data merging completed in {end_merge_time - end_preprocess_time:.2f} seconds.")

    # Step 4: Feature engineering
    ml_ready_df = feature_engineering_service(merged_df)
    end_feature_engineering_time = time.time()
    print(f"Feature engineering completed in {end_feature_engineering_time - end_merge_time:.2f} seconds.")

    # Step 5: Train model (to be implemented)
    model_training_service(df=ml_ready_df,test_size=0.2, random_state=42)
    end_model_training_time = time.time()
    print(f"Model training completed in {end_model_training_time - end_feature_engineering_time:.2f} seconds.")

    print(f"Total pipeline execution time: {end_model_training_time - start_extract_time:.2f} seconds.")