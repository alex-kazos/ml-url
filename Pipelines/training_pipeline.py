import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Services.extract_data import extract_data_service  # noqa: E402
from Services.merge_data import merge_data_service  # noqa: E402
from Services.preprocess_data import preprocess_data_service  # noqa: E402
from Services.feature_engineering import feature_engineering_service  # noqa: E402
from Services.model_training import model_training_service  # noqa: E402

# Import configuration
from Utilities.config import RAW_DATA_PATH  # noqa: E402

# Import utility functions
from Utilities.Services.extract_data_utils import clean_dir  # noqa: E402
from Utilities.Services.merge_data_utils import summarise_sources  # noqa: E402


def run_training_pipeline() -> dict:
    """Run extract -> preprocess -> merge -> features -> train, timing each stage."""
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

    # Data check: how many rows each source contributes (logged to MLflow)
    dataset_stats = summarise_sources(dataToMerge.urls_uci, dataToMerge.urls_kaggle, merged_df)
    print(f"Rows per source after merge: {dataset_stats}")
    for source in ("uci", "kaggle"):
        if dataset_stats[f"rows_{source}_raw"] and not dataset_stats[f"rows_from_{source}"]:
            print(f"WARNING: every {source.upper()} row was dropped during the merge.")

    # Step 4: Feature engineering
    ml_ready_df = feature_engineering_service(merged_df)
    end_feature_engineering_time = time.time()
    print(f"Feature engineering completed in {end_feature_engineering_time - end_merge_time:.2f} seconds.")

    # Step 5: Train, evaluate, log to MLflow and register the challenger
    results = model_training_service(
        df=ml_ready_df,
        test_size=0.2,
        random_state=42,
        dataset_stats=dataset_stats,
    )
    end_model_training_time = time.time()
    print(f"Model training completed in {end_model_training_time - end_feature_engineering_time:.2f} seconds.")

    print(f"Total pipeline execution time: {end_model_training_time - start_extract_time:.2f} seconds.")
    return results


if __name__ == "__main__":
    run_training_pipeline()
