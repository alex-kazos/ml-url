# 🔍 Phishing URL Detection

A machine learning pipeline that classifies URLs as **phishing** (malicious) or **safe** (legitimate), with MLflow experiment tracking and a web UI for real-time predictions.

---

## 📁 Project Structure

```
ml-url/
│
├── Classes/                        # Data classes
│   └── DataToMerge.py              # DTO for passing datasets between pipeline steps
│
├── Data/                           # Raw & processed datasets (gitignored)
│
├── Models/                         # Saved trained models & evaluation plots
│
├── Notebook/                       # Jupyter notebooks for exploration & analysis
│
├── Services/                       # Core pipeline services (orchestration layer)
│   ├── extract_data.py             # ✅ Download datasets from UCI & Kaggle
│   ├── preprocess_data.py          # ✅ Load & preprocess raw datasets
│   ├── merge_data.py               # ✅ Merge, deduplicate & balance datasets
│   ├── feature_engineering.py      # ✅ Engineer features & produce ML-ready dataset
│   └── model_training.py           # ✅ Train, evaluate & save baseline models
│
├── Utilities/                      # Helper functions & configuration
│   ├── config.py                   # Central path & constant definitions
│   └── Services/                   # Utility functions per pipeline step
│       ├── extract_data_utils.py
│       ├── preprocess_data_utils.py
│       ├── merge_data_utils.py
│       ├── feature_engineering_utils.py
│       └── model_training_utils.py
│
├── app/                            # 🔜 Web UI (Streamlit / FastAPI)
│   └── app.py                      #    User inputs a URL → returns safe / suspicious
│
├── mlruns/                         # 🔜 MLflow tracking directory (gitignored)
│
├── requirements.txt
└── README.md
```

---

## ✅ What Has Been Implemented

### Services — Pipeline Steps

Each service can run independently or be chained together as a full pipeline.

| #   | Service                 | File                     | Description                                                                                                                                                                                                                 |
| --- | ----------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Extract Data**        | `extract_data.py`        | Downloads phishing URL datasets from UCI repository and Kaggle.                                                                                                                                                             |
| 2   | **Preprocess Data**     | `preprocess_data.py`     | Reads raw data, applies URL-based feature extraction on the Kaggle set, and returns a `DataToMerge` object.                                                                                                                 |
| 3   | **Merge Data**          | `merge_data.py`          | Merges UCI & Kaggle datasets, removes duplicates/NaNs, and balances the class distribution.                                                                                                                                 |
| 4   | **Feature Engineering** | `feature_engineering.py` | Adds engineered features (URL entropy, char ratios, suspicious patterns, etc.), handles `inf`/`NaN`, and saves the ML-ready dataset.                                                                                        |
| 5   | **Model Training**      | `model_training.py`      | Trains three baseline models (Logistic Regression, Random Forest, XGBoost), evaluates each with accuracy/precision/recall/F1/ROC-AUC, generates confusion matrix & ROC plots, and saves the trained models as `.pkl` files. |

### Pipeline Flow

```
extract_data → preprocess_data → merge_data → feature_engineering → model_training
```

---

## 🔜 Next Steps

- [ ] **MLflow Integration** — Log parameters, metrics, and model artifacts to MLflow for experiment tracking and model comparison.
- [ ] **Web UI** — Build a small Streamlit or FastAPI app so users can input a URL and get a safe/suspicious prediction in real time.
- [ ] **Hyperparameter Tuning** — Use GridSearchCV / RandomizedSearchCV to optimise model performance.
- [ ] **Model Registry** — Register the best model in MLflow for easy deployment.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
tldextract
mlflow          # (upcoming)
streamlit       # (upcoming)
```

---

## 📊 Data Sources

- [UCI — PhiUSIIL Phishing URL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- [Kaggle — Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)