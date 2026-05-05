import pickle

import pandas as pd

from Services.inference_service import (
    _drop_non_feature_columns,
    _load_model,
    _phishing_probability,
    _preprocess_url_for_inference,
)
from Services.model_training import prepare_data
from Utilities.Services.preprocess_data_utils import (
    URLCharacterProbabilityModel,
    _add_label_and_tld_prob_features,
)


class PredictProbaModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.25, 0.75]]


def test_label_binary_uses_one_for_phishing():
    df = pd.DataFrame(
        {
            "URL": ["https://safe.example", "http://bad.example/login"],
            "Label": ["good", "bad"],
            "TLD": ["example", "example"],
        }
    )

    result = _add_label_and_tld_prob_features(df)

    assert result["label_binary"].tolist() == [0, 1]


def test_inference_reuses_training_url_character_model():
    training_urls = pd.Series(
        ["https://safe.example", "http://bad.example/login", "http://bad.example/pay"]
    )
    char_model = URLCharacterProbabilityModel.fit(training_urls)

    feature_df = _preprocess_url_for_inference(
        "http://bad.example/login",
        char_model=char_model,
    )

    assert feature_df.loc[0, "URLCharProb"] == char_model.score("http://bad.example/login")
    assert feature_df.loc[0, "URLSimilarityIndex"] == feature_df.loc[0, "URLCharProb"] * 100


def test_prepare_data_returns_feature_metadata():
    df = pd.DataFrame(
        {
            "URL": ["https://a.example", "http://b.example", "https://c.example", "http://d.example"],
            "Domain": ["a.example", "b.example", "c.example", "d.example"],
            "TLD": ["example", "example", "example", "example"],
            "Label": ["good", "bad", "good", "bad"],
            "TLDLegitimateProb": [0.5, 0.5, 0.5, 0.5],
            "feature": [1, 2, 3, 4],
            "label_binary": [0, 1, 0, 1],
        }
    )

    X_train, X_test, y_train, y_test, char_model = prepare_data(
        df,
        test_size=0.5,
        random_state=42,
    )

    assert list(X_train.columns) == ["feature", "URLCharProb", "URLSimilarityIndex"]
    assert len(X_test) == 2
    assert sorted(y_train.unique().tolist()) == [0, 1]
    assert char_model is not None


def test_load_model_accepts_metadata_bundle(tmp_path):
    model_path = tmp_path / "Bundled.pkl"
    artifact = {
        "model": PredictProbaModel(),
        "feature_names": ["feature_a"],
        "label_mapping": {"legitimate": 0, "phishing": 1},
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    model, metadata = _load_model("Bundled", tmp_path)

    assert isinstance(model, PredictProbaModel)
    assert metadata["feature_names"] == ["feature_a"]


def test_phishing_probability_uses_class_one():
    probability = _phishing_probability(PredictProbaModel(), pd.DataFrame({"feature": [1]}))

    assert probability == 0.75


def test_drop_non_feature_columns_keeps_engineered_features():
    df = pd.DataFrame(
        {
            "URL": ["https://example.com"],
            "Domain": ["example.com"],
            "TLD": ["com"],
            "Label": ["good"],
            "label_binary": [0],
            "feature": [1],
        }
    )

    result = _drop_non_feature_columns(df)

    assert list(result.columns) == ["feature"]
