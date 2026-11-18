import pickle

import pandas as pd
import pytest

from Services.inference_service import (
    CONTENT_PLATFORM_DOMAINS,
    predict_with_bundle,
    _domain_impersonation_signals,
    _drop_non_feature_columns,
    _reference_domain_candidates,
    _reference_domain_index,
    inference_service,
    _load_model,
    _phishing_probability,
    _preprocess_url_for_inference,
)
from Services.model_training import RETIRED_FEATURES, prepare_data, render_model_comparison
from Utilities.Services.preprocess_data_utils import (
    URLCharacterProbabilityModel,
    _add_label_and_tld_prob_features,
    normalise_url,
)
from Utilities.Services.url_utils import split_registered_domain


class PredictProbaModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.25, 0.75]]


class LegitimateModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[0.74, 0.26]]


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

    assert feature_df.loc[0, "URLCharProb"] == char_model.score("bad.example/login")
    assert feature_df.loc[0, "URLSimilarityIndex"] == feature_df.loc[0, "URLCharProb"] * 100


def test_normalise_url_strips_scheme_www_and_trailing_slash():
    assert normalise_url("  HTTPS://www.Example.com/path/  ") == "Example.com/path"
    assert normalise_url("http://www2.example.com/") == "example.com"
    assert normalise_url("example.com/a?b=c") == "example.com/a?b=c"
    assert normalise_url("wwwexample.com") == "wwwexample.com"


def test_scheme_and_www_do_not_change_features():
    # The model must not see how a URL was typed, only what it points to.
    forms = ["google.com", "www.google.com", "https://www.google.com", "http://google.com/"]
    features = [_preprocess_url_for_inference(u).drop(columns=["URL"]) for u in forms]

    for other in features[1:]:
        pd.testing.assert_frame_equal(features[0], other)
    assert features[0].loc[0, "Domain"] == "google.com"


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

    # URLSimilarityIndex is URLCharProb x 100, so it is retired.
    assert list(X_train.columns) == ["feature", "URLCharProb"]
    assert len(X_test) == 2
    assert sorted(y_train.unique().tolist()) == [0, 1]
    assert char_model is not None


def test_prepare_data_fits_the_character_model_on_training_urls_only():
    df = pd.DataFrame(
        {
            "URL": [f"site{i}.example/{'q' * i}" for i in range(10)] + ["zzzz.example/ZZZZ"] * 2,
            "Label": ["good", "bad"] * 6,
            "feature": range(12),
            "label_binary": [0, 1] * 6,
        }
    )

    X_train, X_test, _, _, char_model = prepare_data(df, test_size=0.25, random_state=0)

    train_urls = df.loc[X_train.index, "URL"]
    assert char_model.char_probs == URLCharacterProbabilityModel.fit(train_urls).char_probs
    assert char_model.char_probs != URLCharacterProbabilityModel.fit(df["URL"]).char_probs
    # Both splits are scored with that one model.
    assert X_test["URLCharProb"].tolist() == [
        char_model.score(u) for u in df.loc[X_test.index, "URL"]
    ]


def test_prepare_data_keeps_retired_features_out_of_new_models():
    df = pd.DataFrame(
        {
            "URL": ["a.example", "b.example/login", "c.example", "d.example/verify"],
            "Label": ["good", "bad", "good", "bad"],
            "Pay": [0, 1, 0, 1],
            "HasBrandName": [0, 1, 0, 0],
            "PayNotBrandDomain": [0, 1, 0, 1],
            "label_binary": [0, 1, 0, 1],
        }
    )

    X_train, _, _, _, _ = prepare_data(df, test_size=0.5, random_state=42)

    assert not set(RETIRED_FEATURES) & set(X_train.columns)
    assert "PayNotBrandDomain" in X_train.columns


def test_inference_still_computes_retired_features_for_older_bundles():
    # The v9 champion lists these in feature_names; dropping them from the
    # feature code would silently feed it zeros.
    columns = set(_preprocess_url_for_inference("paypal.com").columns)

    assert set(RETIRED_FEATURES) <= columns


def _features(urls):
    """Inference-time features of several URLs, indexed by (normalised) URL."""
    rows = [_preprocess_url_for_inference(u) for u in urls]
    return pd.concat(rows, ignore_index=True).set_index("URL")


def test_brand_and_pay_features_ignore_the_brands_own_domain():
    f = _features(
        [
            "paypal.com",
            "login.paypal.com/signin",
            "wellsfargo.com",
            "paypal.com.account-verify.xyz/login.php",
            "paypal-secure.com",
            "paypal.com@evil.example/login",
        ]
    )
    cols = ["BrandOutsideDomain", "BrandInDomainNotExact", "PayNotBrandDomain"]

    assert f.loc["paypal.com", cols].tolist() == [0, 0, 0]
    assert f.loc["login.paypal.com/signin", cols].tolist() == [0, 0, 0]
    assert f.loc["wellsfargo.com", cols].tolist() == [0, 0, 0]
    assert f.loc["paypal.com.account-verify.xyz/login.php", cols].tolist() == [1, 0, 1]
    assert f.loc["paypal-secure.com", cols].tolist() == [0, 1, 1]
    assert f.loc["paypal.com@evil.example/login", cols].tolist() == [1, 0, 1]
    # The retired features still fire on the brand's own domain (v9 needs them as is).
    assert f.loc["paypal.com", ["Pay", "HasBrandName"]].tolist() == [1, 1]


def test_at_sign_features_tell_emails_and_user_info_from_handles():
    f = _features(
        [
            "substack.com/@alexkazos",
            "x.example/login.php?email=jo@example.com",
            "paypal.com@evil.example/login",
            "flickr.com/photos/37793774@N03/1",
        ]
    )
    cols = ["NumEmailsInURL", "HasUserInfo"]

    assert f.loc["substack.com/@alexkazos", cols].tolist() == [0, 0]
    assert f.loc["x.example/login.php?email=jo@example.com", cols].tolist() == [1, 0]
    assert f.loc["paypal.com@evil.example/login", "HasUserInfo"] == 1
    assert f.loc["flickr.com/photos/37793774@N03/1", cols].tolist() == [0, 0]


def test_split_registered_domain():
    assert split_registered_domain("paypal.com") == ("paypal", "  ")
    assert split_registered_domain("me@login.PayPal.com.evil.co.uk/pay?x=1") == (
        "evil",
        "me@ login.paypal.com /pay?x=1",
    )
    assert split_registered_domain("http://[::1") == ("", "http://[::1")


def test_prepare_data_rejects_a_dataset_with_flipped_labels():
    # A cached ml_ready_dataset.csv from before the label fix: label_binary = 1
    # for legitimate rows. Training on it produced a model right half the time.
    df = pd.DataFrame(
        {
            "URL": ["a.example", "b.example/login", "c.example", "d.example/verify"],
            "Label": ["good", "bad", "good", "bad"],
            "feature": [1, 2, 3, 4],
            "label_binary": [1, 0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="stale"):
        prepare_data(df, test_size=0.5, random_state=42)


def test_model_comparison_report_describes_the_run():
    report = render_model_comparison(
        {
            "XGBoost": {"Accuracy": 0.87, "Precision": 0.91, "Recall": 0.82, "F1-Score": 0.86,
                        "golden_set_accuracy": 0.9375, "bundle_size_mb": 8.6},
            "Random Forest": {"Accuracy": 0.86, "Precision": 0.90, "Recall": 0.81,
                              "F1-Score": 0.85, "golden_set_accuracy": 0.875},
        },
        versions={"XGBoost": "9", "Random Forest": "8"},
        challenger="XGBoost",
        dataset={"rows": 411_080, "features": 43, "train": 328_864, "test": 82_216,
                 "rows_from_uci": 149_677, "rows_from_kaggle": 261_982, "rows_in_both": 579},
        generated_at="2026-11-18T00:00:00+00:00",
    )

    assert "- Rows: 411,080" in report
    assert "in both: 579" in report
    assert "- Features: 43" in report
    assert "| XGBoost | v9 | 0.8700 | 0.9100 |" in report
    assert "Challenger: **XGBoost** (v9)" in report
    # Metrics a candidate did not log are shown as n/a, not as zeros.
    assert "n/a" in report.split("| Random Forest |")[1].splitlines()[0]


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


def test_domain_impersonation_signals_detect_leetspeak_brand():
    signals = _domain_impersonation_signals(
        "https://www.g00gle.uk/maps",
        reference_domains=["google.com"],
    )

    assert signals == ["domain-lookalike:g00gle->google.com"]


def test_domain_impersonation_signals_use_reference_domains_not_hardcoded_brands():
    signals = _domain_impersonation_signals(
        "https://www.yo0tube.com/maps",
        reference_domains=["youtube.com"],
    )

    assert signals == ["domain-typosquat:yo0tube->youtube.com"]


def test_domain_impersonation_signals_detect_two_edit_lookalike():
    signals = _domain_impersonation_signals(
        "https://www.youtoobe.com/maps",
        reference_domains=["youtube.com"],
    )

    assert signals == ["domain-typosquat:youtoobe->youtube.com"]


def test_domain_impersonation_signals_allow_exact_brand_label():
    signals = _domain_impersonation_signals(
        "https://www.google.com/maps",
        reference_domains=["google.com"],
    )

    assert signals == []


def test_domain_impersonation_signals_detect_one_letter_swap():
    signals = _domain_impersonation_signals("linkedln.com", reference_domains=["linkedin.com"])

    assert signals == ["domain-typosquat:linkedln->linkedin.com"]


def test_typosquat_targets_stop_at_the_target_limit():
    # tubestack.com sits at rank ~55k in top-1m.csv and is two edits from
    # "substack": with every reference as a target, substack.com was flagged.
    references = ["google.com", "tubestack.com"]

    assert _domain_impersonation_signals(
        "substack.com", reference_domains=references, target_limit=1
    ) == []
    assert _domain_impersonation_signals(
        "substack.com", reference_domains=references, target_limit=2
    ) == ["domain-typosquat:substack->tubestack.com"]


def test_exact_and_leetspeak_checks_use_every_reference_domain():
    references = ["google.com", "tubestack.com"]

    # A known site past the target limit is still recognised as known ...
    assert _domain_impersonation_signals(
        "tubestack.com", reference_domains=references, target_limit=1
    ) == []
    # ... and a leetspeak copy of it is still caught.
    assert _domain_impersonation_signals(
        "tub3stack.com", reference_domains=references, target_limit=1
    ) == ["domain-lookalike:tub3stack->tubestack.com"]


def test_reference_domain_candidates_only_use_near_length_buckets():
    # "yootubeyootube" shares every bigram with "yootube", so it passes the
    # bigram-similarity filter. It must still be excluded because its length
    # is more than two characters away from the input.
    index = _reference_domain_index(
        (
            "youtube.com",
            "yootubeyootube.com",
            "verylongreference.com",
            "tiny.io",
        )
    )

    candidates = _reference_domain_candidates("yootube", index)

    assert candidates == {"youtube": "youtube.com"}


def test_reference_domain_candidates_use_bigram_similarity():
    index = _reference_domain_index(
        (
            "youtube.com",
            "abcdefg.com",
            "zzzzzzz.com",
        )
    )

    candidates = _reference_domain_candidates("youtoobe", index)

    assert candidates == {"youtube": "youtube.com"}


def test_inference_overrides_legitimate_prediction_for_brand_lookalike(tmp_path):
    model_path = tmp_path / "Test_Model.pkl"
    artifact = {
        "model": LegitimateModel(),
        "feature_names": ["URLLength"],
        "reference_domains": ["google.com"],
        "label_mapping": {"legitimate": 0, "phishing": 1},
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    result = inference_service(
        "https://www.g00gle.uk/maps",
        model_name="Test_Model",
        models_dir=tmp_path,
    )

    assert result["label"] == 1
    assert result["probability"] == 0.85
    assert result["verdict"] == "[!] SUSPICIOUS / PHISHING"
    assert result["risk_signals"] == ["domain-lookalike:g00gle->google.com"]


class PhishingModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.35, 0.65]]


PHISHING_BUNDLE = {"feature_names": ["URLLength"], "reference_domains": ["google.com"]}


def test_content_platform_rule_overrides_a_phishing_verdict():
    result, details = predict_with_bundle(
        "https://alexkazos.substack.com", PhishingModel(), PHISHING_BUNDLE
    )

    assert result["label"] == 0
    assert result["probability"] == 0.15
    assert result["trust_signals"] == ["content-platform:substack.com"]
    assert details["model_label"] == 1
    assert details["platform_override"] is True


def test_content_platform_rule_leaves_tricks_and_hosting_to_the_model():
    for url in [
        "paypal.substack.com/login",  # brand in the subdomain
        "alexkazos.substack.com/?email=jo@example.com",  # victim's email
        "paypal.com@alexkazos.substack.com",  # user info before the host
        "substack.com.account-verify.xyz/login",  # platform name, other domain
        "substack-login.com",  # platform name inside another domain
        "alexkazos.github.io",  # hosting that serves arbitrary HTML
    ]:
        result, details = predict_with_bundle(url, PhishingModel(), PHISHING_BUNDLE)

        assert result["label"] == 1, url
        assert "trust_signals" not in result, url
        assert details["platform_override"] is False, url


def test_lookalike_signal_beats_the_content_platform_rule():
    # With tubestack.com as a typosquat target, "substack" raises a risk signal.
    bundle = {"feature_names": ["URLLength"], "reference_domains": ["tubestack.com"]}

    result, details = predict_with_bundle("alexkazos.substack.com", PhishingModel(), bundle)

    assert result["label"] == 1
    assert result["risk_signals"] == ["domain-typosquat:substack->tubestack.com"]
    assert "trust_signals" not in result


def test_content_platforms_exclude_hosting_that_serves_arbitrary_html():
    # These host phishing in the training data (web.app and firebaseapp.com
    # 100% of their bare subdomains, github.io too); a fake login form fits
    # on any of them.
    hosting = {"github.io", "blogspot.com", "web.app", "firebaseapp.com", "weeblysite.com"}

    assert not CONTENT_PLATFORM_DOMAINS & hosting


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
