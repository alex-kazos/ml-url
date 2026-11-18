"""Contracts for the MLOps layer: promotion gates, data-source checks, golden set."""
import pandas as pd

from Classes.DataToMerge import DataToMerge
from Services.merge_data import merge_data_service
from Services.model_registry import (
    SERVING_REQUIREMENTS,
    PromotionCriteria,
    load_golden_set,
    promotion_decision,
)
from Utilities.config import PROJECT_ROOT
from Utilities.Services.merge_data_utils import summarise_sources
from Utilities.Services.preprocess_data_utils import preprocess_phishing

CRITERIA = PromotionCriteria(
    metric="F1-Score",
    min_improvement=0.002,
    recall_tolerance=0.005,
    max_bundle_mb=50,
    golden_min_accuracy=0.75,
    min_precision=0.80,
)


def _metrics(f1=0.90, recall=0.90, precision=0.90, golden=1.0, size=2.0):
    return {
        "F1-Score": f1,
        "Recall": recall,
        "Precision": precision,
        "golden_set_accuracy": golden,
        "bundle_size_mb": size,
    }


def test_first_challenger_becomes_champion_when_gates_pass():
    promote, reasons = promotion_decision(None, _metrics(), CRITERIA)
    assert promote
    assert "no current champion" in reasons[0]


def test_golden_set_failure_blocks_promotion():
    promote, reasons = promotion_decision(None, _metrics(golden=0.5), CRITERIA)
    assert not promote
    assert any("golden set" in r for r in reasons)


def test_oversized_bundle_blocks_promotion():
    promote, reasons = promotion_decision(_metrics(), _metrics(f1=0.95, size=180), CRITERIA)
    assert not promote
    assert any("MB" in r for r in reasons)


def test_challenger_needs_a_real_improvement():
    promote, reasons = promotion_decision(_metrics(f1=0.900), _metrics(f1=0.901), CRITERIA)
    assert not promote
    assert any("gain" in r for r in reasons)


def test_recall_regression_blocks_promotion_even_with_better_f1():
    promote, reasons = promotion_decision(
        _metrics(f1=0.90, recall=0.92), _metrics(f1=0.93, recall=0.90), CRITERIA
    )
    assert not promote
    assert any("recall" in r for r in reasons)


def test_clear_winner_is_promoted():
    promote, _ = promotion_decision(
        _metrics(f1=0.89, recall=0.90), _metrics(f1=0.91, recall=0.905), CRITERIA
    )
    assert promote


def test_precision_floor_blocks_promotion():
    promote, reasons = promotion_decision(None, _metrics(precision=0.70), CRITERIA)
    assert not promote
    assert any("precision" in r for r in reasons)


def test_unhealthy_champion_cannot_block_a_healthy_challenger():
    # The September run: v6 had been trained on flipped labels and called
    # almost everything phishing, so its recall was near 1 and the recall gate
    # kept it as champion over v9. Precision and golden-set numbers are the
    # ones measured then through the API path; F1 and recall are illustrative.
    v6 = _metrics(f1=0.737, recall=0.998, precision=0.584, golden=0.50)
    v9 = _metrics(f1=0.834, recall=0.840, precision=0.829, golden=0.9375, size=8.6)

    promote, reasons = promotion_decision(v6, v9, CRITERIA)

    assert promote
    assert "the champion fails its own gates" in reasons[0]


def test_unhealthy_champion_stays_when_the_challenger_also_fails():
    promote, reasons = promotion_decision(
        _metrics(precision=0.58), _metrics(precision=0.75), CRITERIA
    )
    assert not promote
    assert any("precision" in r for r in reasons)


def test_serving_pins_match_requirements_api():
    # Models/champion.pkl only unpickles reliably with the versions that wrote it,
    # so the Render install and the MLflow model must ask for the same ones.
    requirements = (PROJECT_ROOT / "requirements-api.txt").read_text().splitlines()
    pinned = {line.strip() for line in requirements if "==" in line}

    assert set(SERVING_REQUIREMENTS) == pinned


def test_summarise_sources_counts_rows_per_source():
    uci = pd.DataFrame({"URL": ["https://a.example", "https://b.example"]})
    kaggle = pd.DataFrame({"URL": ["c.example/x", "d.example/y", "e.example/z"]})
    merged = pd.DataFrame({"URL": ["https://a.example", "c.example/x"]})

    stats = summarise_sources(uci, kaggle, merged)

    assert stats == {
        "rows_uci_raw": 2,
        "rows_kaggle_raw": 3,
        "rows_after_merge": 2,
        "rows_from_uci": 1,
        "rows_from_kaggle": 1,
        "rows_in_both": 0,
    }


def test_summarise_sources_counts_urls_present_in_both_sources():
    uci = pd.DataFrame({"URL": ["https://www.a.example", "https://b.example"]})
    kaggle = pd.DataFrame({"URL": ["a.example", "c.example/x"]})
    merged = pd.DataFrame({"URL": ["a.example", "b.example", "c.example/x"]})

    stats = summarise_sources(uci, kaggle, merged)

    assert stats["rows_in_both"] == 1
    assert stats["rows_from_uci"] + stats["rows_from_kaggle"] == (
        stats["rows_after_merge"] + stats["rows_in_both"]
    )


def test_merge_saves_the_final_dataset_under_pkl_path(tmp_path):
    uci = pd.DataFrame({"URL": ["https://www.uci-legit.example"], "label": [1]})
    kaggle = preprocess_phishing(
        pd.DataFrame({"URL": ["kaggle-bad.example/verify"], "Label": ["bad"]})
    )

    merge_data_service(DataToMerge(urls_uci=uci, urls_kaggle=kaggle), pkl_path=tmp_path / "out")

    assert (tmp_path / "out" / "final_dataset.pkl").exists()


def test_merge_keeps_rows_from_both_sources():
    # UCI ships a lowercase 'label' where 1 = legitimate, the opposite of label_binary.
    uci = pd.DataFrame(
        {
            "URL": ["https://www.uci-legit.example", "http://uci-phish.example/login"],
            "label": [1, 0],
        }
    )
    kaggle = preprocess_phishing(
        pd.DataFrame(
            {
                "URL": ["kaggle-bad.example/verify", "www.kaggle-good.example/about"],
                "Label": ["bad", "good"],
            }
        )
    )

    merged = merge_data_service(DataToMerge(urls_uci=uci, urls_kaggle=kaggle))
    stats = summarise_sources(uci, kaggle, merged)

    assert stats["rows_from_uci"] == 2
    labels = merged.set_index("URL")["label_binary"]
    assert labels["uci-legit.example"] == 0
    assert labels["uci-phish.example/login"] == 1


def test_golden_set_is_well_formed():
    golden = load_golden_set()

    assert set(golden["label"]) == {0, 1}
    assert golden["url"].is_unique
    assert not golden["url"].str.contains("#").any()
