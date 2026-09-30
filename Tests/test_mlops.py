"""Contracts for the MLOps layer: promotion gates, data-source checks, golden set."""
import pandas as pd

from Classes.DataToMerge import DataToMerge
from Services.merge_data import merge_data_service
from Services.model_registry import PromotionCriteria, load_golden_set, promotion_decision
from Utilities.Services.merge_data_utils import summarise_sources
from Utilities.Services.preprocess_data_utils import preprocess_phishing

CRITERIA = PromotionCriteria(
    metric="F1-Score",
    min_improvement=0.002,
    recall_tolerance=0.005,
    max_bundle_mb=50,
    golden_min_accuracy=0.75,
)


def _metrics(f1=0.90, recall=0.90, golden=1.0, size=2.0):
    return {"F1-Score": f1, "Recall": recall, "golden_set_accuracy": golden, "bundle_size_mb": size}


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
    }


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
