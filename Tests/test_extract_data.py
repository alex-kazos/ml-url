"""The UCI download falls back to the saved copy when the UCI site is unreachable."""
from types import SimpleNamespace

import pandas as pd
import pytest
from ucimlrepo.fetch import DatasetNotFoundError

import Utilities.Services.extract_data_utils as extract_utils
from Utilities.config import UCI_PHISHING_FILE

UCI_ROWS = pd.DataFrame({"URL": ["https://www.uci-legit.example"], "label": [1]})


def _download_fails(id):
    raise DatasetNotFoundError('Error reading data csv file for "PhiUSIIL" dataset (id=967).')


def test_download_saves_a_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        extract_utils,
        "fetch_ucirepo",
        lambda id: SimpleNamespace(data=SimpleNamespace(original=UCI_ROWS)),
    )

    df = extract_utils.extract_uci_data(uci_repo=967, data_path=tmp_path)

    pd.testing.assert_frame_equal(df, UCI_ROWS)
    pd.testing.assert_frame_equal(pd.read_pickle(tmp_path / UCI_PHISHING_FILE), UCI_ROWS)


@pytest.mark.parametrize(
    "error", [DatasetNotFoundError("csv unreadable"), ConnectionError("server down")]
)
def test_failed_download_uses_saved_copy(monkeypatch, tmp_path, capsys, error):
    UCI_ROWS.to_pickle(tmp_path / UCI_PHISHING_FILE)

    def _fails(id):
        raise error

    monkeypatch.setattr(extract_utils, "fetch_ucirepo", _fails)

    df = extract_utils.extract_uci_data(uci_repo=967, data_path=tmp_path)

    pd.testing.assert_frame_equal(df, UCI_ROWS)
    assert "WARNING: could not download UCI dataset 967" in capsys.readouterr().out


def test_failed_download_without_saved_copy_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(extract_utils, "fetch_ucirepo", _download_fails)

    with pytest.raises(RuntimeError, match="no saved copy"):
        extract_utils.extract_uci_data(uci_repo=967, data_path=tmp_path)


def test_failed_download_with_unreadable_copy_raises(monkeypatch, tmp_path):
    (tmp_path / UCI_PHISHING_FILE).write_bytes(b"not a pickle")
    monkeypatch.setattr(extract_utils, "fetch_ucirepo", _download_fails)

    with pytest.raises(RuntimeError, match="cannot be read"):
        extract_utils.extract_uci_data(uci_repo=967, data_path=tmp_path)
