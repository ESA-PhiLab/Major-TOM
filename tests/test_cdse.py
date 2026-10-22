"""CDSE credentials and band names (offline; fake keys only)."""
import os
import stat

import pytest

pytest.importorskip("pystac_client")
from majortom.build import archives, credentials                 # noqa: E402
from majortom.build.collections import COLLECTIONS, processing_baseline   # noqa: E402


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    """No CDSE keys in the environment; the credentials file is a fresh path in tmp_path."""
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws" / "credentials"))
    return tmp_path / "aws" / "credentials"


def test_missing_keys_say_how_to_set_them_up(no_keys):
    with pytest.raises(PermissionError, match="CDSE_S3_ACCESS_KEY") as error:
        archives.cdse_keys()
    assert "majortom.build.credentials" in str(error.value)


def test_saved_profile_is_owner_only_and_keeps_other_sections(no_keys):
    no_keys.parent.mkdir()
    no_keys.write_text("[default]\naws_access_key_id = other\naws_secret_access_key = other\n")
    credentials.save_cdse("fake-key", "fake-secret")
    assert archives.cdse_keys() == ("fake-key", "fake-secret")
    assert archives.profile_keys("default") == ("other", "other")
    assert stat.S_IMODE(no_keys.stat().st_mode) == 0o600


def test_environment_wins_over_the_profile(no_keys, monkeypatch):
    credentials.save_cdse("file-key", "file-secret")
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "env-key")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "env-secret")
    assert archives.cdse_keys() == ("env-key", "env-secret")


def test_settings_stay_out_of_the_process_environment(no_keys):
    credentials.save_cdse("fake-key", "fake-secret")
    before = dict(os.environ)
    env = archives.cdse_env()
    assert env["AWS_S3_ENDPOINT"] == "eodata.dataspace.copernicus.eu" and env["AWS_SESSION_TOKEN"] == ""
    assert dict(os.environ) == before                            # keys only in the returned settings


def test_setup_command_never_prints_the_keys(no_keys, monkeypatch, capsys):
    answers = iter(["fake-key", "fake-secret"])
    monkeypatch.setattr(credentials.getpass, "getpass", lambda prompt: next(answers))
    assert credentials.main(["cdse", "--no-check"]) == 0
    out = capsys.readouterr()
    assert "fake-key" not in out.out + out.err and "fake-secret" not in out.out + out.err


def test_cdse_band_names_and_baseline():
    import datetime as dt
    import pystac

    s2 = COLLECTIONS["sentinel-2-l2a"].source("sentinel-2-l2a", "cdse")
    assert (s2.collection, s2.key("B04"), s2.key("B05"), s2.key("B01"), s2.key("SCL")) == (
        "sentinel-2-l2a", "B04_10m", "B05_20m", "B01_60m", "SCL_20m")
    assert COLLECTIONS["sentinel-2-l1c"].source("sentinel-2-l1c", "cdse").key("B10") == "B10"
    item = pystac.Item("x", None, None, dt.datetime(2021, 8, 25), {"processing:version": "05.00"})
    assert processing_baseline(item) == "05.00"
