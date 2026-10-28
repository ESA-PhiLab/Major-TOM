"""Earth Data Hub keys per person: edh-key (own) or edh-key-<name>, picked with EDH_KEY_NAME (no network)."""
import stat

import pytest

from majortom.build import credentials as c


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("MAJORTOM_CONFIG", str(tmp_path))
    monkeypatch.delenv("EDH_KEY", raising=False)
    monkeypatch.delenv("EDH_KEY_NAME", raising=False)
    c.save_edh("own-key")
    c.save_edh("colleague-key", "ali")
    return tmp_path


def test_files_and_permissions(config):
    assert c.edh_key_file() == config / "edh-key"
    assert c.edh_key_file("ali") == config / "edh-key-ali"
    assert stat.S_IMODE((config / "edh-key-ali").stat().st_mode) == 0o600


def test_default_is_own_key(config):
    assert c.edh_key() == "own-key"


def test_name_from_environment(config, monkeypatch):
    monkeypatch.setenv("EDH_KEY_NAME", "ali")
    assert c.edh_key() == "colleague-key"


def test_edh_key_variable_still_wins(config, monkeypatch):
    monkeypatch.setenv("EDH_KEY_NAME", "ali")
    monkeypatch.setenv("EDH_KEY", "env-key")
    assert c.edh_key() == "env-key"
    assert c.edh_key("ali") == "colleague-key"            # an explicit name reads its file


def test_missing_name_explains(config, monkeypatch):
    monkeypatch.setenv("EDH_KEY_NAME", "nobody")
    with pytest.raises(PermissionError, match="Earth Data Hub"):
        c.edh_key()
