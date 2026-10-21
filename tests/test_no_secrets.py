"""No credentials in tracked files: archive keys must stay in the environment or the user's own files."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "AWS-style access key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "secret assigned a value": re.compile(
        r"(CDSE_S3_SECRET_KEY|AWS_SECRET_ACCESS_KEY|EE_PRIVATE_KEY)\s*[=:]\s*[\"']?[A-Za-z0-9/+]{20,}"),
}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / p for p in out.decode().split("\0") if p]


def test_no_credentials_in_tracked_files():
    found = []
    for path in tracked_files():
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):           # binary or missing: no text secrets
            continue
        found += [f"{path.relative_to(ROOT)}: {kind}" for kind, pattern in PATTERNS.items() if pattern.search(text)]
    assert not found, "possible credentials (values not shown):\n" + "\n".join(found)
