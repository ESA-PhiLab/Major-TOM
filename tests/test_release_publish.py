"""majortom.release.publish: plan, upload, resume and verify against a fake S3 server (moto; offline)."""
import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from majortom.release.publish import _source_coop, publish   # noqa: E402


@pytest.fixture
def release_folder(tmp_path):
    folder = tmp_path / "s2"
    (folder / ".tacocat").mkdir(parents=True)
    (folder / "s2_part0001.zip").write_bytes(b"a" * 300_000)
    (folder / "s2_part0002.zip").write_bytes(b"b" * 200_000)
    (folder / ".tacocat" / "catalog.parquet").write_bytes(b"c" * 1_000)
    (folder / ".stage-junk").write_bytes(b"not part of the release")
    return folder


@pytest.fixture
def s3():
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="bucket")
        yield client


def test_upload_resume_and_change(release_folder, s3):
    first = publish(release_folder, "s3://bucket/major-tom/release-test/s2", client=s3)
    assert first.verified and len(first.uploaded) == 3 and not first.skipped
    assert first.uploaded[-1] == ".tacocat/catalog.parquet"                       # the catalogue goes last
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket="bucket")["Contents"]}
    assert "major-tom/release-test/s2/.stage-junk" not in keys

    again = publish(release_folder, "s3://bucket/major-tom/release-test/s2", client=s3)
    assert not again.uploaded and len(again.skipped) == 3 and again.verified      # resumable: nothing re-sent

    (release_folder / "s2_part0002.zip").write_bytes(b"B" * 200_000)              # same size, new content
    changed = publish(release_folder, "s3://bucket/major-tom/release-test/s2", client=s3)
    assert changed.uploaded == ["s2_part0002.zip"] and changed.verified


def test_include_root_files(release_folder, s3):
    (release_folder / "README.md").write_text("# test")
    (release_folder / "figures").mkdir()
    (release_folder / "figures" / "gallery.png").write_bytes(b"png")
    r = publish(release_folder, "s3://bucket/repo", client=s3, include=["README.md", "figures/*.png"])
    assert r.uploaded == ["README.md", "figures/gallery.png"] and r.verified
    assert s3.head_object(Bucket="bucket", Key="repo/figures/gallery.png")["ContentType"] == "image/png"
    again = publish(release_folder, "s3://bucket/repo", client=s3, include=["README.md"], force=True)
    assert again.uploaded == ["README.md"]                                       # force re-sends unchanged files


def test_dry_run_uploads_nothing(release_folder, s3):
    plan = publish(release_folder, "s3://bucket/x", client=s3, dry_run=True)
    assert len(plan.uploaded) == 3 and "Contents" not in s3.list_objects_v2(Bucket="bucket")


def test_source_coop_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    for v in ("SOURCE_COOP_ACCESS_KEY_ID", "SOURCE_COOP_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(PermissionError):
        _source_coop("source.coop://major-tom/release-test")
    monkeypatch.setenv("SOURCE_COOP_ACCESS_KEY_ID", "k")                          # route A: API key, data proxy
    monkeypatch.setenv("SOURCE_COOP_SECRET_ACCESS_KEY", "s")
    client, bucket, prefix = _source_coop("source.coop://major-tom/release-test/s2")
    assert (bucket, prefix) == ("major-tom", "release-test/s2") and client.meta.endpoint_url == "https://data.source.coop"
    monkeypatch.setenv("SOURCE_COOP_SESSION_TOKEN", "t")                         # temporary proxy credentials
    client, bucket, _ = _source_coop("source.coop://major-tom/release-test/s2")
    assert bucket == "major-tom" and client._request_signer._credentials.token == "t"
    monkeypatch.delenv("SOURCE_COOP_SESSION_TOKEN")
    monkeypatch.delenv("SOURCE_COOP_ACCESS_KEY_ID")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "t")                                  # route B: temporary credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    _, bucket, prefix = _source_coop("source.coop://major-tom/release-test/s2")
    assert (bucket, prefix) == ("eu-central-1.opendata.source.coop", "major-tom/release-test/s2")
    monkeypatch.setenv("SOURCE_COOP_REGION", "us-west-2")                         # e.g. Core-DEM's connection
    assert _source_coop("source.coop://major-tom/core-dem")[1] == "us-west-2.opendata.source.coop"


def test_multipart_etag_matches_s3(tmp_path, s3):
    """A file above the multipart size gets the ETag S3 computes for it, so a rerun skips it."""
    import majortom.release.publish as pub

    folder = tmp_path / "big"
    folder.mkdir()
    (folder / "x_part0001.zip").write_bytes(b"z" * (pub.CHUNK * 2 + 123))
    first = publish(folder, "s3://bucket/big", client=s3)
    assert first.verified and "-" in pub._sums(folder, [folder / "x_part0001.zip"])["x_part0001.zip"][1]
    assert not publish(folder, "s3://bucket/big", client=s3).uploaded
