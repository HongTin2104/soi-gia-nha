import pandas as pd
import pytest

from soi_gia_nha.data import load_raw, shard_names


def test_shard_names():
    assert shard_names(3) == ["shard_0000.parquet", "shard_0001.parquet", "shard_0002.parquet"]


def test_load_raw_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/download.py"):
        load_raw(src=tmp_path)


def test_load_raw_reads_shards_in_order(tmp_path):
    for i in range(3):
        pd.DataFrame({"price": [str(i)], "area": [float(i)]}).to_parquet(
            tmp_path / f"shard_{i:04d}.parquet"
        )
    df = load_raw(columns=["price"], shards=2, src=tmp_path)
    assert df["price"].tolist() == ["0", "1"]
    assert list(df.columns) == ["price"]


def test_download_fetches_every_shard_at_the_pinned_revision(tmp_path, monkeypatch):
    import huggingface_hub

    from soi_gia_nha import data

    calls = []

    def fake_download(repo_id, filename, repo_type, revision, local_dir):
        calls.append((repo_id, filename, repo_type, revision))
        return str(local_dir / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    paths = data.download(shards=2, dest=tmp_path / "raw")

    assert paths == [
        tmp_path / "raw" / "shard_0000.parquet",
        tmp_path / "raw" / "shard_0001.parquet",
    ]
    assert calls == [
        (data.REPO_ID, "shard_0000.parquet", "dataset", data.REVISION),
        (data.REPO_ID, "shard_0001.parquet", "dataset", data.REVISION),
    ]
    assert (tmp_path / "raw").is_dir()
