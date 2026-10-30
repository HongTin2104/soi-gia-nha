"""Run every script in scripts/ on a small synthetic snapshot, in the order the README gives.

The real pipeline needs the 1.7 GB dataset. This checks that the steps fit together and that
each one writes what the next one reads, using a temporary project directory.
"""

import functools
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soi_gia_nha.data import COLUMNS, load_raw
from soi_gia_nha.model import PriceModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_app_data  # noqa: E402
import build_history  # noqa: E402
import download  # noqa: E402
import eval_checks  # noqa: E402
import eval_text  # noqa: E402
import prepare  # noqa: E402
import train  # noqa: E402

WARDS = {
    ("Hà Nội", "Cầu Giấy"): ["Dịch Vọng", "Yên Hòa"],
    ("Hà Nội", "Hà Đông"): ["Văn Quán", "Mộ Lao"],
    ("Hồ Chí Minh", "Quận 7"): ["Tân Quy", "Tân Phong"],
}
BASE_PPM = {"Cầu Giấy": 180e6, "Hà Đông": 110e6, "Quận 7": 140e6}


def raw_snapshot(n: int = 1500, reposted: int = 40, seed: int = 0) -> pd.DataFrame:
    """Listings shaped like the raw shards, June 2025 to March 2026, with some reposts."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-06-01", "2026-03-31", freq="D")
    rows = []
    for i in range(n):
        province, district = list(WARDS)[i % len(WARDS)]
        ward = WARDS[(province, district)][rng.integers(2)]
        kind = "Nhà" if i % 3 else "Căn hộ chung cư"
        area = round(float(rng.uniform(40, 120)), 1)
        floors = int(rng.integers(2, 7))
        price = BASE_PPM[district] * area * rng.lognormal(0, 0.15) * (1 + 0.03 * floors)
        rows.append(
            {
                "name": f"Bán {kind.lower()} {district} {floors} tầng",
                "description": (
                    f"{kind} {floors} tầng, {rng.integers(1, 5)} phòng ngủ, mặt tiền 4m5, "
                    f"hẻm 3m5, sổ hồng chính chủ, gần chợ và trường học. Mã tin {i}."
                ),
                "property_type_name": kind,
                "province_name": province,
                "district_name": district,
                "ward_name": ward,
                "street_name": None,
                "project_name": None,
                "price": str(int(price)),
                "area": area,
                "floor_count": float(floors) if i % 2 else np.nan,
                "frontage_width": np.nan,
                "house_depth": np.nan,
                "road_width": np.nan,
                "bedroom_count": np.nan,
                "bathroom_count": np.nan,
                "house_direction": None,
                "balcony_direction": None,
                "published_at": str(days[rng.integers(len(days) - 60)]),
            }
        )
    # The same property posted three times, 20 days apart, a little cheaper each time.
    for i in range(reposted):
        first = dict(rows[i])
        for step in range(1, 3):
            later = dict(first)
            later["price"] = str(int(int(first["price"]) * (1 - 0.05 * step)))
            later["published_at"] = str(pd.Timestamp(first["published_at"]) + pd.Timedelta(days=20 * step))
            rows.append(later)
    return pd.DataFrame(rows, columns=COLUMNS)


def admin_map() -> pd.DataFrame:
    from soi_gia_nha.admin import canon

    rows = [
        {
            "province_key": canon(province),
            "district_key": canon(district),
            "ward_key": canon(ward),
            "new_province": province,
            "new_ward": f"Phường {ward}",
            "ambiguous": False,
        }
        for (province, district), wards in WARDS.items()
        for ward in wards
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project root with two raw shards; every script is pointed at it."""
    raw, processed, reports = tmp_path / "data" / "raw", tmp_path / "data" / "processed", tmp_path / "reports"
    for d in (raw, processed, reports):
        d.mkdir(parents=True)
    snapshot = raw_snapshot()
    half = len(snapshot) // 2
    snapshot.iloc[:half].to_parquet(raw / "shard_0000.parquet")
    snapshot.iloc[half:].to_parquet(raw / "shard_0001.parquet")

    for module in (prepare, eval_text, train, eval_checks, build_history, build_app_data):
        for name, value in {
            "ROOT": tmp_path,
            "RAW_DIR": raw,
            "PROCESSED_DIR": processed,
            "MODEL_DIR": tmp_path / "models" / "v0.1",
            "DATA_DIR": tmp_path / "data" / "app",
            "load_raw": functools.partial(load_raw, src=raw),
        }.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(prepare, "build_map", lambda provinces: admin_map())
    monkeypatch.setattr(
        prepare, "save_map", lambda m: m.to_parquet(processed / "admin_map.parquet", index=False)
    )
    monkeypatch.setattr(sys, "argv", ["script"])
    return tmp_path


def test_pipeline_end_to_end(project, capsys):
    processed = project / "data" / "processed"

    prepare.main()
    listings = pd.read_parquet(processed / "listings.parquet")
    assert len(listings) > 1000
    assert listings["new_ward"].notna().all()
    assert listings["floor_count"].notna().all()  # half of them only say "N tầng" in the text
    assert (processed / "admin_map.parquet").exists()

    eval_text.main()
    report = (project / "reports" / "text_extraction.md").read_text(encoding="utf-8")
    assert "| Floors |" in report
    assert "| Frontage width (m) |" in report

    train.main()
    model = PriceModel.load(project / "models" / "v0.1")
    assert set(model.boosters) == {"low", "mid", "high"}
    report = (project / "reports" / "model.md").read_text(encoding="utf-8")
    assert "| LightGBM |" in report and "| District median |" in report

    eval_checks.main()
    report = (project / "reports" / "checks.md").read_text(encoding="utf-8")
    assert "| Verdict | Listings |" in report

    build_history.main()
    posts = pd.read_parquet(processed / "posts.parquet")
    assert "description" not in posts  # dropped to keep the app bundle small
    assert posts.groupby("property_id").size().max() >= 3  # the reposts were linked
    assert len(pd.read_parquet(processed / "samples.parquet")) > 0
    assert "| Posts |" in (project / "reports" / "history.md").read_text(encoding="utf-8")

    build_app_data.main()
    app_dir = project / "data" / "app"
    with zipfile.ZipFile(project / "data" / f"soi-gia-nha-data-{build_app_data.DATA_VERSION}.zip") as z:
        assert sorted(z.namelist()) == sorted(build_app_data.DATA_FILES)
    assert all((app_dir / f).exists() for f in build_app_data.DATA_FILES)


def test_train_with_sample_prints_the_report_without_saving(project, monkeypatch, capsys):
    prepare.main()
    monkeypatch.setattr(sys, "argv", ["train.py", "--sample", "0.5"])
    train.main()
    assert "# Price model" in capsys.readouterr().out
    assert not (project / "models").exists()
    assert not (project / "reports" / "model.md").exists()


def test_download_prints_each_shard(tmp_path, monkeypatch, capsys):
    shard = tmp_path / "shard_0000.parquet"
    shard.write_bytes(b"x" * 2_000_000)
    asked = []
    monkeypatch.setattr(download, "download", lambda n: asked.append(n) or [shard])
    monkeypatch.setattr(sys, "argv", ["download.py", "--shards", "1"])

    download.main()

    assert asked == [1]
    assert capsys.readouterr().out.split() == ["2.0", "MB", "shard_0000.parquet"]
