import json
import unicodedata

import pandas as pd
import pytest

from soi_gia_nha import admin
from soi_gia_nha.admin import attach_new_units, build_map, canon, save_map


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Phường 07", "7"),
        ("Quận Cầu Giấy", "cầu giấy"),
        ("Trung Hòa", "trung hoà"),
        ("Trung Hoà", "trung hoà"),
        ("Thụy Lâm", "thuỵ lâm"),
        ("Hoàng Mai", "hoàng mai"),
        ("  Thành phố   Hồ Chí Minh ", "hồ chí minh"),
        (None, None),
        (float("nan"), None),
    ],
)
def test_canon(raw, expected):
    assert canon(raw) == expected


def test_canon_handles_decomposed_input():
    nfd = unicodedata.normalize("NFD", "Phường Trung Hòa")
    assert nfd != "Phường Trung Hòa"
    assert canon(nfd) == "trung hoà"


def test_attach_new_units():
    admin_map = pd.DataFrame(
        {
            "province_key": ["hồ chí minh", "hồ chí minh"],
            "district_key": ["7", "1"],
            "ward_key": ["tân quy", "7"],
            "new_province": ["Thành phố Hồ Chí Minh"] * 2,
            "new_ward": ["Phường Tân Hưng", "Phường A / Phường B"],
            "ambiguous": [False, True],
        }
    )
    listings = pd.DataFrame(
        {
            "province_name": ["Hồ Chí Minh", "Hồ Chí Minh", "Hồ Chí Minh"],
            "district_name": ["7", "1", "7"],
            "ward_name": ["Tân Quy", "07", None],
        },
        index=[10, 11, 12],
    )
    out = attach_new_units(listings, admin_map)
    assert out.index.tolist() == [10, 11, 12]
    assert out["new_ward"].tolist()[:2] == ["Phường Tân Hưng", "Phường A / Phường B"]
    assert pd.isna(out["new_ward"].iloc[2])
    assert out["ambiguous"].tolist() == [False, True, False]


# Old units: [id, name, type, ?, children]. Two provinces, one old ward split between two new ones.
OLD_UNITS = [
    [
        "01",
        "Thành phố Hà Nội",
        "",
        "",
        [
            [
                "005",
                "Quận Cầu Giấy",
                "",
                "",
                [["00157", "Phường Trung Hòa", ""], ["00160", "Phường Yên Hòa", ""]],
            ]
        ],
    ],
    [
        "79",
        "Thành phố Hồ Chí Minh",
        "",
        "",
        [["778", "Quận 7", "", "", [["27259", "Phường Tân Quy", ""]]]],
    ],
]
SPLITS = {
    "data": [
        {
            "level1_id": "01",
            "splits": [{"name": "Thành phố Hà Nội"}],
            "level3s": [
                {"level3_id": "00157", "splits": [{"name": "Phường Yên Hoà"}]},
                {
                    "level3_id": "00160",
                    "splits": [{"name": "Phường Cầu Giấy"}, {"name": "Phường Yên Hoà"}],
                },
            ],
        },
        {"level1_id": "79", "splits": [{"name": "Thành phố Hồ Chí Minh"}], "level3s": []},
    ]
}


@pytest.fixture
def cache(tmp_path):
    (tmp_path / "sorted.json").write_text(
        json.dumps(OLD_UNITS, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "splits.json").write_text(json.dumps(SPLITS, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_build_map_marks_split_wards_as_ambiguous(cache):
    m = build_map(cache=cache).set_index("ward_key")
    assert m.loc["trung hoà", "new_ward"] == "Phường Yên Hoà"
    assert not m.loc["trung hoà", "ambiguous"]
    assert m.loc["yên hoà", "new_ward"] == "Phường Cầu Giấy / Phường Yên Hoà"
    assert m.loc["yên hoà", "ambiguous"]
    assert m.loc["tân quy", "new_province"] == "Thành phố Hồ Chí Minh"
    assert pd.isna(m.loc["tân quy", "new_ward"])  # no mapping published for this ward


def test_build_map_keeps_only_the_requested_provinces(cache):
    m = build_map(provinces=["Hà Nội"], cache=cache)
    assert set(m["province_key"]) == {"hà nội"}
    assert len(m) == 2


def test_fetch_downloads_once_and_reuses_the_cache(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("[1]", encoding="utf-8")
    dest = tmp_path / "cache" / "copy.json"

    assert admin._fetch(source.as_uri(), dest).read_text(encoding="utf-8") == "[1]"
    source.write_text("[2]", encoding="utf-8")
    assert admin._fetch(source.as_uri(), dest).read_text(encoding="utf-8") == "[1]"


def test_save_map_round_trips(tmp_path):
    admin_map = pd.DataFrame({"province_key": ["hà nội"], "new_ward": [None], "ambiguous": [False]})
    path = save_map(admin_map, tmp_path / "nested" / "admin_map.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(path), admin_map)
