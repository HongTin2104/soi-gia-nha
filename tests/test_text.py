import pandas as pd
import pytest

from soi_gia_nha.text import extract, fill_missing, prepare_text


def one(text: str) -> pd.Series:
    return extract(prepare_text(pd.Series([""]), pd.Series([text]))).iloc[0]


@pytest.mark.parametrize(
    ("text", "floors"),
    [
        ("Nhà 5 tầng, thang máy", 5),
        ("nhà 4tầng mới xây", 4),
        ("Kết cấu trệt 2 lầu sân thượng", 3),
        ("Nhà 1 trệt 1 lầu", 2),
        ("3 tầng 1 tum", 3),
        ("Đất 100m2", None),
        ("Tòa nhà 45 tầng", None),  # out of range for a house
    ],
)
def test_floors(text, floors):
    got = one(text)["txt_floors"]
    assert (pd.isna(got) and floors is None) or got == floors


@pytest.mark.parametrize(
    ("text", "bedrooms"),
    [("Căn 2PN 2WC", 2), ("3 phòng ngủ rộng", 3), ("nhà 4 ngủ khép kín", 4), ("căn studio", None)],
)
def test_bedrooms(text, bedrooms):
    got = one(text)["txt_bedrooms"]
    assert (pd.isna(got) and bedrooms is None) or got == bedrooms


def test_widths_accept_comma_decimals():
    row = one("Mặt tiền 4,5m, hẻm 6m thông")
    assert row["txt_frontage"] == 4.5
    assert row["txt_road"] == 6


def test_flags():
    row = one("Bán gấp nhà hẻm xe hơi đỗ cửa, sổ hồng riêng, full nội thất, lô góc")
    assert row["urgent"] and row["car_access"] and row["legal_book"]
    assert row["furnished"] and row["corner"] and row["alley"]
    assert not row["legal_weak"] and not row["elevator"]


def test_weak_legal_papers():
    assert one("Nhà sổ chung, vi bằng công chứng")["legal_weak"]
    assert one("Mua bán bằng HĐMB")["legal_weak"]


def test_title_is_read_too():
    feats = extract(prepare_text(pd.Series(["Bán gấp nhà 3 tầng"]), pd.Series([None])))
    assert feats["txt_floors"].iloc[0] == 3
    assert feats["urgent"].iloc[0]


def test_fill_missing_prefers_structured_values():
    df = pd.DataFrame(
        {
            "floor_count": [4.0, None],
            "bedroom_count": [None, None],
            "frontage_width": [None, None],
            "road_width": [None, None],
        }
    )
    feats = extract(pd.Series(["nhà 5 tầng", "nhà 5 tầng 3pn"]))
    out = fill_missing(df, feats)
    assert out["floor_count"].tolist() == [4.0, 5.0]
    assert out["bedroom_count"].isna().iloc[0]
    assert out["bedroom_count"].iloc[1] == 3
    assert "urgent" in out


@pytest.mark.parametrize(
    ("text", "column", "expected"),
    [
        ("hẻm 3m5", "txt_road", 3.5),
        ("mặt tiền 4m5", "txt_frontage", 4.5),
        ("ngang rộng 4m5", "txt_frontage", 4.5),
        ("nhà 4 tầng hẻm xe hơi 6m", "txt_road", 6),
        ("ngõ ô tô rộng 3m5", "txt_road", 3.5),
        ("ngõ 2,5m", "txt_road", 2.5),
        ("mặt tiền 4.5 m", "txt_frontage", 4.5),
        ("hẻm trước nhà rộng 6 m", "txt_road", 6),
        ("hẻm 3m50", "txt_road", None),
        ("mặt tiền 0m5", "txt_frontage", None),
        ("mặt tiền 51m5", "txt_frontage", None),
    ],
)
def test_compact_widths_and_car_alleys(text, column, expected):
    value = one(text)[column]
    if expected is None:
        assert pd.isna(value)
    else:
        assert value == expected
