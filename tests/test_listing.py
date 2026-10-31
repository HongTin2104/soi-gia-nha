import pandas as pd
import pytest

from soi_gia_nha.listing import build, parse_area, parse_price


@pytest.mark.parametrize(
    ("text", "price"),
    [
        ("Giá 9,1 tỷ có thương lượng", 9.1e9),
        ("giá: 9.1 tỷ", 9.1e9),
        ("Chào 9 tỷ 500", 9.5e9),
        ("chỉ 9 tỷ 5 TL", 9.5e9),
        ("Giá 116 tỷ", 116e9),
        ("Giá 8.5 tỉ", 8.5e9),
        ("Chào 8 tỉ 5", 8.5e9),
        ("Bán 8ty5", 8.5e9),
        ("Bán 8tỷ5", 8.5e9),
        ("Bán gấp 850 triệu", 850e6),
        ("giá 850tr", 850e6),
        ("Đơn giá 120 triệu/m2, tổng 6 tỷ", 6e9),
        ("Liên hệ để biết giá", None),
    ],
)
def test_parse_price(text, price):
    assert parse_price(text) == pytest.approx(price) if price else parse_price(text) is None


@pytest.mark.parametrize(
    ("text", "area"),
    [
        ("Nhà 48m2, 4 tầng", 48),
        ("DT: 70.12m2", 70.12),
        ("diện tích 300 m²", 300),
        ("mặt tiền 5m, dài 12m, 60m2", 60),
        ("DT 45m, nở hậu", 45),
        ("Không ghi diện tích", None),
        # Width × length forms
        ("4x15m", 60),
        ("DT 4,5x20m", 90),
        ("4.5 x 12 m", 54),
        ("ngang 5×18", 90),
        ("60m2, 4x15m", 60),  # explicit area wins
        ("4x15m, DT: 70m2", 70),
    ],
)
def test_parse_area(text, area):
    assert parse_area(text) == pytest.approx(area) if area else parse_area(text) is None


def test_build_reads_text_and_respects_overrides():
    text = "Bán nhà 4 tầng hẻm xe hơi, sổ hồng riêng, 50m2, giá 9 tỷ"
    df = build(text, province="Hồ Chí Minh", district="Gò Vấp", property_type="Nhà", floor_count=5)
    row = df.iloc[0]
    assert row["price"] == 9e9 and row["area"] == 50
    assert row["price_per_m2"] == 9e9 / 50
    assert row["floor_count"] == 5  # explicit value wins over "4 tầng"
    assert row["car_access"] and row["legal_book"]
    assert isinstance(row["published_at"], pd.Timestamp)


def test_build_fills_frontage_from_dimensions():
    text = "Nhà 4x15m, giá 8 tỷ"
    df = build(text, province="Hà Nội", district="Ba Đình", property_type="Nhà")
    row = df.iloc[0]
    assert row["area"] == pytest.approx(60)
    assert row["frontage_width"] == pytest.approx(4)
