"""Build a model-ready row from a listing someone pasted into the app."""

from __future__ import annotations

import re

import pandas as pd

from .text import extract, prepare_text

_DEC = r"(\d+(?:[.,]\d+)?)"
# "9,1 tỷ", "9 tỷ 500", "9 tỷ 5", "850 triệu", "850tr"
_PRICE_TY = re.compile(_DEC + r"\s*(?:tỷ|tỉ|ty)(?![^\W\d_])\s*(\d{1,3})?(?!\s*(?:m|/))")
_PRICE_TR = re.compile(_DEC + r"\s*(?:triệu|tr)\b(?!\s*/)")
_AREA = re.compile(_DEC + r"\s*(?:m2|m²|mét vuông)")
# A bare "m" is a width ("mặt tiền 5m") unless the text says it is the area.
_AREA_LABELLED = re.compile(r"(?:dt|diện tích)\s*:?\s*" + _DEC + r"\s*m\b")


def _num(s: str) -> float:
    # "9,1" and "9.1" are decimals; "1.200" is a thousands separator.
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        return float(s.replace(".", ""))
    return float(s.replace(",", "."))


def parse_price(text: str) -> float | None:
    """First total price in the text, in VND. Prices per m2 ("120 triệu/m2") are skipped."""
    t = text.lower()
    m = _PRICE_TY.search(t)
    if m:
        value = _num(m.group(1)) * 1e9
        if m.group(2):  # "9 tỷ 500" means 9.5 billion, "9 tỷ 5" too
            tail = m.group(2)
            value += int(tail) * 10 ** (9 - len(tail))
        return value
    m = _PRICE_TR.search(t)
    if m:
        return _num(m.group(1)) * 1e6
    return None


def parse_area(text: str) -> float | None:
    t = text.lower()
    m = _AREA.search(t) or _AREA_LABELLED.search(t)
    if not m:
        return None
    area = _num(m.group(1))
    return area if 5 <= area <= 100_000 else None


def build(
    text: str,
    *,
    province: str,
    district: str,
    property_type: str,
    ward: str | None = None,
    project: str | None = None,
    price: float | None = None,
    area: float | None = None,
    published_at: str | pd.Timestamp | None = None,
    **details: float | None,
) -> pd.DataFrame:
    """One-row frame for ``PriceModel``. Values passed explicitly win over the text."""
    feats = extract(prepare_text(pd.Series([""]), pd.Series([text]))).iloc[0]
    row = {
        "name": "",
        "description": text,
        "province_name": province,
        "district_name": district,
        "ward_name": ward,
        "project_name": project,
        "property_type_name": property_type,
        "price": price if price is not None else parse_price(text),
        "area": area if area is not None else parse_area(text),
        "published_at": pd.Timestamp(published_at or pd.Timestamp.now()),
        "floor_count": details.get("floor_count") or feats["txt_floors"],
        "bedroom_count": details.get("bedroom_count") or feats["txt_bedrooms"],
        "bathroom_count": details.get("bathroom_count"),
        "frontage_width": details.get("frontage_width") or feats["txt_frontage"],
        "house_depth": details.get("house_depth"),
        "road_width": details.get("road_width") or feats["txt_road"],
        "house_direction": details.get("house_direction"),
        "balcony_direction": details.get("balcony_direction"),
    }
    for flag in feats.index:
        if not flag.startswith("txt_"):
            row[flag] = bool(feats[flag])
    df = pd.DataFrame([row])
    if df["price"].notna().all() and df["area"].notna().all():
        df["price_per_m2"] = df["price"] / df["area"]
    return df
