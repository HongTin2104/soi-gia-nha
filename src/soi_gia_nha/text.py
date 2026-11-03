"""Read structured facts out of the listing title and description.

The structured columns are mostly empty (floors 77% missing, road width 81%), but sellers
usually write those details in the text: "nhà 5 tầng", "trệt 2 lầu", "hẻm xe hơi",
"sổ hồng riêng", "bán gấp". Plain regular expressions get most of it, they are fast on
1.5M rows and every decision can be traced back to a pattern.
"""

from __future__ import annotations

import pandas as pd

_WIDTH = r"(\d{1,2}m\d\b|\d{1,2}(?:[.,]\d)?\s*m\b)"

# Southern listings count floors as ground floor (trệt) + upper floors (lầu):
# "trệt 2 lầu" is 3 floors, the same house the North calls "3 tầng".
FLOORS_TANG = r"(\d{1,2})\s*(?:tầng|tấm)\b"
FLOORS_LAU = r"(\d{1,2})\s*lầu\b"
BEDROOMS = r"(\d{1,2})\s*(?:phòng ngủ|pn\b|ngủ\b)"
FRONTAGE = r"(?:mặt tiền|ngang)\s*(?:rộng\s*)?" + _WIDTH
ROAD = (
    r"(?:đường|lộ|ngõ|ngách|hẻm|kiệt)\s*(?:trước nhà\s*)?"
    r"(?:(?:xe hơi|ô tô|oto|xe tải)\s*)?(?:rộng\s*)?" + _WIDTH
)

FLAGS = {
    "legal_book": r"sổ (?:hồng|đỏ|riêng)|\bshr\b|\bsđcc\b",
    "legal_weak": r"sổ chung|vi bằng|giấy tay|chưa có sổ|chờ sổ|\bhđmb\b|hợp đồng mua bán",
    "car_access": (
        r"(?:hẻm|ngõ|ngách|kiệt|đường)[^.\n]{0,15}(?:xe hơi|ô tô|oto|xe tải)"
        r"|(?:xe hơi|ô tô|oto)\s*(?:đỗ|đậu|vào|tránh|tới|đến|chạy)"
    ),
    "main_street": r"mặt phố|mặt đường|nhà mặt tiền|mặt tiền đường|\bmtđ\b",
    "alley": r"\b(?:hẻm|ngõ|ngách|kiệt)\b",
    "corner": r"lô góc|căn góc|nhà góc|góc hai mặt|góc 2 mặt|hai mặt tiền|2 mặt tiền",
    "elevator": r"thang máy",
    "furnished": r"(?:full|đầy đủ|nguyên)\s*nội thất|nội thất (?:đầy đủ|cao cấp|xịn|full)",
    "business": r"kinh doanh|cho thuê",
    "urgent": r"bán gấp|cần tiền|cắt lỗ|ngộp|hạ giá|giảm giá",
}

NUMERIC = ["txt_floors", "txt_bedrooms", "txt_frontage", "txt_road"]
FEATURES = NUMERIC + list(FLAGS)


def _first_number(text: pd.Series, pattern: str) -> pd.Series:
    found = text.str.extract(pattern, expand=False)
    # Widths may put the metre unit between digits (3m5), or after them (3.5 m).
    found = found.str.replace(r"\s*m$", "", regex=True)
    found = found.str.replace("m", ".", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(found, errors="coerce")


def prepare_text(name: pd.Series, description: pd.Series) -> pd.Series:
    return (name.fillna("") + " . " + description.fillna("")).str.lower()


def extract(text: pd.Series) -> pd.DataFrame:
    """One row per listing. ``text`` should come from :func:`prepare_text`."""
    tang = _first_number(text, FLOORS_TANG)
    lau = _first_number(text, FLOORS_LAU)
    out = pd.DataFrame(index=text.index)
    out["txt_floors"] = tang.fillna(lau + 1).where(lambda s: s.between(1, 30))
    out["txt_bedrooms"] = _first_number(text, BEDROOMS).where(lambda s: s.between(1, 20))
    out["txt_frontage"] = _first_number(text, FRONTAGE).where(lambda s: s.between(1, 50))
    out["txt_road"] = _first_number(text, ROAD).where(lambda s: s.between(1, 100))
    for name, pattern in FLAGS.items():
        out[name] = text.str.contains(pattern, regex=True)
    return out


def fill_missing(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Use the structured value when the seller filled it in, the text value otherwise."""
    out = df.copy()
    pairs = {
        "floor_count": "txt_floors",
        "bedroom_count": "txt_bedrooms",
        "frontage_width": "txt_frontage",
        "road_width": "txt_road",
    }
    for col, txt in pairs.items():
        out[col] = out[col].fillna(feats[txt])
    for flag in FLAGS:
        out[flag] = feats[flag]
    return out
