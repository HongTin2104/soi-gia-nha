# soi-gia-nha

[![CI](https://github.com/HongTin2104/soi-gia-nha/actions/workflows/ci.yml/badge.svg)](https://github.com/HongTin2104/soi-gia-nha/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/HongTin2104/soi-gia-nha)](https://github.com/HongTin2104/soi-gia-nha/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data and model: CC BY-NC 4.0](https://img.shields.io/badge/data%20%26%20model-CC%20BY--NC%204.0-lightgrey)](#license)

**Is this listing overpriced, and how long has it been on the market?** Paste a house or
apartment listing from Hanoi or Ho Chi Minh City. You get a fair price range with the reasons
behind it, the listing's own history (first posted when, reposted how often, price cut or
raised), the price trend in its ward, and the closest similar properties. Built on 2.2 million
real listings from June 2025 to March 2026.

<p align="center"><img src="docs/demo.png" width="860" alt="Demo: a Thủ Đức house listed at 10 billion VND has been on the market for 9 months and was cut 9%; the ward trend and five similar houses are shown below the verdict"></p>

## Why

Someone looking at *"Bán nhà 4 tầng Linh Chiểu, 86m², 10 tỷ"* has no quick way to tell whether
that is a good price. Listing sites show a price, sometimes a price per m², never *why* a place
costs what it costs, and never **how long it has been for sale**.

That last part is hidden on purpose. Agents repost the same property again and again so it
shows up as new. In this data:

- **35% of all posts repeat a property that was already listed**, one of them 125 times;
- **74,497 properties changed their asking price** between posts, and 45% of those went *up*;
- **69,665 properties stayed on the market for two months or more.**

A buyer who knows a house has sat unsold for nine months and was already cut 9% is in a very
different position from one who thinks it was posted yesterday.

For a single listing, this project answers:

1. **What is a fair asking price here?** A range, not one number, with the reasons.
2. **Has it been listed before?** First seen when, how many posts, every price change.
3. **What is the ward doing?** Monthly price per m² over ten months.
4. **What else is there?** The five closest properties, to check the model against.
5. **Where is it now?** The ward after the 1 July 2025 reorganisation.

## Data

| | |
|---|---|
| Listings | [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates), 3,500,744 listings for sale, June 2025 to March 2026, CC BY-NC 4.0 |
| Used in v0.1 | Hanoi and Ho Chi Minh City: **1,515,980** listings after cleaning |
| Administrative units | Old units from [daohoangson/dvhcvn](https://github.com/daohoangson/dvhcvn), old → new wards from [dvhcvn/20250701](https://github.com/dvhcvn/20250701) |

Almost a third of the raw listings are reposts of the same property. They are removed before
anything else, otherwise the same house ends up in both the training and the test data.
The full cleaning funnel, bounds and column list are in [data/DATASET.md](data/DATASET.md).

## How it works

```mermaid
flowchart LR
    A[3.5M raw listings] --> B[Clean<br/>drop reposts and impossible prices]
    B --> C[Read the text<br/>floors, widths, legal papers, 'bán gấp']
    C --> D[Map to post-2025 wards]
    D --> E[LightGBM<br/>log price per m²]
    E --> F[Fair range<br/>quantile models + conformal widening]
    E --> G[Reasons<br/>TreeSHAP]
```

**Reading the listing text.** Floors are missing in 76% of listings, road width in 85%. Sellers
usually write them in the description anyway: *"nhà 5 tầng"*, *"trệt 2 lầu"* (southern
counting: ground floor + 2), *"hẻm xe hơi"*, *"sổ hồng riêng"*. Regular expressions recover
them, checked against the listings where the seller also filled in the field
([reports/text_extraction.md](reports/text_extraction.md)):

| Field | Agrees with seller-entered value | Missing before → after |
|---|---:|---:|
| Floors | 88.9% | 76.4% → 40.7% |
| Bedrooms | 87.8% | 39.8% → 33.3% |
| Frontage width | 94.9% (±0.5 m) | 51.8% → 49.4% |
| Road / alley width | 88.7% (±0.5 m) | 84.8% → 76.7% |

**Price model.** LightGBM predicts log price per m² from location (district, ward, project,
smoothed ward and district medians), size, building details and the text flags. Price per m²
keeps a 30 m² flat and a 300 m² plot on the same scale; area stays in as a feature.

**Fair range.** Two extra LightGBM models predict the 10th and 90th percentile. Quantile models
are usually too confident on a later month, so the range is widened on the February validation
data until it covers 80% of asking prices there (conformalized quantile regression).

**Reasons.** TreeSHAP values from LightGBM itself, grouped into things a buyer recognises
(location, size, floors, car access, legal papers...). Each one reads as *"this makes the price
per m² X% higher than an average listing in the data"*.

**Listing history.** Two posts are the same property when district, ward, type and exact area
match, and either the description or the title matches once numbers, punctuation and case are
ignored. Dropping the numbers is what makes it work: a price cut usually edits the number and
nothing else. Links are transitive, so a property whose title changed in one repost and
description in another is still one property.

There is no ground truth for "same property", so the check uses details sellers type into
separate fields ([reports/history.md](reports/history.md)). A repost of the same property should
repeat them; two different properties in the same ward with the same area usually do not:

| Field | Linked reposts agree | Same ward and area, not linked |
|---|---:|---:|
| Floors | 99.1% | 43.2% |
| Bedrooms | 98.5% | 55.4% |
| Frontage | 96.7% | 35.5% |

One known failure: agents describe several units of one project with the same text, and those
get merged. Their prices then jump back and forth (4.2, 5.7, 4.2, 5.7 billion), which a single
property almost never does. That happens for 6.5% of repriced properties; the app says so
instead of showing a fake history.

## Results

Trained on June 2025 to January 2026, tested on **March 2026** (160,891 listings the model never
saw), the same situation as checking a new listing. Errors are relative to the asking price.
Full tables in [reports/model.md](reports/model.md).

| Model | Median error | Within 10% | Within 20% |
|---|---:|---:|---:|
| Median price per m² of the district | 23.2% | 23.1% | 43.9% |
| Median price per m² of the ward | 20.1% | 26.2% | 49.8% |
| **LightGBM** | **11.6%** | **44.7%** | **70.1%** |

| | Hanoi | Ho Chi Minh City |
|---|---:|---:|
| Apartments | 6.4% | 7.6% |
| Houses | 14.6% | 13.5% |
| Land | 16.5% | 15.6% |

The fair range contains the asking price for 77.5% of March listings (80% targeted).
Apartments are the easiest to price, they are similar to each other and the project name
carries a lot of information. Land and houses depend on details listings rarely state:
the exact alley, the shape of the plot, the condition of the building.

One finding from [reports/checks.md](reports/checks.md): listings that say *"bán gấp"* or
*"cắt lỗ"* are only marginally cheaper than similar listings (−0.5% vs −0.1% for the rest).
The words alone are not a reason to expect a discount.

## Run it

```sh
git clone https://github.com/HongTin2104/soi-gia-nha
cd soi-gia-nha
pip install -e ".[app]"
streamlit run app/app.py
```

The trained model (~80 MB) and the property data for the history and market sections (~70 MB)
are fetched from the [releases](https://github.com/HongTin2104/soi-gia-nha/releases) on first
start. The **Lấy một tin có thật để thử** button fills the form with a real listing that has a
price history. With Docker:

```sh
docker build -t soi-gia-nha .
docker run -p 7860:7860 soi-gia-nha   # then open http://localhost:7860
```

From Python:

```python
from soi_gia_nha.assets import ensure_model
from soi_gia_nha.explain import check, summary_vi
from soi_gia_nha.listing import build
from soi_gia_nha.model import PriceModel

model = PriceModel.load(ensure_model())
listing = build(
    "Bán nhà 4 tầng hẻm xe hơi 60m2 giá 8,5 tỷ",
    province="Hồ Chí Minh",
    district="Gò Vấp",
    property_type="Nhà",
)
print(summary_vi(check(model, listing)))
```

```
Giá rao nằm trong khoảng hợp lý. Khoảng giá hợp lý: 7,02 tỷ – 9,58 tỷ, mức giữa 7,91 tỷ.
Đẩy giá lên: số tầng (+10%), thời điểm đăng (+6%), số phòng (+5%).
Kéo giá xuống: vị trí (quận, phường, dự án) (-8%), nhà trong ngõ, hẻm (-3%).
```

### Reproduce everything

About half an hour and 16 GB of RAM on a laptop, no GPU needed.

```sh
pip install -e ".[dev]"
python scripts/download.py      # 1.7 GB of parquet into data/raw/
python scripts/prepare.py       # clean, read text, map wards -> data/processed/
python scripts/eval_text.py     # reports/text_extraction.md
python scripts/train.py         # models/v0.1/ and reports/model.md
python scripts/eval_checks.py   # reports/checks.md
python scripts/build_history.py # link reposts, reports/history.md
python scripts/build_app_data.py  # data/app/ for the demo
pytest
```

`notebooks/01_eda.ipynb` is where the cleaning rules come from.

## Project layout

```
src/soi_gia_nha/
  data.py      download and load the raw shards (pinned dataset revision)
  clean.py     filters and repost removal
  text.py      floors, widths and flags from the listing text
  admin.py     old address -> post-2025 ward
  model.py     LightGBM models, fair range, save/load
  explain.py   verdict, TreeSHAP reasons, Vietnamese summary
  listing.py   turn a pasted listing into a model row
  history.py   link reposts of one property, price steps, match a pasted listing
  market.py    monthly ward trends, similar properties
  ui.py        HTML for the demo
  assets.py    fetch the trained model and app data from the releases
app/app.py     Streamlit demo
scripts/       pipeline steps, each writes its own report
reports/       numbers quoted in this README
```

## Limitations

- **Asking prices, not sale prices.** The tool compares a listing with other listings. It is
  not a valuation, and the real transaction price is usually lower.
- Only Hanoi and Ho Chi Minh City, and only what was listed between June 2025 and March 2026.
  The market moves; the model needs retraining to stay current.
- A wide range is an honest answer: without the ward, project or floors the model knows less.
- Reasons are relative to an average listing in the data, not to the neighbouring houses.
- 39% of listings sit in an old ward that was split; the app then shows every candidate new ward.
- Text extraction is rule-based. It misses unusual phrasing and can misread numbers.
- The history only covers June 2025 to March 2026. A listing pasted today has a history only if
  the same property was listed in that period with a matching text.
- Reposts with a rewritten title **and** a rewritten description are not linked.

Ideas for later: other cities, rental listings, a learned extractor for the description text,
alerts for new listings below the fair range.

## Dành cho sinh viên

Repo này có thể dùng làm tài liệu tham khảo cho đồ án môn Học máy, Khai phá dữ liệu hay Xử lý
ngôn ngữ tự nhiên. Một vài điểm nên xem:

- **Chia dữ liệu theo thời gian, không chia ngẫu nhiên.** Tin đăng lại (chiếm gần 1/3 dữ liệu)
  mà nằm ở cả tập train lẫn tập test thì kết quả sẽ đẹp giả. Xem `clean.py` và `model.time_split`.
- **Luôn so với baseline.** Trung vị giá/m² theo phường đã sai số 20%; mô hình chỉ có giá trị
  khi tốt hơn rõ rệt so với cách đơn giản đó.
- **Khoảng dự đoán phải được kiểm tra.** Khoảng 10–90% của mô hình quantile ban đầu chỉ đúng
  70% trên tháng mới; hiệu chỉnh conformal đưa lên 77,5%.
- **Giải thích bằng SHAP** không cần thư viện `shap`: LightGBM có sẵn `pred_contrib=True`.
- **Kiểm tra một thứ không có nhãn.** Không có dữ liệu nào cho biết hai tin có phải cùng một
  căn hay không. Cách làm ở `scripts/build_history.py`: so các trường người bán tự điền (số
  tầng, phòng ngủ) giữa các tin được nối, và so với một nhóm đối chứng (cùng phường, cùng diện
  tích nhưng không được nối). 99% so với 43% là bằng chứng đủ mạnh.
- Mỗi con số trong README đều do một script trong `scripts/` sinh ra; chạy lại sẽ ra cùng kết quả.

Nếu dùng lại code hoặc kết quả trong báo cáo, hãy trích dẫn repo này và bộ dữ liệu của TiniX AI.

## Citation

```bibtex
@software{soi_gia_nha,
  author = {HongTin2104},
  title  = {soi-gia-nha: explainable price checks for Vietnamese real estate listings},
  year   = {2026},
  url    = {https://github.com/HongTin2104/soi-gia-nha}
}
```

## License

Code: [MIT](LICENSE). The listings and the trained model are derived from the TiniX AI dataset
and are **CC BY-NC 4.0**: non-commercial use only, with credit to
[TiniX AI](https://tinix.ai/).
