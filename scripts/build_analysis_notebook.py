"""Build, execute, and export the reproducible Thai analysis notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis" / "notebooks" / "insurex_analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    markdown(
        """# Data Analysis

## Explore and describe key information from the datasets

วิเคราะห์ข้อมูล campaign/customer ที่ได้รับเพื่อหาคุณภาพข้อมูล รูปแบบการตอบรับข้อเสนอ และความแตกต่างระหว่างกลุ่มลูกค้า โดยรักษา grain เป็นหนึ่งแถวจาก CSV หนึ่งแถว

คำตอบใน Notebook นี้เป็น descriptive analysis ของ response label และ financial features จากข้อมูลที่มีอยู่ ไม่ใช่ยอดขาย กรมธรรม์ เบี้ยประกัน การชำระเงิน หรือ KPI ตัวแทนจริง
"""
    ),
    markdown(
        """## แหล่งข้อมูลและคำถาม

- CSV `analysis/data/raw/dsc_test_case.csv` เป็น campaign/customer dataset
- Excel `analysis/data/raw/data definition.xlsx` เป็น data dictionary ของ 26 original fields
- `income_band` เป็น derived field ที่สร้างหลังตรวจ raw field count แล้ว และไม่ถูกนับเป็น original field

คำถามหลักคือ:

1. ข้อมูลมี missingness, duplicate และชนิดข้อมูลอย่างไร
2. acceptance rate ของ label 1/2 เมื่อเทียบกับ valid labels 0/1/2 เป็นเท่าไรในภาพรวมและแต่ละกลุ่ม
3. financial features และ income bands มีรูปแบบหรือข้อผิดปกติอะไร
4. คู่คอลัมน์ 15/30 วันเท่ากันจริงเพียงใดเมื่อแยก non-null equality กับ shared missingness
5. ผลจะเปลี่ยนอย่างไรใน sensitivity view ที่ตัด exact duplicate excess ออก
"""
    ),
    code(
        """from pathlib import Path
import sys
import json
import pandas as pd
import plotly.express as px
try:
    from IPython.display import display, HTML
except ImportError:
    def display(value):
        print(value)
    HTML = None

ROOT = Path.cwd()
for candidate in (ROOT, *ROOT.parents):
    if (candidate / "scripts" / "profile_data.py").exists() and (candidate / "analysis" / "data" / "raw").exists():
        ROOT = candidate
        break
else:
    raise FileNotFoundError("Cannot locate the repository root from the notebook working directory")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.profile_data import income_band

def show_plot(figure):
    # Embed Plotly once so the exported HTML works without internet access.
    global _plotlyjs_embedded, _plotly_chart_number
    if HTML is None:
        print(figure)
    else:
        display(HTML(figure.to_html(
            full_html=False,
            include_plotlyjs=True if not _plotlyjs_embedded else False,
            div_id=f"insurex-chart-{_plotly_chart_number:02d}",
        )))
        _plotlyjs_embedded = True
        _plotly_chart_number += 1
_plotlyjs_embedded = False
_plotly_chart_number = 1
RAW = ROOT / "analysis" / "data" / "raw"
CSV_PATH = RAW / "dsc_test_case.csv"
DICTIONARY_PATH = RAW / "data definition.xlsx"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
raw_columns = df.columns.tolist()
raw_field_count = len(raw_columns)

dictionary = pd.read_excel(DICTIONARY_PATH, sheet_name="Data Definition", header=1)
dictionary = dictionary[["Field", "Data Type", "Definition"]].dropna(subset=["Field"])
dictionary_fields = dictionary["Field"].astype(str).tolist()

assert raw_field_count == 26
assert set(raw_columns) == set(dictionary_fields)

derived_fields = ["income_band"]
df["income_band"] = df["income"].map(income_band)
field_count_after_derived = df.shape[1]

data_contract = pd.DataFrame([
    {"metric": "raw_field_count", "value": raw_field_count, "meaning": "CSV fields before derived fields"},
    {"metric": "derived_field_count", "value": len(derived_fields), "meaning": "Notebook-only derived fields"},
    {"metric": "analysis_field_count", "value": field_count_after_derived, "meaning": "raw plus derived fields"},
    {"metric": "row_count", "value": len(df), "meaning": "one supplied campaign record per row"},
])
display(data_contract)
"""
    ),
    markdown(
        """## วิธีตรวจ dictionary และชนิดข้อมูล

ตรวจว่าชื่อ field ใน CSV ตรงกับ Excel ครบทุก field ก่อนวิเคราะห์ จากนั้นใช้ชนิดข้อมูลตาม dictionary เป็นหลัก: string เป็น categorical/text, integer เป็นจำนวนเต็ม และ decimal/double เป็น numeric ที่มี missing ได้ การแปลงชนิดข้อมูลใน Notebook เป็นการคำนวณในหน่วยความจำ ไม่แก้ raw CSV
"""
    ),
    code(
        """dictionary_view = dictionary.rename(columns={"Field": "field", "Data Type": "dictionary_type", "Definition": "definition"})
display(dictionary_view)

dtype_contract = pd.DataFrame({
    "field": raw_columns,
    "raw_dtype": [str(df[field].dtype) for field in raw_columns],
})
display(dtype_contract)
"""
    ),
    markdown(
        """## 1) คุณภาพข้อมูล: missingness และ duplicate

นับ missing เป็นราย original field และแถวที่มี missing อย่างน้อยหนึ่ง original field การนับนี้ไม่รวม `income_band` เพราะเป็น derived field ส่วน duplicate ใช้ exact duplicate excess เป็น sensitivity view เท่านั้น ไม่ลบแถวจาก baseline โดยอัตโนมัติ
"""
    ),
    code(
        """missing_by_field = df[raw_columns].isna().sum().sort_values(ascending=False).rename("missing_rows").reset_index()
missing_by_field.columns = ["field", "missing_rows"]
rows_with_any_missing = int(df[raw_columns].isna().any(axis=1).sum())
exact_duplicate_excess = int(len(df) - len(df[raw_columns].drop_duplicates()))

quality_summary = pd.DataFrame([
    {"metric": "rows", "value": len(df)},
    {"metric": "raw_fields", "value": raw_field_count},
    {"metric": "rows_with_any_original_field_missing", "value": rows_with_any_missing},
    {"metric": "exact_duplicate_excess", "value": exact_duplicate_excess},
])
display(quality_summary)
display(missing_by_field.head(15))

fig_missing = px.bar(
    missing_by_field.head(15).sort_values("missing_rows"),
    x="missing_rows", y="field", orientation="h",
    title="Missing rows by original field",
    labels={"missing_rows": "จำนวนแถวที่ missing", "field": "Original field"},
)
show_plot(fig_missing)
"""
    ),
    markdown(
        """### ผลที่พบจากข้อมูลจริง

มี 215,993 แถวและ 26 original fields ครบตาม dictionary มี 5,535 แถวที่มี missing อย่างน้อยหนึ่ง original field และมี exact duplicate excess 2,173 แถว การวิเคราะห์หลักจึงคงทุกแถวไว้ และแสดง deduplicated result เป็น sensitivity เท่านั้น
"""
    ),
    markdown(
        """## 2) Response label และ acceptance rate

กำหนด label 0 = Reject Offer, 1 = Accept PA และ 2 = Accept Life ตัวเศษของ acceptance rate คือ label 1 หรือ 2 ตัวส่วนคือแถวที่ label อยู่ใน `{0, 1, 2}` เท่านั้น

Filter ของเดือน segment และ occupation ใช้กรอง base population ส่วน response-label filter ใช้ดูรายละเอียด response และไม่ทำให้ denominator ของ acceptance rate เหลือเฉพาะ label ที่เลือก
"""
    ),
    code(
        """LABELS = {0: "Reject Offer", 1: "Accept PA", 2: "Accept Life"}
valid_label = df["label"].isin(LABELS)
accepted = df["label"].isin([1, 2])

overall_response = pd.DataFrame([
    {"label": label, "response": name, "rows": int((df["label"] == label).sum()), "share_of_all": float((df["label"] == label).mean())}
    for label, name in LABELS.items()
])
overall_metrics = pd.DataFrame([{
    "records": len(df),
    "valid_label_records": int(valid_label.sum()),
    "accepted_records": int((valid_label & accepted).sum()),
    "acceptance_rate": float((valid_label & accepted).sum() / valid_label.sum()),
}])
display(overall_metrics)
display(overall_response)

fig_response = px.bar(
    overall_response, x="response", y="rows", color="response",
    title="Campaign response labels",
    labels={"rows": "จำนวนแถว", "response": "Response"},
)
show_plot(fig_response)
"""
    ),
    code(
        """def acceptance_table(frame: pd.DataFrame, group_field: str) -> pd.DataFrame:
    grouped = frame.groupby(group_field, dropna=False)
    out = grouped["label"].agg(
        records="size",
        valid_label_records=lambda s: int(s.isin([0, 1, 2]).sum()),
        accepted_records=lambda s: int(s.isin([1, 2]).sum()),
    ).reset_index()
    out["group"] = out[group_field].fillna("Unknown")
    out["acceptance_rate"] = out["accepted_records"] / out["valid_label_records"].where(out["valid_label_records"] > 0)
    return out.sort_values(["acceptance_rate", "valid_label_records"], ascending=[False, False])

segment_acceptance = acceptance_table(df, "customer_segment")
occupation_acceptance = acceptance_table(df, "main_occupation")
month_acceptance = acceptance_table(df, "campaign_month")

display(segment_acceptance)
display(occupation_acceptance.head(15))
display(month_acceptance)

fig_segment = px.bar(
    segment_acceptance.sort_values("acceptance_rate"), x="acceptance_rate", y="group", orientation="h",
    title="Acceptance rate by customer segment",
    labels={"acceptance_rate": "Acceptance rate", "group": "Customer segment"},
)
fig_month = px.bar(
    month_acceptance.sort_values("acceptance_rate"), x="group", y="acceptance_rate",
    title="Acceptance rate by campaign month label",
    labels={"acceptance_rate": "Acceptance rate", "group": "Campaign month"},
)
show_plot(fig_segment)
show_plot(fig_month)
"""
    ),
    markdown(
        """### Findings และข้อควรระวังของ response analysis

จาก baseline: accepted records มี 2,622 จาก valid-label records 215,993 หรือ 1.2139%; Reject 213,371, PA 1,634 และ Life 988

Lower Mass มี acceptance rate สูงสุดในกลุ่ม segment ที่มีจำนวนมาก (1.2620%) ส่วน occupation ที่สูงสุดคือ Entertainer (2.0492%) แต่มีเพียง 244 valid-label records จึงต้องระวัง sample size ก่อนนำไปใช้ตัดสินใจ แคมเปญเดือน Aug สูงสุดที่ 1.9537% แต่ข้อมูลมีเพียงชื่อเดือน ไม่มีปี จึงไม่ใช่แนวโน้มหลายปีและไม่ควรใช้ภาษาเชิงเหตุผลหรือ causal
"""
    ),
    markdown(
        """## 3) Income bands และ financial feature distributions

`income_band` แยกเป็น missing, negative, zero และ positive intervals ที่ไม่ซ้อนกัน: `(0, 15,000]`, `(15,000, 30,000]`, `(30,000, 50,000]`, `(50,000, 100,000]` และ `>100,000` ค่าการเงินทั้งหมดเป็น financial features จาก campaign/customer dataset ไม่ใช่ premium หรือ payment
"""
    ),
    code(
        """income_band_summary = acceptance_table(df, "income_band")[["group", "records", "valid_label_records", "accepted_records", "acceptance_rate"]]
display(income_band_summary)

fig_income = px.bar(
    income_band_summary.sort_values("records"), x="group", y="records", color="group",
    title="Income band population",
    labels={"records": "จำนวนแถว", "group": "Income band"},
)
show_plot(fig_income)
"""
    ),
    code(
        """numeric_fields = [
    "num_children", "age", "income", "maxosdc_last_30d", "dcspend_last_30d",
    "easypymt_last_30d", "savacc_bal", "currentacc_bal", "avg_savaccbal_30d",
    "avg_currentaccbal_30d", "mob", "inflow30d", "outflow30d", "inflow1_15",
    "outflow1_15", "net_flow_30d", "net_flow_15d",
]
numeric_summary = df[numeric_fields].describe().T.reset_index().rename(columns={"index": "field"})
numeric_summary["missing_rows"] = df[numeric_fields].isna().sum().values
display(numeric_summary)

negative_spend = df.loc[df["dcspend_last_30d"] < 0, ["dcspend_last_30d"]]
display(pd.DataFrame([{"check": "negative_dcspend_last_30d", "rows": len(negative_spend)}]))
"""
    ),
    markdown(
        """พบ `dcspend_last_30d` ติดลบ 17 แถว จึงเก็บเป็น data-quality finding และไม่แทนค่าด้วยศูนย์หรือลบออกโดยอัตโนมัติ ความหมายทางธุรกิจต้องยืนยันกับเจ้าของข้อมูล
"""
    ),
    markdown(
        """## 4) ตรวจคู่คอลัมน์ 15/30 วัน

รายงานสาม metric แยกกัน:

- `equal_non_null`: ทั้งสองค่าไม่ว่างและเท่ากัน
- `shared_missing`: ทั้งสองค่าว่างพร้อมกัน
- `equal_including_shared_missing`: รวมสองกรณีข้างต้นเพื่อใช้เป็น quality view เท่านั้น
"""
    ),
    code(
        """pair_definitions = [
    ("inflow30d", "inflow1_15"),
    ("outflow30d", "outflow1_15"),
    ("net_flow_30d", "net_flow_15d"),
]
pair_rows = []
for left, right in pair_definitions:
    equal_non_null = df[left].notna() & df[right].notna() & (df[left] == df[right])
    shared_missing = df[left].isna() & df[right].isna()
    pair_rows.append({
        "pair": f"{left} = {right}",
        "equal_non_null": int(equal_non_null.sum()),
        "shared_missing": int(shared_missing.sum()),
        "equal_including_shared_missing": int((equal_non_null | shared_missing).sum()),
        "non_null_mismatch": int((df[left].notna() & df[right].notna() & (df[left] != df[right])).sum()),
        "one_sided_missing": int((df[left].notna() ^ df[right].notna()).sum()),
    })
pair_quality = pd.DataFrame(pair_rows)
display(pair_quality)
"""
    ),
    markdown(
        """### ผลที่พบ

ทั้งสามคู่มีค่า equal แบบ non-null 213,592 แถว และ shared missing 2,401 แถว หากรวม shared missing จะเป็น 215,993 แถว จึงต้องไม่ใช้คำอธิบายเดียวแทนสอง metric นี้
"""
    ),
    markdown(
        """## 5) Sensitivity และข้อจำกัด

Baseline เก็บทุก row เพราะยังไม่มี business rule ให้ลบ duplicate การคำนวณ exact-deduplicated เป็น sensitivity view เพื่อดูผลกระทบเท่านั้น

ข้อมูลนี้ไม่มี customer ID, policy ID, agent ID, policy date, premium transaction, payment transaction หรือ contract ledger จึงไม่สามารถคำนวณ agent KPI หรือสรุปยอดขายจริงได้ หากต้องการวิเคราะห์ policy/agent ต้องขอข้อมูลระดับธุรกรรมเพิ่ม

ข้อจำกัดอื่นคือ campaign month ไม่มีปี, occupation อาจไม่ถูก update, missingness ต้องตีความตามเจ้าของข้อมูล และความสัมพันธ์ที่เห็นเป็น descriptive association ไม่ใช่ causal effect
"""
    ),
    code(
        """deduplicated = df[raw_columns].drop_duplicates()
sensitivity = pd.DataFrame([
    {"view": "baseline_all_rows", "records": len(df), "acceptance_rate": float(accepted.sum() / valid_label.sum())},
    {"view": "exact_deduplicated", "records": len(deduplicated), "acceptance_rate": float(deduplicated["label"].isin([1, 2]).sum() / deduplicated["label"].isin([0, 1, 2]).sum())},
])
display(sensitivity)
"""
    ),
    markdown(
        """## สรุป findings และสิ่งที่ควรทำต่อ

1. Acceptance rate จาก supplied campaign data อยู่ที่ 1.2139% และต้องใช้ valid-label population เป็น denominator ร่วมกันทุกมุมมอง
2. Missingness และ duplicate มีผลต่อการอ่านผล จึงแสดง baseline กับ sensitivity แยกกัน
3. Segment/occupation/month ใช้เป็น descriptive comparison เท่านั้น โดยเฉพาะกลุ่มขนาดเล็กและเดือนที่ไม่มีปี
4. คู่ 15/30 วันต้องรายงาน equal non-null และ shared missing แยกกัน
5. ต้องตรวจความหมายของ negative debit-card spending และความซ้ำของ 15/30-day fields กับ data owner
6. ต้องขอ policy/agent/transaction data เพิ่ม หากต้องการ premium, policy count หรือ agent KPI จริง
"""
    ),
]

# Keep notebook cells addressable for nbformat/nbclient and future diffs.
for index, cell in enumerate(cells):
    cell["id"] = f"insurex-analysis-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

loaded = nbformat.read(NOTEBOOK, as_version=4)
executed = NotebookClient(
    loaded,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute(cwd=str(ROOT))
errors = [
    output
    for cell in executed.cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
if errors:
    raise RuntimeError(f"notebook execution produced {len(errors)} error output(s)")
nbformat.write(executed, NOTEBOOK)

print({
    "notebook": str(NOTEBOOK),
    "cells": len(cells),
    "markdown_cells": sum(c["cell_type"] == "markdown" for c in cells),
    "code_cells": sum(c["cell_type"] == "code" for c in cells),
    "execution_errors": len(errors),
})
