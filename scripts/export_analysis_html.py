"""Execute analysis cells in a fresh Python process and export a readable HTML report.

This is a dependency-light fallback for environments without nbconvert. It uses
the same code cells as the notebook and records the execution mode in the HTML.
"""

from __future__ import annotations

import contextlib
import html
import io
import json
import sys
from pathlib import Path

import plotly.io as pio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NOTEBOOK_PATH = ROOT / "analysis" / "notebooks" / "insurex_analysis.ipynb"
OUTPUT_PATH = ROOT / "analysis" / "notebooks" / "insurex_analysis.html"


def run_notebook_cells() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "__analysis_notebook__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compiled = compile(source, f"notebook-cell-{index}", "exec")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compiled, namespace)
    return namespace


def table(value) -> str:
    return value.to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:,.6f}")


def figure(value, include_plotlyjs: str | bool) -> str:
    return pio.to_html(value, full_html=False, include_plotlyjs=include_plotlyjs)


def main() -> None:
    ns = run_notebook_cells()
    sections = []
    sections.append("<h1>InsureX Data Analysis</h1>")
    sections.append("<p class='status'>Generated from the notebook code cells in a fresh Python process. nbclient/nbconvert were not available in this environment; this export is a fallback and is not a Jupyter kernel execution record.</p>")
    sections.append("<h2>วัตถุประสงค์และขอบเขต</h2><p>วิเคราะห์ campaign/customer dataset ระดับหนึ่งแถวต่อหนึ่ง record โดยแยก response label และ financial features ออกจาก premium/payment เพราะไม่มี policy ledger</p>")

    sections.append("<h2>Data contract และ field count</h2>" + table(ns["data_contract"]))
    sections.append("<h2>Dictionary fields และชนิดข้อมูล</h2>" + table(ns["dictionary_view"]) + table(ns["dtype_contract"]))
    sections.append("<h2>คุณภาพข้อมูล</h2>" + table(ns["quality_summary"]) + table(ns["missing_by_field"]) + figure(ns["fig_missing"], "inline"))
    sections.append("<h2>Response และ acceptance rate</h2>" + table(ns["overall_metrics"]) + table(ns["overall_response"]) + figure(ns["fig_response"], False))
    sections.append("<h2>Acceptance ตามกลุ่ม</h2>" + table(ns["segment_acceptance"]) + table(ns["occupation_acceptance"].head(15)) + table(ns["month_acceptance"]) + figure(ns["fig_segment"], False) + figure(ns["fig_month"], False))
    sections.append("<h2>Income bands</h2><p>ช่วง income แยก missing, negative, zero และ positive bands ที่ไม่ซ้อนกัน ค่าเหล่านี้เป็น financial features ของข้อมูล campaign/customer</p>" + table(ns["income_band_summary"]) + figure(ns["fig_income"], False))
    sections.append("<h2>Numeric features และ quality flags</h2>" + table(ns["numeric_summary"]) + table(ns["negative_spend"]))
    sections.append("<h2>คู่คอลัมน์ 15/30 วัน</h2><p>แยก equal non-null, shared missing และ equal เมื่อรวม shared missing เพื่อไม่ใช้คำอธิบายเดียวแทนคนละ metric</p>" + table(ns["pair_quality"]))
    sections.append("<h2>Sensitivity และข้อจำกัด</h2>" + table(ns["sensitivity"]) + "<ul><li>ข้อมูลไม่มี customer/policy/agent/payment transaction identifiers</li><li>ไม่สามารถคำนวณ premium หรือ agent KPI จริงจาก CSV นี้</li><li>campaign month ไม่มีปี จึงไม่ใช่ multi-year trend</li><li>ผลเป็น descriptive association ไม่ใช่ causal effect</li><li>dcspend_last_30d มีค่าติดลบ 17 แถว ต้องยืนยันกับ data owner</li></ul>")
    sections.append("<h2>Findings</h2><ol><li>Baseline มี 215,993 rows และ 26 original fields; income_band เป็น derived field</li><li>Acceptance rate 1.2139% จาก accepted 2,622 / valid-label 215,993</li><li>มี 5,535 แถวที่มี missing อย่างน้อยหนึ่ง original field และ exact duplicate excess 2,173 แถว</li><li>คู่ 15/30 วันเท่ากันแบบ non-null 213,592 แถว และ shared missing 2,401 แถว; รวมเป็น 215,993 เมื่อใช้ quality view</li></ol>")
    document = "<!doctype html><html lang='th'><head><meta charset='utf-8'><title>InsureX Data Analysis</title><style>body{font-family:Segoe UI,Tahoma,sans-serif;max-width:1200px;margin:32px auto;line-height:1.55;color:#1f2937}h1,h2{color:#0d5c63}.status{background:#fff4cc;border:1px solid #d6b656;padding:10px}.data-table{border-collapse:collapse;margin:12px 0 24px;font-size:13px}.data-table th,.data-table td{border:1px solid #cbd5e1;padding:5px 8px}.data-table th{background:#e2e8f0}.plotly-graph-div{margin:16px 0}</style></head><body>" + "".join(sections) + "</body></html>"
    OUTPUT_PATH.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "bytes": OUTPUT_PATH.stat().st_size, "mode": "fresh_python_process_fallback"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
