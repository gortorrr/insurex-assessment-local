# Data Modeling

ส่วนนี้เป็น Test Case #1 ด้าน schema และกฎ KPI ของตัวแทนขาย

- `schema.sql` เป็น SQLite schema หลักสำหรับ demo
- `db/analytics.sqlite` เป็น database local ของ agent, monthly performance, evaluation และ contract events (mock data)
- `queries/monthly_kpi.sql` เป็นรายงานรายเดือนที่แสดงยอดขาย ผลประเมิน streak และการเปลี่ยนสัญญา
- `fixtures/kpi_seed_summary.json` เป็นหลักฐาน synthetic ที่ระบุแหล่งข้อมูลชัดเจน
- `erd.md` อธิบาย relation และ assumptions

## Validation rule

ผลรายเดือนเป็น `PASS` เมื่อเงื่อนไขทั้งสองข้อเป็นจริง:

- `total_premium_satang > 1,500,000` หรือเบี้ยรวมมากกว่า 15,000 บาท
- `new_policy_count > 5`

เมื่อ `FAIL` ติดต่อกัน 3 เดือน สัญญาจะเปลี่ยนเป็น `commission` และเมื่อ `PASS` ติดต่อกัน 3 เดือน สัญญาจะเปลี่ยนเป็น `salary`. เดือนที่ข้อมูล `PENDING`, เดือนที่ยังไม่ปิด หรือเดือนปฏิทินที่ขาดช่วงจะเริ่มนับ streak ใหม่

## Assumptions

- การเปลี่ยนสัญญามีผลวันแรกของเดือนถัดจากเดือนที่ streak ครบ 3 เดือน
- เงินจัดเก็บเป็นจำนวนเต็มหน่วยสตางค์เพื่อหลีกเลี่ยงความคลาดเคลื่อนของ floating point
- ข้อมูล demo เป็น synthetic
- `initial_contract_type` เป็นสถานะก่อนเริ่มช่วงประเมิน ส่วนทุกการเปลี่ยนแปลงหลังจากนั้นอยู่ใน `contract_history`