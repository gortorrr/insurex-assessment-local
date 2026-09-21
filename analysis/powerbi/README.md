# Power BI Dashboard

`InsureX_Analysis.pbip` เป็นรายงานหน้าเดียวสำหรับสำรวจและสรุปข้อมูลจาก `analysis/data/raw/dsc_test_case.csv` โดยอ้างอิงความหมายของ 26 คอลัมน์จาก `analysis/data/raw/data definition.xlsx`

หน้า `Dashboard` ประกอบด้วย:

- Filter `campaign_month`, `main_occupation` และ `customer_segment`
- KPI: Total records, Accepted records, Rows with missing values และ Offer acceptance rate
- จำนวนผู้ตอบรับ offer แยกตาม customer segment, occupation และ campaign month
- การกระจาย income band และ campaign response
- ตารางรายละเอียดลูกค้า

Key definitions:

- Accepted records คือ `label` 1 (PA Insurance) หรือ 2 (Life Insurance)
- Offer acceptance rate ใช้แถวที่ `label` อยู่ใน `{0,1,2}` เป็น denominator
- Rows with missing values นับแถวที่มีค่าว่างอย่างน้อยหนึ่งช่องจาก 26 original fields
- `campaign_month_order`, `response_label`, `income_band` และ `Has Missing Original Field` เป็น derived fields

## วิธีเปิด

1. เปิด `InsureX_Analysis.pbip` ด้วย Power BI Desktop
2. ตรวจ parameter `DataRoot` ให้ชี้ไปที่ root ของ repository (Home → Transform data → Manage Parameters เปลี่ยน Current Value เป็น root ของ repository)
3. กด **Home > Refresh** และบันทึกโปรเจกต์
