# Relational model — Test Case #1 KPI

ไฟล์นี้อธิบาย Data Modeling/KPI.

```mermaid
erDiagram
    agents ||--o{ monthly_agent_performance : records
    monthly_agent_performance ||--o{ monthly_kpi_evaluations : evaluated_as
    kpi_rules ||--o{ monthly_kpi_evaluations : applies_to
    agents ||--o{ contract_history : changes
    monthly_kpi_evaluations ||--o{ contract_history : triggers

    agents {
        TEXT agent_id PK
        TEXT agent_name
        TEXT joined_on
        TEXT inactive_on
        TEXT initial_contract_type
    }
    kpi_rules {
        TEXT rule_id PK
        TEXT valid_from_month
        TEXT valid_to_month
        INTEGER premium_threshold_satang
        INTEGER policy_threshold
        INTEGER required_streak
        TEXT comparison_operator
    }
    monthly_agent_performance {
        INTEGER performance_id PK
        TEXT agent_id FK
        TEXT month_start
        INTEGER total_premium_satang
        INTEGER new_policy_count
        TEXT data_status
        TEXT source_kind
        INTEGER month_closed
        INTEGER input_revision
        TEXT input_hash
    }
    monthly_kpi_evaluations {
        INTEGER evaluation_id PK
        INTEGER performance_id FK
        TEXT rule_id FK
        INTEGER evaluation_revision
        TEXT result
        INTEGER pass_streak
        INTEGER fail_streak
        INTEGER performance_input_revision
        INTEGER snapshot_total_premium_satang
        INTEGER snapshot_new_policy_count
        INTEGER is_current
    }
    contract_history {
        INTEGER contract_event_id PK
        TEXT agent_id FK
        TEXT from_type
        TEXT to_type
        TEXT effective_from
        INTEGER trigger_evaluation_id FK
        TEXT reason
        TEXT superseded_at
    }
```

Important requirements:

- หนึ่งผลการขายต่อ `(agent_id, month_start)` และเก็บ revision/hash สำหรับ audit
- เงินใช้ integer satang เพื่อไม่ให้ floating-point ทำให้เกณฑ์ `> 15,000 บาท` ผิด
- ผลประเมินปัจจุบันมีได้หนึ่งแถวต่อ performance; revision เดิมยังอยู่
- Contract event เป็นประวัติแบบมี effective date และอ้าง evaluation ที่ทำให้เปลี่ยนสัญญา
- สัญญาใหม่มีผลวันแรกของเดือนถัดจากเดือนที่ streak ครบ 3 เดือน
- `PENDING`, เดือนที่ยังไม่ปิด และเดือนปฏิทินที่ขาดช่วงจะไม่ถูกนำไปต่อ streak
- ข้อมูลตัวอย่าง KPI เป็น synthetic
