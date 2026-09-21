# InsureX Assessment

เป็น Repo สำหรับส่งงาน assessment

## ขอบเขตข้อมูล

ไฟล์ CSV มี 215,993 แถว x 26 fields และ Excel เป็น data dictionary

## Structure

- `analysis/notebooks/insurex_analysis.ipynb` - การวิเคราะห์เดียวกับ metric contract พร้อม Markdown ตาราง และกราฟ Plotly
- `analysis/shared/metric_contract.json` - นิยาม metric, denominator, null และ filter semantics ที่ใช้ร่วมกัน
- `analysis/powerbi/` - Power BI PBIP report, semantic model, source query และ measures; Desktop refresh/filter acceptance ยังแยกเป็น gate
- `analysis/data/raw/` - raw dataset ของ assessment;
- `analysis/data/derived/`, `analysis/reports/` - profile, aggregate และ analysis report
- `data_modeling/` - ERD, SQLite schema, queries และ synthetic fixtures
- `rag/ui/app.py` - customer/staff workspace: ลูกค้าหนึ่งบัญชีมีหนึ่งบทสนทนาและหนึ่ง Lead; พนักงานเห็นเฉพาะเคสที่ AI/RAG ตอบไม่ได้ รับเคส ตอบ และปิดเคสได้
- `insurex/` - KPI, Lead, Session และ LangGraph/RAG business logic

## Power BI

`analysis/powerbi/InsureX_Analysis.pbip` มี report, semantic model, source query และ measures สำหรับข้อมูล CSV ที่แนบมา โดยกำหนด type ครบทั้ง 26 original fields และสร้าง `Has Missing Original Field` ก่อน derived `income_band` หากย้ายโฟลเดอร์ ให้เปิด Transform data > Manage Parameters แล้วตั้ง `DataRoot` เป็น path ของโฟลเดอร์ส่งงานก่อน Refresh

## Data Modeling/KPI

กฎหลักคือ `total_premium > 15,000 THB AND new policies > 5` ต่อเดือน โดยเก็บจำนวนเงินเป็น Integer หน่วยสตางค์และใช้ Strict `>` ทั้งสองเงื่อนไข หากตัวแทนที่ถือสัญญา Commission ผ่านการประเมิน 3 เดือนปฏิทินติดต่อกัน สัญญาจะเปลี่ยนเป็น Salary-based และหากตัวแทนที่ถือสัญญา Salary ไม่ผ่านการประเมิน 3 เดือนปฏิทินติดต่อกัน สัญญาจะเปลี่ยนเป็น Commission-based

Joined Date, Inactive Date, Incomplete Month, Missing Calendar Month และ Contract Effective Date เป็น Assumptions ที่กำหนดเพิ่มเติมและระบุแยกไว้ในเอกสารกับ Tests โดยเดือนที่ข้อมูลไม่สมบูรณ์หรือยังไม่ปิดจะมีผลเป็น `PENDING` และไม่นำไปต่อ Pass/Fail Streak ส่วนการเปลี่ยนสัญญามีผลตั้งแต่วันแรกของเดือนถัดจากเดือนที่ Streak ครบตามเกณฑ์

## Setup และการเปิด Web App

รันคำสั่งจากโฟลเดอร์หลักของโปรเจกต์:

```powershell
# สร้าง Virtual Environment
python -m venv .venv

# ติดตั้ง Dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt

# สร้างไฟล์ Environment Configuration
Copy-Item .env.example .env
```

แก้ไขไฟล์ `.env` และกำหนดค่าต่อไปนี้สำหรับการใช้ Gemini จริง:

```env
GOOGLE_API_KEY=ใส่_API_KEY
```

สร้างบัญชีตัวอย่างสำหรับทดสอบระบบ Multi-user:

```powershell
.venv\Scripts\python.exe -X utf8 -m scripts.seed_demo_accounts
```

เปิด Web App:

```powershell
.venv\Scripts\python.exe -X utf8 -m streamlit run rag/ui/app.py
```

## คำสั่งเพิ่มเติมสำหรับสร้างข้อมูลและตรวจสอบระบบ

คำสั่งต่อไปนี้ไม่จำเป็นต้องรันก่อนเปิด Web App

```powershell
# สร้าง Vector Database ใหม่จาก PDF
# ใช้เมื่อยังไม่มี ChromaDB หรือมีการเปลี่ยนเอกสาร
.venv\Scripts\python.exe -X utf8 -m scripts.ingest_rag

# ตรวจ Unit และ Integration Tests
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q

# ประเมิน Retrieval แบบ Offline โดยไม่เรียก Gemini
.venv\Scripts\python.exe -X utf8 scripts\evaluate_rag_offline.py

# ทดสอบ Flow และสร้างหลักฐาน Demo
.venv\Scripts\python.exe -X utf8 -m scripts.local_e2e
```

ชุดส่งแบบโฟลเดอร์มี active PDF 5 ฉบับ, corrected extraction, local embedding model และ Chroma index 31 chunks ตาม manifest จึงเปิดตรวจ retrieval ได้โดยไม่ต้องดาวน์โหลดโมเดลเพิ่ม สามารถรัน `python -X utf8 -m scripts.ingest_rag` เพื่อยืนยัน index ซ้ำได้ การ ingest จะ fail closed หาก PDF, corrected extraction, model revision หรือ hash ไม่ตรง manifest

## Architecture และ LangGraph

กราฟมี State ของคำถามปัจจุบัน, Bounded Conversation History, Retrieved Chunks, Evidence Status, Citations, Lead Draft และ Error State โดย `route_intent` แยก Greeting, Thanks, Goodbye, Help, Out-of-scope และ Human Request ออกจากคำถามผลิตภัณฑ์ประกันตั้งแต่ต้น ข้อความทั่วไปไปยัง `direct_response` โดยไม่เรียก Gemini หรือ ChromaDB ส่วนคำถามผลิตภัณฑ์ประกันใช้เส้นทาง `resolve_query → retrieve → assess_evidence → rewrite_query_once/generate_grounded_answer → validate_answer_citations`

การ Rewrite ถูกจำกัดไว้ไม่เกินหนึ่งรอบ และมีเส้นทาง Lead Collection, Ambiguous Product, No-answer และ Service Error แยกกัน UI ไม่บังคับให้เลือกผลิตภัณฑ์ประกันหรือ PDF ก่อนถาม โดยค้นจาก ChromaDB ทั้ง Corpus เมื่อไม่ได้ระบุผลิตภัณฑ์ประกัน และขอให้ผู้ใช้เลือกเมื่อหลักฐานครอบคลุมหลายผลิตภัณฑ์ประกัน

Conversation History ใช้ช่วยแก้คำถามต่อเนื่องเท่านั้น โดยใช้ข้อความล่าสุดไม่เกิน 6 รายการและจำกัดความยาวรวมไม่เกิน 2,400 ตัวอักษร การประเมินหลักฐานยังอ้างอิงคำถามปัจจุบันและ Retrieved Chunks ไม่ถือว่าประวัติการสนทนาเป็นหลักฐานข้อเท็จจริง ข้อความที่ตรวจพบว่าเป็น Lead จะไม่ถูกนำเข้า Product History และข้อมูลโทรศัพท์หรือรายได้จะถูกปกปิดก่อนสร้าง Standalone Query

หน้า Streamlit แยก Role เป็น `customer` และ `staff` ลูกค้าหนึ่งบัญชีมีหนึ่ง Active Session, หนึ่ง LangGraph Thread, หนึ่งบทสนทนา และหนึ่ง Lead/Profile เมื่อคำถามผลิตภัณฑ์จบด้วย Evidence Status เป็น `insufficient` หรือ `error` โดยไม่มี Citation หรือเมื่อลูกค้าขอพนักงานโดยตรง ระบบจะสร้าง `handoff_cases` สถานะ `pending` และพัก AI/RAG ของ Session นั้น

พนักงานทุกคนเห็นเคสที่รอรับ แต่เมื่อพนักงานหนึ่งคนรับเคสแล้ว เคสจะเห็นเฉพาะผู้รับและป้องกันพนักงานคนอื่นตอบซ้อน พนักงานตอบโดยไม่เรียก AI/RAG และกด **แก้ไขปัญหาแล้ว** เพื่อปิดเคส ฝั่งลูกค้าตรวจสถานะทุก 2 วินาทีและกลับไปใช้ AI/RAG อัตโนมัติเมื่อเคสปิด

ข้อความเก็บ `sender_type`, `sender_account_id` และ `sender_label` Lead ที่ยืนยันล่าสุดเกิน 183 วันจะเริ่มรอบเก็บข้อมูลใหม่ โดยเก็บข้อมูลเดิมไว้จนกว่าข้อมูลใหม่จะครบและบันทึกสำเร็จ

คำถามแนะนำแสดง 4 ข้อ โดยสร้างจาก Product History ของบัญชีปัจจุบัน แปลงประวัติเป็นเฉพาะชื่อผลิตภัณฑ์ประกันและหัวข้อที่อนุญาต และตรวจแต่ละคำถามผ่าน Retrieval และ Evidence Gate ของ Corpus ก่อนแสดง โดยไม่ส่ง Lead PII เข้า Prompt

RAG ใช้ E5 Query/Passage Prefixes, Chroma Persistent Collection และ Page-aware Citations ส่วน Lead ใช้ Pydantic Validation บันทึกแต่ละ Slot ลง `lead_drafts` ทันที และ Upsert Lead/Profile เพียงหนึ่งแถวต่อ `owner_id` รายได้เก็บใน `income_thb` หน่วยบาท และ `lead_request_log` ใช้รักษา Request-ID Idempotency

แต่ละบัญชีมีหนึ่ง Active Session/Thread ที่ตรวจ Owner Context ตาราง `messages` ใช้ `purpose` แยกข้อความ Product ออกจากข้อความ Private และ UI ใช้ `metadata_json` เก็บ Citation Metadata ของคำตอบ

## สถานะ

ผลตรวจ: 146 Unit/Integration Tests ผ่าน, Requirement Verifier แบบ Isolated ผ่าน, Retrieval ชุดเดิม 30/30 และ Held-out-v2 20/20 ผ่าน, PDF 5 ฉบับรวม 14 หน้าและมี SHA-256 ตรงตาม Manifest ส่วน ChromaDB มี 31 Chunks

## วิธีสาธิต Lead และ Session

1. เปิด Web App และพิมพ์ `สนใจสมัคร คุ้มตลอดชีพ พลัส` ระบบจะเข้าเส้นทาง Lead Collection และจับคู่ชื่อผลิตภัณฑ์จาก Knowledge Base Manifest

2. ส่งข้อมูลสมมติ เช่น `ชื่อ เอก อาชีพ ครู รายได้ 30000 บาทต่อเดือน เบอร์โทร 0812345678`

3. ระบบสกัดข้อมูลเข้าสู่ Pydantic Model และบันทึกข้อมูลแต่ละช่องลงตาราง `lead_drafts` ใน SQLite ทันที หากข้อมูลยังไม่ครบ ระบบจะถามเฉพาะช่องที่ขาด เมื่อข้อมูลชื่อ อาชีพ รายได้ เบอร์โทร และผลิตภัณฑ์ครบ ระบบจะตรวจสอบข้อมูลและ Upsert Lead/Profile สมบูรณ์โดยอัตโนมัติ

4. เมื่อบัญชีเดิมกลับมาใช้งาน ระบบจะเรียก Active Session และ LangGraph Thread เดิม จึงเรียกคืนประวัติการสนทนาและ Lead/Profile เดิมได้ ตามเงื่อนไข 1 Account : 1 Active Session : 1 Active Thread : 1 Chat : 1 Lead/Profile

5. ออกจากระบบและเข้าสู่ระบบด้วยบัญชีลูกค้าอีกบัญชี เพื่อยืนยันว่าประวัติการสนทนา, LangGraph Thread, Lead Draft และ Lead/Profile ของแต่ละบัญชีแยกจากกัน

## บัญชี Demo สำหรับตรวจ Multi-user

หลังติดตั้งระบบ ให้สร้างบัญชี Synthetic สำหรับทดสอบด้วยคำสั่ง:

```powershell
.venv\Scripts\python.exe -X utf8 -m scripts.seed_demo_accounts
```

คำสั่งนี้สามารถรันซ้ำได้โดยไม่สร้างบัญชีซ้ำ และจะสร้างบัญชีดังต่อไปนี้:

| บัญชี | บทบาท | รหัสผ่าน |
|---|---|---|
| `demo-agent-a` | ลูกค้า | `Demo-Agent-A-2026!` |
| `demo-agent-b` | ลูกค้า | `Demo-Agent-B-2026!` |
| `demo-staff-a` | พนักงานขาย | `Demo-Staff-A-2026!` |
| `demo-staff-b` | พนักงานขาย | `Demo-Staff-B-2026!` |

รหัสผ่านไม่ได้ถูกเก็บเป็นข้อความปกติในฐานข้อมูล โดย SQLite เก็บ Scrypt Hash และ Salt ส่วนรหัสผ่าน Demo ที่แสดงในเอกสารเป็นข้อมูล Synthetic สำหรับการตรวจระบบเท่านั้น

หลักฐานของ Structured Lead Collection และ Session Separation อยู่ที่ `rag/reports/demo/assessment_requirements_current.json` โดยใช้ Temporary SQLite แยกจากฐานข้อมูลของ Web App และตรวจว่าบัญชี A/B มี Session และ LangGraph Thread คนละรายการ ไม่สามารถอ่านข้อมูลข้าม Owner และสามารถบันทึก Lead แยกกันได้

## Persistent Login

หลังเข้าสู่ระบบ Web App จะจำบัญชีใน Browser เป็นเวลา 30 นาทีแม้ Refresh หน้า โดย Cookie เก็บ Opaque Login Token และฐานข้อมูลเก็บเฉพาะ SHA-256 Hash ของ Token ในตาราง `login_sessions` ไม่มีรหัสผ่านอยู่ใน Cookie

เมื่อ Token หมดอายุหรือผู้ใช้กดออกจากระบบ ระบบจะเพิกถอน Login Session และ Token เดิมจะใช้เข้าสู่ระบบไม่ได้ หลักฐาน Browser Test อยู่ที่ `rag/reports/demo/persistent_login_browser.json`

## Human Handoff

Human Handoff เกิดเมื่อลูกค้าขอคุยกับพนักงานโดยตรง หรือเมื่อคำถามเกี่ยวกับผลิตภัณฑ์ไม่สามารถตอบได้จากหลักฐานใน Corpus หลังผ่าน Retrieval และ Evidence Gate แล้ว ระบบจะสร้าง `handoff_cases` สถานะ `pending` และพัก AI/RAG ของ Session นั้น

พนักงานทุกคนเห็นเคสที่รอรับ เมื่อ `demo-staff-a` รับเคสแล้ว เคสจะเห็นเฉพาะพนักงานผู้รับ และ `demo-staff-b` จะไม่สามารถตอบซ้อนได้ เมื่อพนักงานปิดเคส ลูกค้าจะกลับไปใช้ AI/RAG ได้อัตโนมัติ

ไฟล์ `rag/reports/demo/human_handoff_browser.json` เป็นหลักฐานของการรับเคส การตอบ การป้องกันพนักงานตอบซ้อน การปิดเคส และการกลับไปใช้ RAG พร้อม Citation คำถามนอกขอบเขตของผลิตภัณฑ์ประกันจะถูกตอบผ่าน `direct_response` โดยไม่สร้าง Human Handoff ส่วนคำถามเกี่ยวกับผลิตภัณฑ์ที่ผ่าน Retrieval แล้วไม่พบหลักฐานเพียงพอ หรือเกิดข้อผิดพลาดระหว่างประมวลผล จะถูกส่งต่อเข้าสู่ Human Handoff นอกจากนี้ ผู้ใช้สามารถขอคุยกับพนักงานโดยตรงเพื่อสร้าง Human Handoff ได้

## Security, Trace และ Corpus Safety

`insurex/telemetry.py` เก็บ trace แบบจำกัดข้อมูลเพื่อป้องกัน PII โดยแปลง `owner_id` และ `session_id` เป็นค่า hash เก็บเฉพาะชื่อ node, latency, จำนวน request, document/page/chunk ID และ token usage ที่อยู่ใน allowlist ระบบไม่บันทึก prompt, คำตอบ, ข้อมูล Lead หรือข้อความ exception ลงใน trace

`scripts/verify_multiuser_live.py --live` ใช้ Gemini จริงและข้อมูล synthetic ทดสอบผู้ใช้ A/B บน temporary SQLite เดียวกัน โดยผู้ใช้ A สามารถบันทึก Lead ที่ข้อมูลครบได้ ส่วนผู้ใช้ B มี Lead Draft แยกของตนเองและไม่ทำให้ข้อมูลของ A เปลี่ยนแปลง ผลการทดสอบ live อยู่ที่ `rag/reports/demo/multiuser_live.json` ส่วนหลักฐานการปฏิเสธการเข้าถึงข้อมูลข้ามบัญชีและการตรวจ requirements ปัจจุบันอยู่ที่ `rag/reports/demo/assessment_requirements_current.json`

Corpus ingestion ใช้ write lock และ incomplete marker เพื่อป้องกันการอ่าน index ที่สร้างไม่สมบูรณ์ ระหว่างการสร้าง index ระบบตรวจ manifest, hash ของ PDF, corrected extraction, embedding model และ chunk configuration ก่อนบันทึก fingerprint ลง Chroma เมื่อเปิดใช้งาน retrieval ระบบจะตรวจ fingerprint และรายการ chunk ของ index กับ corpus ปัจจุบัน หากไม่ตรงกันจะหยุดทำงานแทนการตอบจาก index เก่า

Retrieval ใช้ E5 prefixes โดยกำหนด `query:` สำหรับคำค้นและ `passage:` สำหรับเนื้อหาเอกสาร ค่าเริ่มต้นเป็น hybrid retrieval ซึ่งรวม semantic retrieval กับ lexical matching เพื่อช่วยค้นคำเฉพาะจากตารางใน PDF เช่น waiting period สามารถเปลี่ยนเป็น semantic-only ได้ด้วย:

```env
RETRIEVAL_MODE=semantic
```

ปุ่ม **ตัวอย่าง Log** ข้างเมนูจุดสามจุดแสดงสถานะการทำงานของ LangGraph แบบ near real time เช่น intent routing, Chroma retrieval, evidence assessment, answer generation, citation validation, structured Lead, checkpoint และ Human Handoff

Log เก็บเฉพาะข้อมูลที่จำเป็น เช่น:

- ชื่อขั้นตอน
- ระยะเวลาทำงาน
- จำนวน chunk
- document ID
- page number
- chunk ID
- ชื่อช่อง Lead ที่ยังขาด
- Lead ID
- ประเภทข้อผิดพลาด

Log ไม่เก็บคำถาม คำตอบ ชื่อ เบอร์โทรศัพท์ รายได้ หรือค่าจริงของ Lead slot หลักฐานการตรวจ requirements ล่าสุดอยู่ที่ `rag/reports/demo/assessment_requirements_current.json` และผลทดสอบ Gemini ที่ตอบคำถามจาก PDF ครบทั้ง 5 ฉบับอยู่ที่ `rag/reports/demo/live_five_pdf_answers.json`

## Grounded Question Suggestions

ระบบคำถามแนะนำใช้ประวัติที่จัดเก็บของบัญชีนั้นเมื่อเริ่มสร้าง suggestion โดยให้ Gemini สร้างคำถามตัวเลือกสูงสุด 8 ข้อ จากนั้นค้น Chroma/PDF และใช้ evidence gate คัดเฉพาะคำถามที่มีหลักฐาน ตัดคำถามซ้ำ และเลือกแสดง 4 ข้อ

คำถามจะถูก cache ตามบัญชีและสถานะ corpus เพื่อไม่ให้การสร้างคำถามใหม่ทุกครั้งรบกวนช่องส่งข้อความ ดังนั้นคำถามแนะนำอาจไม่เปลี่ยนทันทีหลังส่งข้อความล่าสุด ระบบจะสร้างใหม่เมื่อเริ่ม UI session ใหม่ หรือเมื่อ corpus หรือ retrieval configuration เปลี่ยน

หากกระบวนการสร้างหรือคัดคำถามเกิดข้อผิดพลาด ระบบปัจจุบันจะแสดง `DEFAULT_QUESTIONS` เป็น fallback โดยใช้ label `คำถามเริ่มต้นจากฐานความรู้` คำถาม fallback ชุดนี้ไม่ได้ผ่าน evidence gate ซ้ำในรอบที่เกิดข้อผิดพลาด จึงไม่แสดง label `ตรวจหลักฐานแล้ว`

ผลทดสอบผ่าน browser ที่ตรวจคำถามซึ่งผ่าน evidence gate 4 ข้อและตรวจว่าคำตอบมี citation อยู่ที่ `rag/reports/demo/grounded_suggestions_browser.json`

สามารถทดสอบ browser flow ซ้ำขณะเปิด Streamlit อยู่ได้ด้วย:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\verify_browser_suggestions.py `
    --rounds 4 `
    --output rag/reports/demo/grounded_suggestions_browser.json
```

## Environment และ Workflow

| ตัวแปร | การใช้งาน |
|---|---|
| `APP_ENV=local` | ระบุว่าสภาพแวดล้อมปัจจุบันเป็น local |
| `DATABASE_BACKEND=local` | ใช้ SQLite สำหรับ local demo |
| `ASSISTANT_DB_PATH` | ฐานข้อมูล Test Case #2 สำหรับ account, session, chat, Lead และ Human Handoff |
| `ANALYTICS_DB_PATH` | ฐานข้อมูล Test Case #1 สำหรับ agent, KPI, monthly performance และ contract |
| `CHECKPOINT_DB_PATH` | ฐานข้อมูล LangGraph checkpoint |
| `CHROMA_PATH` | ตำแหน่ง persistent Chroma vector database |
| `KB_MANIFEST_PATH` | ตำแหน่ง manifest ของ PDF และข้อมูล corpus |
| `EMBEDDING_MODEL` | path ของ E5 ที่ `rag/models/multilingual-e5-small` หรือชื่อโมเดล embeddings |
| `EMBEDDING_DEVICE=cpu` | ใช้ CPU ประมวลผล embeddings |
| `LLM_PROVIDER=google_ai_studio` | ใช้ Gemini สำหรับสร้างคำตอบและสกัดข้อมูลแบบ structured |
| `LLM_PROVIDER=none` | ไม่เรียก LLM ภายนอก และใช้ deterministic evidence extraction สำหรับการทดสอบแบบ offline |
| `GOOGLE_API_KEY` | API key ของ Google AI Studio โดยต้องกำหนดใน `.env` ของเครื่องผู้รัน |
| `LLM_MODEL` | ชื่อ Gemini model ที่ระบบเรียกใช้ |
| `LLM_TIMEOUT_SECONDS` | เวลาสูงสุดสำหรับรอ LLM provider; ถ้าไม่กำหนดจะใช้ค่าเริ่มต้น `30` วินาที |
| `LLM_MAX_OUTPUT_TOKENS` | จำนวน output tokens สูงสุด โดยค่าใน `.env.example` คือ `1024` |
| `MAX_RETRIEVAL_RETRIES` | จำนวน retry ของ LLM/provider ไม่ใช่จำนวนครั้งที่ค้น Chroma หรือ rewrite query |
| `RETRIEVAL_MODE=hybrid` | รวม semantic retrieval จาก E5 กับ lexical ranking สำหรับคำเฉพาะใน PDF |
| `TRACE_PATH=rag/traces` | ตำแหน่งจัดเก็บ PII-safe runtime trace |

## LangGraph State

State หลักของ LangGraph ประกอบด้วย:

- `messages`
- `session_id`
- `query`
- `standalone_query`
- `conversation_history`
- `active_product_id`
- `intent`
- `retrieved_chunks`
- `evidence_status`
- `evidence_reason`
- `answer`
- `response_type`
- `citations`
- `retrieval_attempts`
- `error_code`
- `lead_collection_active`
- `lead_draft`
- `lead_missing`
- `lead_saved_id`
- `lead_summary`
- `lead_product_candidates`
- `lead_refresh_required`
- `handoff_requested`

`thread_id` ไม่ได้เก็บเป็น field ภายใน graph state แต่ส่งผ่าน LangGraph config:

```python
{"configurable": {"thread_id": thread_id}}
```

Field `lead_confirmation_pending` และ `lead_confirmation_fingerprint` ยังอยู่ใน schema เพื่อรองรับโค้ดเดิม แต่ workflow ปัจจุบันไม่รอให้ลูกค้ายืนยันก่อนบันทึก โดยกำหนด `lead_confirmation_pending=False`

## LangGraph Routing

Workflow เริ่มจาก:

```text
START
  → prepare_turn
  → route_intent
```

`route_intent` แยกเส้นทางดังนี้:

- ข้อความทักทาย ขอบคุณ ลาก่อน ขอความช่วยเหลือ และข้อความนอกขอบเขตไป `direct_response` โดยไม่เรียก Gemini หรือ Chroma
- คำขอคุยกับพนักงานไป `direct_response` พร้อมกำหนด `handoff_requested=True`
- ข้อความแสดงความสนใจสมัครผลิตภัณฑ์ซึ่งไม่ใช่คำถามเกี่ยวกับรายละเอียดผลิตภัณฑ์โดยตรงไป `collect_lead`
- ระหว่าง Lead mode ข้อมูลเพิ่มเติมจากลูกค้าไป `collect_lead` ตราบใดที่ข้อความนั้นไม่ใช่คำถามเกี่ยวกับผลิตภัณฑ์
- ถ้าข้อมูล Lead ล่าสุดเกิน 183 วัน ระบบกำหนด `lead_refresh_required` เพื่อเริ่มเก็บข้อมูล Lead ชุดใหม่
- คำถามเกี่ยวกับผลิตภัณฑ์ไปยัง RAG workflow

เส้นทาง RAG เริ่มจาก:

```text
resolve_query
  → retrieve
  → assess_evidence
```

เมื่อหลักฐานเพียงพอ:

```text
assess_evidence
  → generate_grounded_answer
  → validate_answer_citations
  → END
```

เมื่อหลักฐานอ่อน ระบบปรับ retrieval query และค้นใหม่ได้หนึ่งรอบ:

```text
assess_evidence
  → rewrite_query_once
  → retrieve
  → assess_evidence
```

จำนวน rewrite ของ graph จำกัดไว้หนึ่งรอบด้วย `retrieval_attempts` และแยกจาก `MAX_RETRIEVAL_RETRIES` ซึ่งควบคุมการ retry ฝั่ง LLM/provider

กรณีต่อไปนี้ไปยัง `fallback`:

- ไม่พบหลักฐานเพียงพอ
- คำถามกำกวมและตรงกับหลายผลิตภัณฑ์ประกัน
- Retrieval, evidence assessment, generation หรือ citation validation เกิดข้อผิดพลาด
- คำตอบที่สร้างขึ้นไม่มี citation ที่ผ่านการตรวจสอบ

คำถามทั่วไปที่ดึงหลักฐานจากหลายผลิตภัณฑ์ประกันโดยไม่มีเจตนาเปรียบเทียบ จะให้ผู้ใช้ระบุผลิตภัณฑ์ประกันที่ต้องการก่อน

คำถามเปรียบเทียบสามารถใช้หลักฐานจากหลายผลิตภัณฑ์ประกันได้ ส่วนคำถามที่ระบุผลิตภัณฑ์ประกันชัดเจนจะจำกัด retrieval ตามผลิตภัณฑ์ประกันนั้นเมื่อระบบระบุ `active_product_id` ได้

## Human Handoff

หลังจบ LangGraph หน้าเว็บจะพิจารณา Human Handoff ในกรณีต่อไปนี้:

- ผู้ใช้ร้องขอพนักงานโดยตรง
- คำถามเกี่ยวกับผลิตภัณฑ์จบด้วยสถานะ `insufficient` และไม่มี citation
- คำถามเกี่ยวกับผลิตภัณฑ์จบด้วยสถานะ `error` และไม่มี citation

ข้อความนอกขอบเขตทั่วไปจะได้รับคำตอบผ่าน `direct_response` และไม่สร้าง Human Handoff อัตโนมัติ

เมื่อเข้า Human Handoff ระบบจะ:

1. สร้างเคสใน `handoff_cases`
2. พัก AI/RAG ของ session ลูกค้านั้น
3. แสดงเคสในคิวของพนักงาน
4. ให้พนักงานรับเคสและตอบลูกค้าโดยไม่เรียก AI/RAG
5. ป้องกันพนักงานคนอื่นตอบเคสเดียวกันพร้อมกัน
6. ปิดเคสเมื่อพนักงานกด **แก้ไขปัญหาแล้ว**
7. เปิดให้ session ของลูกค้ากลับมาใช้ AI/RAG โดยอัตโนมัติ
