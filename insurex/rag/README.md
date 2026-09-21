# RAG Application

ส่วนนี้เป็นงาน **Test Case #2** ครอบคลุม LangChain, LangGraph, ChromaDB, PDF Knowledge Base, Structured Lead Collection, Conversation Memory, Multi-user Session Separation และ Human Handoff

## โครงสร้าง Directory

- `ui/app.py` เป็น Streamlit workspace สำหรับลูกค้าและพนักงานขาย
- `knowledge_base/` เก็บ corpus manifest, PDF ต้นฉบับ และ verified text extraction
- `chroma_db/` เป็น persistent Chroma vector database ที่สร้างจาก PDF ใน Knowledge Base
- `models/` เก็บ local multilingual E5 embedding model
- `eval/` เก็บชุดคำถามและ expected evidence สำหรับประเมิน retrieval
- `db/assistant.sqlite` เก็บ account, session, message, Lead, login session และ Human Handoff 
- `db/checkpoints.sqlite` เก็บ LangGraph checkpoints แยกจากข้อมูล application
- `reports/demo/` เก็บผลการทดสอบ, structured evidence และ browser test artifacts
- `traces/` เก็บ runtime telemetry แบบ PII-safe โดยไม่เก็บ prompt, คำตอบ หรือค่าข้อมูล Lead

## การสร้าง Vector Database

รันคำสั่งนี้จาก root ของ repository เมื่อยังไม่มี `rag/chroma_db` หรือเมื่อ PDF, manifest, extraction, embedding model หรือ chunk configuration มีการเปลี่ยนแปลง:

```powershell
.venv\Scripts\python.exe -X utf8 -m scripts.ingest_rag
```

ไม่จำเป็นต้อง ingest ใหม่ทุกครั้งก่อนเปิดแอป หาก Chroma index ปัจจุบันตรงกับ corpus fingerprint แล้ว

หาก index ไม่ตรงกับ PDF, manifest หรือ configuration ปัจจุบัน ระบบจะหยุด retrieval แทนการตอบจาก index เก่า

## การเปิด Web Application

ตรวจว่าไฟล์ `.env` มี Gemini API key ก่อนเปิด Live RAG:

```env
LLM_PROVIDER=google_ai_studio
LLM_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=ใส่_API_KEY_ของผู้รัน
```

จากนั้นเปิด Streamlit จาก root ของ repository:

```powershell
.venv\Scripts\python.exe -X utf8 -m streamlit run rag/ui/app.py --server.port 8513
```

เปิดหน้าเว็บที่:

```text
http://localhost:8513/
```

หากกำหนด `LLM_PROVIDER=none` ระบบจะไม่เรียก Gemini และใช้ offline deterministic evidence extraction สำหรับการทดสอบแทน