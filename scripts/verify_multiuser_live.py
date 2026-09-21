"""Exercise the actual UI turn worker with Gemini, isolated synthetic accounts and SQLite."""
import argparse
import json
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from insurex.auth import register
from insurex.config import Settings
from insurex.db import connect_local
from insurex.session import create_session, append_message, load_history
from rag.ui.app import _run_turn_job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',required=True,help='make bounded Gemini calls using synthetic data')
    parser.parse_args()
    settings=Settings.from_env()
    records=[]
    with tempfile.TemporaryDirectory() as t:
        settings=replace(settings,business_db_path=Path(t)/'db.sqlite',checkpoint_db_path=Path(t)/'cp.sqlite',
                         trace_path=Path(t)/'traces',max_retrieval_retries=0)
        catalog={d['product_id'] for d in json.loads(settings.kb_manifest_path.read_text(encoding='utf-8'))['documents']}
        c=connect_local(settings.business_db_path)
        try:
            a=create_session(c,register(c,'qa-alice','Synthetic-test-A-2026'))
            b=create_session(c,register(c,'qa-bob','Synthetic-test-B-2026'))
            def ask(session,label,query):
                request=f'qa-{len(records)}'
                history=load_history(c,session_id=session.session_id,access_token=session.access_token,product_only=True)
                append_message(c,session_id=session.session_id,access_token=session.access_token,role='user',content=query,request_id=request)
                result=_run_turn_job(settings,session_id=session.session_id,access_token=session.access_token,owner_id=session.owner_id,
                                     question=query,request_id=request,history=history,product_ids=catalog,mode='Gemini live',
                                     thread_id=session.thread_id,cancel_event=Event())
                records.append({'session':label,'step':len(records)+1,'answer':result.get('answer','').replace('0812345678','081•••••78'),
                                'missing':result.get('lead_missing'),'citations':result.get('citations'), 'error':result.get('error_code')})
                print(json.dumps({'step':len(records),'session':label,'error':result.get('error_code'),'missing':result.get('lead_missing')},ensure_ascii=False),flush=True)
                assert not result.get('error_code'), result.get('error_code')
                return result
            ask(a,'A','สนใจสมัคร คุ้มตลอดชีพ พลัส')
            ask(b,'B','สนใจสมัคร คุ้มออมสุข 25/15')
            data=ask(a,'A','ผมชื่อ เอกทดสอบ เป็นครู ได้เดือนละ 30,000 บาท ติดต่อ 0812345678 ครับ')
            assert data['lead_missing']==[]
            assert data['lead_saved_id'] is not None
            interrupted=ask(a,'A','คุ้มตลอดชีพ พลัส คุ้มครองถึงอายุเท่าไร')
            assert interrupted['citations']
            missing=ask(b,'B','ชื่อ บีทดสอบ')
            assert missing['lead_missing']
            rows=c.execute('SELECT owner_id,session_id,name,occupation,income_thb,phone_normalized FROM leads').fetchall()
            assert len(rows)==1 and rows[0]['session_id']==a.session_id and rows[0]['name']=='เอกทดสอบ'
            assert rows[0]['income_thb']==30000 and rows[0]['occupation']=='ครู'
            draft_b=c.execute('SELECT name FROM lead_drafts WHERE owner_id=?',(b.owner_id,)).fetchone()
            assert draft_b and draft_b['name']=='บีทดสอบ'
            trace_data=[json.loads(p.read_text()) for p in settings.trace_path.glob('*.json')]
            serialized=json.dumps(trace_data,ensure_ascii=False)
            assert 'เอกทดสอบ' not in serialized and '0812345678' not in serialized
            payload={'executed_at':datetime.now(UTC).isoformat(),'model':settings.llm_model,'passed':True,'synthetic_only':True,
                     'turns':records,'lead_counts':{'A':1,'B':0},'shared_database':True,'fresh_worker_per_turn':True,
                     'trace_count':len(trace_data),'trace_pii_check':True,'requests':sum(x['requests'] for x in trace_data)}
            Path('rag/reports/demo/multiuser_live.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        finally:c.close()


if __name__=='__main__':main()

