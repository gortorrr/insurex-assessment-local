import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from insurex.auth import authenticate, register
from insurex.db import connect_local
from insurex.session import create_session, append_message, list_owned_sessions, resume_owned_session, load_history, annotate_turn
from insurex.telemetry import TurnTrace
from insurex.rag.lead_extractor import LlmLeadExtractor
from insurex.rag.runtime import RequestBudget


class FakeModel:
    def __init__(self, payload): self.payload = payload
    def with_structured_output(self, *args, **kwargs): return self
    def invoke(self, prompt): return {'parsed': self.payload}


class WorkspaceTests(unittest.TestCase):
    def test_product_history_excludes_free_form_lead_and_preserves_citations(self):
        with tempfile.TemporaryDirectory() as t:
            c=connect_local(Path(t)/'db.sqlite')
            try:
                s=create_session(c,'owner')
                for req,content in [('lead','เอกครับ เป็นครู เดือนละ 30000 บาท'),('rag','คุ้มตลอดชีพ พลัส คุ้มครองกี่ปี')]:
                    append_message(c,session_id=s.session_id,access_token=s.access_token,role='user',content=content,request_id=req)
                    append_message(c,session_id=s.session_id,access_token=s.access_token,role='assistant',content='คำตอบ '+req,request_id=req+':assistant')
                annotate_turn(c,s.session_id,s.access_token,'rag',product_turn=True,citations=[{'doc_id':'d','page':1}])
                history=load_history(c,session_id=s.session_id,access_token=s.access_token,product_only=True,include_metadata=True)
                self.assertEqual(len(history),2)
                self.assertNotIn('30000',str(history))
                self.assertEqual(history[1]['metadata']['citations'][0]['doc_id'],'d')
            finally:c.close()

    def test_background_job_rejects_wrong_owner_and_thread_before_model_calls(self):
        from rag.ui.app import _run_turn_job
        from insurex.config import Settings
        from threading import Event
        with tempfile.TemporaryDirectory() as t:
            settings=Settings(business_db_path=Path(t)/'db.sqlite',checkpoint_db_path=Path(t)/'cp.sqlite',trace_path=Path(t)/'traces')
            c=connect_local(settings.business_db_path)
            try:s=create_session(c,'alice')
            finally:c.close()
            for owner,thread in [('bob',s.thread_id),('alice','wrong-thread')]:
                with patch('rag.ui.app.build_gemini_services') as model:
                    result=_run_turn_job(settings,session_id=s.session_id,access_token=s.access_token,owner_id=owner,
                                         question='private',request_id='r',history=[],product_ids={'p'},mode='Gemini live',
                                         thread_id=thread,cancel_event=Event())
                    self.assertEqual(result['error_code'],'PermissionError')
                    model.assert_not_called()
    def test_accounts_cannot_read_resume_or_see_each_others_history(self):
        with tempfile.TemporaryDirectory() as t:
            c = connect_local(Path(t)/'db.sqlite')
            try:
                a = register(c,'alice','long-password-a')
                b = register(c,'bob','long-password-b')
                self.assertNotEqual(a,b)
                self.assertEqual(authenticate(c,'ALICE','long-password-a')['owner_id'],a)
                with self.assertRaises(PermissionError): authenticate(c,'alice','wrong')
                sa,sb=create_session(c,a),create_session(c,b)
                append_message(c,session_id=sa.session_id,access_token=sa.access_token,role='user',content='Private customer A')
                self.assertEqual(list_owned_sessions(c,b),[])
                with self.assertRaises(PermissionError): resume_owned_session(c,sa.session_id,b)
                with self.assertRaises(PermissionError): load_history(c,session_id=sa.session_id,access_token=sb.access_token)
                row=c.execute('SELECT password_hash,salt FROM user_accounts WHERE owner_id=?',(a,)).fetchone()
                self.assertNotIn('long-password',row['password_hash'])
            finally:c.close()

    def test_login_throttles_and_duplicate_accounts_cannot_take_over(self):
        with tempfile.TemporaryDirectory() as t:
            c=connect_local(Path(t)/'db.sqlite')
            try:
                register(c,'alice','long-password-a')
                with self.assertRaises(ValueError):register(c,'Alice','replacement-pass')
                for _ in range(5):
                    with self.assertRaises(PermissionError):authenticate(c,'alice','wrong')
                with self.assertRaises(PermissionError):authenticate(c,'alice','long-password-a')
            finally:c.close()

    def test_extractor_quotes_natural_language_and_rejects_invented_values(self):
        model=FakeModel({'name':{'value':'เอก','evidence':'ผมเอกครับ'},
                         'occupation':{'value':'ครู','evidence':'ทำงานเป็นครู'},
                         'income':{'value':'30,000 บาท','evidence':'ได้เดือนละ 30,000 บาท'}})
        extractor=LlmLeadExtractor(model,RequestBudget())
        draft=extractor('ผมเอกครับ ทำงานเป็นครู ได้เดือนละ 30,000 บาท','p',['name','occupation','income'])
        self.assertEqual(draft.name,'เอก')
        self.assertEqual(draft.occupation,'ครู')
        self.assertEqual(str(draft.income),'30000')
        self.assertEqual(draft.income_period,'monthly_thb')
        model.payload={'name':{'value':'สมชาย','evidence':'ผมเอกครับ'}}
        with self.assertRaises(ValueError):extractor('ผมเอกครับ','p',['name'])

    def test_income_range_and_annual_are_never_silently_normalized(self):
        for raw in ['30000-40000 บาทต่อเดือน','360000 บาทต่อปี']:
            draft=LlmLeadExtractor(FakeModel({'income':{'value':raw,'evidence':raw}}),RequestBudget())(raw,'p',['income'])
            self.assertTrue(draft.income is None or draft.income_period=='annual_thb')

    def test_trace_drops_all_chat_lead_and_exception_text(self):
        with tempfile.TemporaryDirectory() as t:
            trace=TurnTrace(t,'owner','session')
            trace.node('collect_lead',{'query':'ชื่อ เอก 0812345678','lead_draft':{'income':45000},'answer':'secret'})
            trace.finish('complete',RequestBudget())
            text=next(Path(t).glob('*.json')).read_text()
            for forbidden in ['เอก','0812345678','45000','secret','lead_draft','query']:
                self.assertNotIn(forbidden,text)

    def test_ui_login_logout_and_two_browser_sessions_are_isolated(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ,{'ASSISTANT_DB_PATH':str(Path(t)/'db.sqlite'),
             'CHECKPOINT_DB_PATH':str(Path(t)/'cp.sqlite'),'LLM_PROVIDER':'none'}):
            c=connect_local(Path(t)/'db.sqlite')
            try:
                register(c,'alice','long-password-a'); register(c,'bob','long-password-b')
                a=AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'rag/ui/app.py'),default_timeout=60).run()
                self.assertFalse(a.exception)
                self.assertEqual(len(a.chat_input),0)
                a.text_input(key='login_username').set_value('alice')
                a.text_input(key='login_password').set_value('long-password-a')
                a.button(key='FormSubmitter:login_form-เข้าสู่พื้นที่ทำงาน').click().run()
                self.assertFalse(a.exception)
                sid=a.session_state['lead_session_id']
                append_message(c,session_id=sid,access_token=a.session_state['lead_access_token'],role='user',content='Alice private lead')
                b=AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'rag/ui/app.py'),default_timeout=60).run()
                b.text_input(key='login_username').set_value('bob')
                b.text_input(key='login_password').set_value('long-password-b')
                b.button(key='FormSubmitter:login_form-เข้าสู่พื้นที่ทำงาน').click().run()
                self.assertFalse(b.exception)
                self.assertNotEqual(sid,b.session_state['lead_session_id'])
                self.assertNotIn('Alice private lead',str([m.value for m in b.markdown]))
                a.button(key='logout').click().run()
                self.assertFalse(a.exception)
                self.assertNotIn('principal',a.session_state)
                self.assertNotIn('lead_access_token',a.session_state)
            finally:c.close()

    def test_cookie_delete_is_idempotent_when_component_cache_lags(self):
        from rag.ui.app import delete_cookie_safely

        class LaggingCookieManager:
            def __init__(self):
                self.calls = []

            def delete(self, cookie, key="delete"):
                self.calls.append((cookie, key))
                raise KeyError(cookie)

        manager = LaggingCookieManager()
        self.assertFalse(
            delete_cookie_safely(manager, "insurex_login", key="expired-cookie")
        )
        self.assertEqual(manager.calls, [("insurex_login", "expired-cookie")])



