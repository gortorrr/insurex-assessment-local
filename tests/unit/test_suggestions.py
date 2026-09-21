import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from insurex.db import connect_local
from insurex.session import create_session, append_message, annotate_turn
from insurex.suggestions import product_history, context_key, generate_questions, DEFAULT_QUESTIONS, select_grounded_questions
from insurex.rag.contracts import RetrievedChunk
from insurex.rag.runtime import RequestBudget


class Model:
    def __init__(self,questions):self.questions=questions;self.prompt=''
    def with_structured_output(self,*args,**kwargs):return self
    def invoke(self,prompt):self.prompt=prompt;return {'parsed':{'questions':self.questions}}


class SuggestionTests(unittest.TestCase):
    def test_selects_only_questions_that_pass_the_rag_evidence_gate(self):
        calls=[]
        def retrieve(question, product_id):
            calls.append((question, product_id))
            if 'ตอบไม่ได้' in question:
                return []
            return [RetrievedChunk(
                doc_id='doc-a', product_id='product-a', product_name='Product A',
                page_number=1, chunk_id='doc-a-p1-c1', text='Product A ความคุ้มครอง เบี้ย อายุ เงื่อนไข',
                source_url='https://example.test/a.pdf', score=.9, text_quality='verified',
            )]
        services=SimpleNamespace(
            retrieve=retrieve,
            assess=lambda question, chunks: 'sufficient' if chunks else 'insufficient',
        )
        products={'product-a': {'product_name':'Product A','product_aliases':['ผลิตภัณฑ์เอ']}}
        candidates=[
            'ผลิตภัณฑ์เอ มีความคุ้มครองอะไร',
            'ผลิตภัณฑ์เอ ตอบไม่ได้หรือไม่',
            'ผลิตภัณฑ์เอ มีเบี้ยอย่างไร',
            'ผลิตภัณฑ์เอ รับอายุเท่าไร',
            'ผลิตภัณฑ์เอ มีเงื่อนไขอะไร',
        ]
        selected, diagnostics=select_grounded_questions(
            candidates, services, products, desired=4, fallbacks=[]
        )
        self.assertEqual(len(selected),4)
        self.assertNotIn('ผลิตภัณฑ์เอ ตอบไม่ได้หรือไม่',selected)
        self.assertTrue(any(row['status']=='insufficient' for row in diagnostics))
        self.assertTrue(all(product_id=='product-a' for _,product_id in calls))

    def test_history_is_owner_scoped_and_excludes_lead_turns(self):
        with tempfile.TemporaryDirectory() as t:
            c=connect_local(Path(t)/'db.sqlite')
            try:
                a,b=create_session(c,'alice'),create_session(c,'bob')
                for session,text,req,product in [(a,'คุ้มออมสุข ออมกี่ปี','a',True),(b,'ตลอดชีพ 90/20','b',True),
                                                 (a,'ผมเอก ทำงานครู ได้เดือนละ 30000 บาท','private',False)]:
                    append_message(c,session_id=session.session_id,access_token=session.access_token,role='user',content=text,request_id=req)
                    annotate_turn(c,session.session_id,session.access_token,req,product_turn=product,citations=[])
                self.assertEqual(product_history(c,'alice'),['คุ้มออมสุข ออมกี่ปี'])
                self.assertEqual(product_history(c,'bob'),['ตลอดชีพ 90/20'])
                self.assertNotEqual(context_key('alice',[]),context_key('bob',[]))
            finally:c.close()

    def test_exactly_four_unique_safe_questions_and_history_in_prompt(self):
        model=Model(DEFAULT_QUESTIONS)
        result=generate_questions(model,RequestBudget(),['คุ้มออมสุข'],['คุ้มออมสุข'])
        self.assertEqual(len(result),4)
        self.assertIn('คุ้มออมสุข',model.prompt)
        generate_questions(model,RequestBudget(),['ผมเอกทดสอบ คุ้มออมสุข เบี้ย 0812345678'],['คุ้มออมสุข'])
        self.assertNotIn('เอกทดสอบ',model.prompt)
        self.assertNotIn('0812345678',model.prompt)
        for bad in [DEFAULT_QUESTIONS[:3],['คำถาม']*4,[*DEFAULT_QUESTIONS[:3],'เบอร์โทร 0812345678 ใช่ไหม'],
                    [*DEFAULT_QUESTIONS[:3],'สนใจสมัคร คุ้มออมสุข']]:
            with self.assertRaises(ValueError):generate_questions(Model(bad),RequestBudget(),[],[])
