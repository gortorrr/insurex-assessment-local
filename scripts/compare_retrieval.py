"""Paired semantic/hybrid retrieval comparison; does not grade LLM answers."""
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from insurex.config import Settings
from insurex.rag.runtime import _retrieve_factory, _assess_local
from insurex.rag.contracts import assess_candidates
from insurex.rag.graph import build_standalone_query


def main():
    base=Settings.from_env()
    result={'executed_at':datetime.now(UTC).isoformat(), 'note':'Paired local retrieval evaluation, not blinded and not answer accuracy.', 'modes':{}}
    for mode in ('semantic','hybrid'):
        retrieve=_retrieve_factory(replace(base,retrieval_mode=mode))
        retrieve('คุ้มออมสุข',None)  # Warm model load outside timing.
        results=[]
        for path in [Path('rag/eval/rag_cases.jsonl'),Path('rag/eval/rag_cases_heldout_v2.jsonl'),Path('rag/eval/rag_cases_additional_v3.jsonl')]:
            for line in path.read_text(encoding='utf-8').splitlines():
                case=json.loads(line)
                start=time.perf_counter()
                chunks=retrieve(build_standalone_query(case['query'],case.get('history')),case.get('product_id'))
                ms=(time.perf_counter()-start)*1000
                status=assess_candidates(chunks,evidence_status=_assess_local(case['query'],chunks),query=case['query'],active_product_id=case.get('product_id')).status
                ranks=[i for i,c in enumerate(chunks,1) if c.doc_id==case.get('expected_doc_id') and c.page_number in case.get('expected_pages',[])]
                expected=case['expected_type']
                passed=(bool(ranks) and status=='sufficient') if expected=='answerable' else status==('insufficient' if expected=='no_answer' else 'ambiguous')
                results.append({'case_id':case['case_id'],'dataset':path.name,'passed':passed,'rank':min(ranks) if ranks else None,'status':status,'retrieval_ms':round(ms,2)})
        answerable=[r for r in results if r['rank'] is not None]
        result['modes'][mode]={'passed':sum(r['passed'] for r in results),'total':len(results),
                               'mean_retrieval_ms':round(sum(r['retrieval_ms'] for r in results)/len(results),2),
                               'results':results}
        print(mode,result['modes'][mode]['passed'], '/',len(results),flush=True)
    Path('rag/reports/demo/retrieval_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
