"""Allowlisted local traces: never serialize prompts, lead values or exceptions."""
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path


class TurnTrace:
    def __init__(self, directory, owner_id, session_id):
        self.directory = Path(directory)
        self.started = self.last = time.perf_counter()
        self.data = {"trace_id": uuid.uuid4().hex, "started_at": datetime.now(UTC).isoformat(),
                     "owner_ref": hashlib.sha256(owner_id.encode()).hexdigest()[:16],
                     "session_ref": hashlib.sha256(session_id.encode()).hexdigest()[:16], "nodes": []}

    def node(self, name, update):
        now = time.perf_counter()
        node = {"node": name, "elapsed_ms": round((now-self.last)*1000, 2)}
        for key in ('candidate_count', 'grounded_count'):
            if isinstance(update.get(key), int):
                node[key] = update[key]
        self.data['nodes'].append(node)
        self.last = now
        if 'retrieved_chunks' in update:
            self.data['retrieved'] = [{"doc_id": c.doc_id, "page": c.page_number, "chunk_id": c.chunk_id}
                                      for c in update['retrieved_chunks']]

    def finish(self, status, budget=None):
        self.data['status'] = status if status in {'complete', 'cancelled', 'error'} else 'error'
        self.data['duration_ms'] = round((time.perf_counter()-self.started)*1000, 2)
        self.data['requests'] = budget.requests if budget else 0
        self.data['usage'] = [{k: v for k, v in item.items()
                               if k in {'input_tokens','output_tokens','total_tokens'} and isinstance(v, int)}
                              for item in (budget.usage if budget else [])]
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / (self.data['trace_id']+'.json')).write_text(json.dumps(self.data, indent=2), encoding='utf-8')
        return self.data['trace_id']
