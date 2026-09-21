import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from insurex.rag.index import ingest_manifest_to_chroma, verify_corpus_current, _client
from insurex.rag.manifest import file_sha256


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(b)/255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


class CorpusRefreshTests(unittest.TestCase):
    def test_updated_document_blocks_stale_reads_and_reingestion_removes_old_chunks(self):
        root = Path(__file__).resolve().parents[2]
        corpus_root = root / 'rag'
        # Chroma holds native handles on Windows until process shutdown.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            copy = Path(temporary)
            manifest_path = copy/'knowledge_base/manifest.json'
            manifest = json.loads((corpus_root/'knowledge_base/manifest.json').read_text(encoding='utf-8'))
            for document in manifest['documents']:
                for key in ('local_path','corrected_text_path'):
                    target = copy/document[key]
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(corpus_root/document[key], target)
            manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            index = copy/'index'
            args = dict(root=copy,chroma_path=index,embedding_model='fake-test-only')
            with patch('insurex.rag.index._cached_embeddings',return_value=FakeEmbeddings()):
                ingest_manifest_to_chroma(manifest_path,**args)
                verify_corpus_current(index,manifest_path)
                document = manifest['documents'][0]
                corrected = copy/document['corrected_text_path']
                data = json.loads(corrected.read_text(encoding='utf-8'))
                old_text = data['pages'][0]['text']
                data['pages'][0]['text']='UPDATED DOCUMENT TEST ONLY'
                data['pages'][0]['sha256']=hashlib.sha256(data['pages'][0]['text'].encode()).hexdigest()
                corrected.write_text(json.dumps(data),encoding='utf-8')
                with self.assertRaises(RuntimeError):verify_corpus_current(index,manifest_path)
                ingest_manifest_to_chroma(manifest_path,**args)
                verify_corpus_current(index,manifest_path)
                records=_client(index).get_collection('insurex_kb').get(where={'doc_id':document['doc_id']})
                self.assertIn('UPDATED DOCUMENT TEST ONLY',records['documents'])
                self.assertFalse(any(text in old_text for text,meta in zip(records['documents'],records['metadatas']) if meta['page_number']==1))
                with patch('insurex.rag.index.validate_manifest',return_value=['simulated failure']):
                    with self.assertRaises(ValueError):ingest_manifest_to_chroma(manifest_path,**args)
                with self.assertRaises(RuntimeError):verify_corpus_current(index,manifest_path)
                ingest_manifest_to_chroma(manifest_path,**args)
                verify_corpus_current(index,manifest_path)
