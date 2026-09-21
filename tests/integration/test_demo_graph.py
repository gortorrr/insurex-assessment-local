"""Exercise separate chat memory with one account-level customer profile."""

import tempfile
import unittest
from pathlib import Path

from insurex.config import Settings
from insurex.db import connect_local
from insurex.rag.checkpoint import persistent_checkpointer
from insurex.rag.graph import RagServices, build_rag_graph
from insurex.rag.runtime import LeadTurnHandler
from insurex.session import create_session


class DemoGraphTests(unittest.TestCase):
    def test_account_profile_is_shared_across_separate_graph_threads(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(checkpoint_db_path=Path(temporary) / "checkpoints.sqlite")
            connection = connect_local(Path(temporary) / "business.sqlite")
            try:
                a = create_session(connection, "demo-owner")
                b = create_session(connection, "demo-owner")
                aliases = {"product-a": ["คุ้มตลอดชีพ พลัส"]}

                def graph_for(session, saver):
                    handler = LeadTurnHandler(connection, session.session_id, session.access_token,
                                              {"product-a"}, session.session_id, aliases)
                    services = RagServices(lambda q, p: [], lambda q, c: "insufficient",
                                           lambda q, c: ("", []), lambda q: q, handler)
                    return build_rag_graph(services, checkpointer=saver)

                def ask(graph, session, query):
                    return graph.invoke({"query": query, "session_id": session.session_id,
                                         "active_product_id": None},
                                        {"configurable": {"thread_id": session.thread_id}})

                with persistent_checkpointer(settings) as saver:
                    ga, gb = graph_for(a, saver), graph_for(b, saver)
                    for graph, session in ((ga, a), (gb, b)):
                        ask(graph, session, "สนใจสมัคร คุ้มตลอดชีพ พลัส")
                    ask(ga, a, "ชื่อ เอก อาชีพ ครู รายได้ 30000 บาทต่อเดือน")
                    interrupted = ask(ga, a, "แล้วคุ้มครองกี่ปี")
                    self.assertEqual(interrupted['intent'], 'question')
                    self.assertEqual(interrupted['lead_draft']['name'], 'เอก')
                    self.assertIn('product a', interrupted['standalone_query'])
                with persistent_checkpointer(settings) as saver:
                    ga, gb = graph_for(a, saver), graph_for(b, saver)
                    completed = ask(gb, b, "เบอร์โทร 0812345678")
                    self.assertEqual(completed["lead_draft"]["name"], "เอก")
                    self.assertIsNotNone(completed["lead_saved_id"])
                    self.assertEqual(ask(ga, a, "ข้อมูลของแชตนี้")["lead_draft"]["name"], "เอก")
                rows = connection.execute("SELECT owner_id, session_id, name, product_id FROM leads").fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(tuple(rows[0]), ("demo-owner", a.session_id, "เอก", "product-a"))
                drafts = connection.execute(
                    "SELECT owner_id, last_session_id, name FROM lead_drafts"
                ).fetchall()
                self.assertEqual([tuple(row) for row in drafts], [("demo-owner", a.session_id, "เอก")])
            finally:
                connection.close()
