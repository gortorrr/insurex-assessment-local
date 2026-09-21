"""Persistent LangGraph checkpointer factories for the two runtime profiles.

The optional imports stay lazy so the verified analytics/KPI/lead baseline can
run without installing the live RAG stack. Callers must keep this context open
for the lifetime of the compiled graph and close it after the last invocation.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
import sqlite3
from pathlib import Path
from typing import Any, Iterator


class CheckpointDependencyError(RuntimeError):
    """Raised when the selected persistent checkpointer is not installed."""


@contextmanager
def persistent_checkpointer(settings: Any) -> Iterator[Any]:
    """Yield a file-backed SQLite or PostgreSQL LangGraph checkpointer.

    The PostgreSQL saver uses the same Azure PostgreSQL connection string as
    the business database, but owns its LangGraph checkpoint tables. The
    caller must pass the trusted runtime settings rather than user input.
    """

    if settings.database_backend == "local":
        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise CheckpointDependencyError(
                "Install langgraph-checkpoint-sqlite for the local profile"
            ) from exc
        path = Path(settings.checkpoint_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("insurex.rag.contracts", "RetrievedChunk"),
                ("insurex.rag.contracts", "Citation"),
            ]
        )
        # sqlite3's transaction context does not close its connection.
        with closing(sqlite3.connect(str(path), check_same_thread=False)) as connection:
            saver = SqliteSaver(connection, serde=serde)
            saver.setup()
            yield saver
        return

    if settings.database_backend == "postgresql":
        try:
            import psycopg
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise CheckpointDependencyError(
                "Install langgraph-checkpoint-postgres for the Azure profile"
            ) from exc
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("insurex.rag.contracts", "RetrievedChunk"),
                ("insurex.rag.contracts", "Citation"),
            ]
        )
        with psycopg.connect(settings.database_url) as connection:
            saver = PostgresSaver(connection, serde=serde)
            saver.setup()
            yield saver
        return

    raise ValueError("DATABASE_BACKEND must be local or postgresql")
