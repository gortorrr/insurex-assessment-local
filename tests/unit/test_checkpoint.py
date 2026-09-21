import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from insurex.config import Settings
from insurex.rag.checkpoint import CheckpointDependencyError, persistent_checkpointer


class CheckpointContractTests(unittest.TestCase):
    def test_local_missing_optional_dependency_is_explicit(self):
        with TemporaryDirectory() as directory:
            settings = Settings(
                database_backend="local",
                checkpoint_db_path=Path(directory) / "checkpoints.sqlite",
            )
            with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None}):
                with self.assertRaises(CheckpointDependencyError):
                    with persistent_checkpointer(settings):
                        pass

    def test_unsupported_backend_is_rejected_before_optional_import(self):
        settings = Settings(database_backend="unsupported")
        with self.assertRaisesRegex(ValueError, "DATABASE_BACKEND"):
            with persistent_checkpointer(settings):
                pass


if __name__ == "__main__":
    unittest.main()
