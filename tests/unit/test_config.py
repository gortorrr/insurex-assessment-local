import os
import sys
import types
import unittest
from unittest.mock import patch

from insurex.config import Settings


class ConfigTests(unittest.TestCase):
    def test_streamlit_secrets_are_fallback_when_environment_is_empty(self):
        fake_streamlit = types.SimpleNamespace(
            secrets={
                "DATABASE_BACKEND": "postgresql",
                "DATABASE_URL": "postgresql://redacted",
                "LLM_PROVIDER": "google_ai_studio",
                "LLM_MODEL": "gemini-test",
                "GOOGLE_API_KEY": "redacted",
            }
        )
        names = [
            "DATABASE_BACKEND",
            "DATABASE_URL",
            "LLM_PROVIDER",
            "LLM_MODEL",
            "GOOGLE_API_KEY",
        ]
        clean_env = {"INSUREX_ENV_FILE": os.path.join(os.getcwd(), "missing.env")}
        clean_env.update({name: "" for name in names})
        with patch.dict(os.environ, clean_env, clear=False), patch.dict(
            sys.modules, {"streamlit": fake_streamlit}
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.database_backend, "postgresql")
        self.assertEqual(settings.llm_provider, "google_ai_studio")
        self.assertEqual(settings.llm_model, "gemini-test")
        self.assertEqual(settings.validate(), [])


if __name__ == "__main__":
    unittest.main()
