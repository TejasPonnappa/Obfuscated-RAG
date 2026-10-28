"""Settings shared by the app and the evaluation scripts."""
import os

# Groq retires models; llama-3.1-8b-instant (the original choice) now returns 404. Override with GROQ_MODEL in .env.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def groq_model():
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

IDLE_LIMIT_SECONDS = 30 * 60      # after this long without a click, the token table and vector store are wiped


def is_idle(last_active, now, limit=IDLE_LIMIT_SECONDS):
    """True if the session has been quiet for longer than limit (never idle before the first interaction)."""
    return last_active is not None and now - last_active > limit
