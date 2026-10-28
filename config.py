"""Settings shared by the app and the evaluation scripts."""
import os

# Groq retires models; llama-3.1-8b-instant (the original choice) now returns 404. Override with GROQ_MODEL in .env.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def groq_model():
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
