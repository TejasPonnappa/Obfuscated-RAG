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


# The prompt the app sends with the retrieved (redacted) chunks. The first version made the model refuse to repeat password tokens
# ("I can't help with that"); this one says plainly that tags are harmless placeholders and that repeating them is the job.
ANSWER_PROMPT = """You answer questions about a document in which personal details were replaced by placeholder tags such as
[PERSON_XXXX], [SALARY_XXXX], [EMAIL_ADDRESS_XXXX] or [PASSWORD_XXXX]. The tags are meaningless placeholders, not real secrets,
so repeating them is safe and expected. The user's own system swaps each tag back to the real value after you answer.

Rules:
1. When the answer is a value that appears as a tag, reply with the tag exactly as written, for example "The salary is [SALARY_XXXX]."
2. Never say that you cannot access, reveal or help with a value. Just give the tag.
3. If asked to calculate or compare using a tag, say that you cannot compute on hidden values, and give the tag.
4. Use only the context below. If the answer is not in it, say that it is not in the document.

Context:
{context}

Question: {question}"""
