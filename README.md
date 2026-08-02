# Obfuscated-RAG

A Streamlit app that lets you ask questions over a sensitive text document using a cloud LLM (Groq) **without ever sending the raw sensitive data to the llm**. It combines two privacy techniques, PII redaction and differential-privacy noise on embeddings and includes a self-auditing "red team" agent that tests whether the pipeline actually leaks anything.

## The problem this solves

You want to use a powerful cloud LLM (Groq) to answer questions about a document that contains PII — names, salaries, passwords. But you don't want that raw data leaving your machine and hitting a third-party API. This app redacts and noises the data *before* it goes out, and only reconstructs the real values locally, for authorized users, after the LLM responds.

## High-level workflow

```
Upload .txt file
      ↓
Presidio scans for PII → replaces with tokens like [PERSON_A1B2], [SALARY_C3D4]
      ↓
Text is chunked (500 chars, 50 overlap)
      ↓
Chunks are embedded (MiniLM) → Laplace noise added to embedding vectors (DP)
      ↓
Stored in a local Chroma vector database
      ↓
User asks a question → question also gets PII-redacted → similarity search over noisy embeddings
      ↓
Retrieved (already-redacted) chunks + question sent to Gemini
      ↓
Groq answers using only tokens, never real values
      ↓
If role = Admin → tokens are swapped back to real values locally
If role = Guest → tokens stay redacte(?i)(?:password|secret)\s*(?:[:=]|\s+is\s+)\s*([^\s\n.,]d
```

## Step-by-step breakdown

### 1. Document ingestion & PII redaction (`PresidioTranslator.blur_text`)
- Microsoft Presidio's `AnalyzerEngine` scans the uploaded text for built-in entities (names, emails, phone numbers, etc.) plus two custom regex recognizers added in `get_presidio_analyzer()`:
  - `PASSWORD` — catches patterns like `password: xyz` or `secret is xyz`
  - `SALARY` — catches `$` amounts
- Each detected span is replaced with a random token, e.g. `[PERSON_A1B2]`, and the real value is stored in `st.session_state.secure_mapping` (a token → real-value lookup table that never leaves your session).
- Overlapping detections are filtered out (keeps the longest match starting first) so tokens don't overlap or corrupt each other.

### 2. Chunking
- `RecursiveCharacterTextSplitter` splits the redacted document into 500-character chunks with 50-character overlap — standard RAG chunking so each chunk is small enough to embed and retrieve meaningfully, with overlap to avoid cutting sentences awkwardly at boundaries.

### 3. Embedding + differential privacy noise (`PrivacyAwareEmbeddings`, `gan_logic.py`)
- Chunks are embedded using `all-MiniLM-L6-v2` (a local HuggingFace sentence-embedding model — text → vector of numbers representing meaning).
- Before storing, `optimize_privacy_budget()` adds **Laplace noise** to every embedding vector (`DifferentialPrivacyAgent`). This is standard differential privacy: noise scaled by `sensitivity / epsilon`, so smaller epsilon = more noise = harder to reverse-engineer the original vector.
- To decide *how much* noise is enough, a `FastDiscriminator` (logistic regression) tries to tell clean vectors apart from noisy ones. If it can still tell them apart easily (accuracy > 0.55), the loop adds more noise (`epsilon /= 1.5`, up to 5 tries) and re-checks. Once the noisy vectors are indistinguishable enough, that noise level is locked in.

### 4. Vector storage
- Noisy embeddings + redacted chunk text are stored in a local **Chroma** vector database — this only ever contains tokens, never raw PII.

### 5. Query time
- User's question is also run through `blur_prompt()`, which swaps any recognizable real values (matched against the stored mapping) into their tokens, so the query itself doesn't reintroduce raw PII into what gets sent to the LLM.
- `similarity_search` retrieves the top-k most relevant (already-redacted) chunks from Chroma.
- A **canary token** (`CANARY_TOKEN`) is silently appended to the context sent to Groq. If this exact fake secret ever comes back in the LLM's response, the app knows something in the prompt/response pipeline leaked unexpectedly, and blocks the message.
- The prompt explicitly instructs Groq: don't try to compute on redacted values, just echo the token back if relevant.

### 6. Response handling
- **Admin role**: `reassemble_text()` swaps tokens in the LLM's response back to their real values *locally*, using the session's `secure_mapping`. The real data never touched Gemini — only the final display, on your machine, does the unredacting.
- **Guest role**: response is shown as-is, still redacted.
- Either way, if the canary token leaks through, the message is blocked and flagged instead of shown.

### 7. Agent 2 — adversarial audit (red-teaming your own pipeline)
- On demand, this generates a fake "honey token" secret, redacts it the same way real data would be, and asks Groq to repeat it back.
- Compares Groq's answer to the real fake secret using `difflib.SequenceMatcher` (string similarity).
- If the secret leaks (exact match or similarity > 0.6), the app locks itself down (`system_locked = True`) — a hard stop simulating "we caught a real leak, halt everything."

### 8. Privacy visualization (PCA plot)
- Projects both the original and noisy embedding vectors down to 2D using PCA, so you can visually see how much the noise shifted the data — a sanity check that noise is actually being applied and roughly how much it distorts the vector space.

## Key files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, PII redaction (Presidio), RAG pipeline, LLM calls, audit UI, PCA visualization |
| `noise.py` | Differential-privacy noise injection + noise-level calibration loop |

##Summary
The document is first analyzed by Presidio for any PII. This is redacted. Then we chunk it. We add laplacian noise to this. When the user asks a questions we run
Presidio through that and then add laplacian noise. similarity search will return the closest vectors. This is what the llm sees. Based on the user if he is admin or not we show.
