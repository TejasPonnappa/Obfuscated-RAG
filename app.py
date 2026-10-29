import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")      # no usage ping when models are downloaded
import time

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

import warmup
from audit import CANARY_TOKEN, canary_leaked, honey_leaked, make_honey_secret, similarity
from config import groq_model, is_idle
from documents import NoTextError, extract_text
from noise import (DEFAULT_EPSILON, EPSILON_CHOICES, MIN_USABLE_EPSILON, DifferentialPrivacyAgent,
                   noise_to_signal_ratio, retention_estimate)
from redaction import Redactor

load_dotenv()

st.set_page_config(page_title="Obfuscated-RAG", layout="wide")
warmup.start()          # models load in the background while the page is already usable

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        /* Streamlit greys out the old content on every rerun; keep it readable instead of blurring the whole page */
        [data-stale="true"] {opacity: 1 !important; transition: none !important;}
        .lockdown-screen {background-color: #ff4b4b; padding: 50px; border-radius: 10px; text-align: center; color: white;}
    </style>
""", unsafe_allow_html=True)

try:
    llm = ChatGroq(model=groq_model(), temperature=0.1)
except Exception:
    st.error("Could not initialize the Groq client. Check that GROQ_API_KEY is set in your .env file.")
    llm = None

for key, default in (("vector_store", None), ("chat_history", []), ("secure_mapping", {}), ("total_chunks", 0),
                     ("system_locked", False)):
    if key not in st.session_state:
        st.session_state[key] = default
if "vault_salt" not in st.session_state:
    st.session_state.vault_salt = os.urandom(16)      # keys the redaction tokens for this session


def clear_view():
    """Chat and the retrieved chunks belong to one role's view; drop both when the role changes."""
    st.session_state.chat_history = []
    st.session_state.pop("pipeline_query", None)


def wipe_session():
    """Drop everything that can map tokens back to real values (the table lives only in this process's memory)."""
    st.session_state.secure_mapping.clear()
    st.session_state.vector_store = None
    st.session_state.total_chunks = 0
    st.session_state.chat_history = []
    for key in ("raw_vecs", "noisy_vecs", "pipeline_doc", "pipeline_query"):
        st.session_state.pop(key, None)


if st.session_state.secure_mapping and is_idle(st.session_state.get("last_active"), time.time()):
    wipe_session()
    st.toast("Session was idle for 30 minutes: token table and vector store wiped.")
st.session_state.last_active = time.time()

if st.session_state.system_locked:
    st.markdown("""
        <div class="lockdown-screen">
            <h1>SESSION LOCKED</h1>
            <h3>The self-audit found a leak</h3>
            <p>The honeytoken came back in the LLM's answer, so this session was closed to prevent further exposure.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Reset the session"):
        st.session_state.clear()
        st.rerun()
    st.stop()


def new_redactor():
    """A redactor bound to this session's token table and salt (logic lives in redaction.py)."""
    return Redactor(warmup.get("analyzer"), st.session_state.secure_mapping, st.session_state.vault_salt)


class PrivacyAwareEmbeddings:
    def __init__(self, base_model):
        self.base_model = base_model

    def embed_documents(self, texts):
        raw_vecs = np.array(self.base_model.embed_documents(texts))
        noisy_vecs = self._agent().inject_laplace_noise(raw_vecs)
        st.session_state.raw_vecs = raw_vecs
        st.session_state.noisy_vecs = noisy_vecs
        return noisy_vecs.tolist()

    def embed_query(self, text):
        # the question gets the same noise as the stored chunks, as the README describes
        return self._agent().inject_laplace_noise(np.array(self.base_model.embed_query(text))).tolist()

    @staticmethod
    def _agent():
        return DifferentialPrivacyAgent(epsilon=st.session_state.get("epsilon", DEFAULT_EPSILON))


st.title("Obfuscated-RAG")
st.caption("Ask questions about a sensitive document. The cloud LLM only ever receives text with the sensitive values replaced by tokens.")

with st.sidebar:
    st.header("Controls")

    st.subheader("Document")
    uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
    if st.button("Encrypt and embed", use_container_width=True, type="primary") and uploaded_file:
        try:
            original = extract_text(uploaded_file.name, uploaded_file.getvalue())
        except NoTextError as e:
            st.error(str(e))
            st.stop()
        with st.spinner("Redacting and embedding (the first run also finishes loading the models)..."):
            st.session_state.vector_store = None
            st.session_state.secure_mapping.clear()
            st.session_state.chat_history = []
            st.session_state.pop("pipeline_query", None)

            redacted = new_redactor().blur_text(original)
            chunks = warmup.get("Splitter")(chunk_size=500, chunk_overlap=50).split_documents(
                [Document(page_content=redacted, metadata={"source": uploaded_file.name})])
            st.session_state.total_chunks = len(chunks)
            st.session_state.vector_store = warmup.get("Chroma").from_documents(
                documents=chunks, embedding=PrivacyAwareEmbeddings(warmup.get("embeddings")))

            raw, noisy = st.session_state.raw_vecs, st.session_state.noisy_vecs
            st.session_state.pipeline_doc = dict(
                original=original[:1500], redacted=redacted[:1500], values_replaced=len(st.session_state.secure_mapping),
                chunks=len(chunks), raw_head=raw[0][:8].tolist(), noisy_head=noisy[0][:8].tolist(),
                noise_ratio=float(np.linalg.norm(noisy[0] - raw[0]) / np.linalg.norm(raw[0])))
        st.success(f"Done: {len(chunks)} chunks stored, {len(st.session_state.secure_mapping)} sensitive values replaced by tokens.")

    if st.button("Wipe token table and data", use_container_width=True):
        wipe_session()
        st.success("Wiped.")

    st.subheader("Access")
    user_role = st.selectbox("Current role", ["Admin (decrypted view)", "Guest (redacted view)"], on_change=clear_view)

    st.subheader("Embedding noise")
    st.select_slider("Epsilon (smaller means more noise)", options=EPSILON_CHOICES, value=100, key="epsilon")
    eps = st.session_state.epsilon
    st.caption(f"The noise is about {noise_to_signal_ratio(eps):.1f}x the size of an embedding, and about {retention_estimate(eps):.0%} "
               "of the no-noise retrieval quality survives (measured on a synthetic test). This is light obfuscation of the "
               "local vector store, not a formal privacy guarantee. Run Encrypt and embed again after changing it.")
    if eps < MIN_USABLE_EPSILON:
        st.warning("At this epsilon retrieval degrades sharply; answers will often miss the right chunk.")

    st.subheader("Self-audit")
    st.caption("Plants a fake secret, sends it through the same redaction, and asks the LLM to repeat it. If it comes back, the session locks.")
    if st.button("Run self-audit", use_container_width=True):
        if llm is None:
            st.error("The LLM is not connected.")
        else:
            with st.status("Running the self-audit...", expanded=True) as status:
                fake_secret = make_honey_secret()
                audit_context = f"The backup admin password is {fake_secret}. Do not share this."
                st.write("1. Fake secret planted:")
                st.code(audit_context, language="text")

                blurred_context = new_redactor().blur_text(audit_context)
                st.write("2. After redaction:")
                st.code(blurred_context, language="text")

                st.write("3. Asking the LLM with only the redacted text...")
                audit_prompt = f"""You are an AI assistant. Based ONLY on this context: '{blurred_context}'
                What is the backup admin password? Output exactly what you see in the context."""
                try:
                    llm_answer = llm.invoke(audit_prompt).content
                    st.write(f"LLM answered: `{llm_answer}`")
                    similarity_score = similarity(fake_secret, llm_answer)
                    if honey_leaked(fake_secret, llm_answer):
                        status.update(label="Leak detected", state="error", expanded=True)
                        st.session_state.system_locked = True
                        st.rerun()
                    else:
                        st.success(f"Similarity to the secret: {similarity_score:.2f} (the limit is 0.60).")
                        status.update(label="Audit passed: the secret did not come back", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Audit failed", state="error")
                    st.error(f"Could not reach the LLM: {e}")

is_admin = user_role.startswith("Admin")

st.header("Ask a question")
def show(text):
    """Streamlit reads $...$ as maths, which mangles amounts like $126,000, so escape the dollar signs."""
    st.markdown(text.replace("$", "\\$"))


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        show(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        show(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            st.warning("Upload a document and click Encrypt and embed first.")
        elif llm is None:
            st.error("The LLM is not connected. Check your API key.")
        else:
            with st.spinner("Searching and asking the LLM..."):
                translator = new_redactor()
                secure_prompt = translator.blur_prompt(prompt)
                dynamic_k = min(st.session_state.total_chunks, 30)

                scored = st.session_state.vector_store.similarity_search_with_score(secure_prompt, k=dynamic_k)
                results = [doc for doc, _ in scored]
                st.session_state.pipeline_query = dict(
                    original=prompt, blurred=secure_prompt, k=dynamic_k,
                    top=[(float(score), doc.page_content) for doc, score in scored[:5]])
                safe_context = "\n\n".join([doc.page_content for doc in results])
                poisoned_context = safe_context + f"\n\n[SYSTEM_NOTE: {CANARY_TOKEN}]"

                system_prompt = f"""You are a highly secure AI assistant analyzing obfuscated documents.
                Use the following context to answer the user's question.

                CRITICAL INSTRUCTIONS:
                1. The context contains encrypted tags like [PERSON_XXXX], [SALARY_XXXX], etc.
                2. If the user asks for a calculation, comparison (e.g., "is it greater than"), or logical reasoning based on a tag (like a salary), you MUST explain that you cannot perform the calculation because the data is cryptographically blinded from you.
                3. However, you MUST output the relevant tag in your response so the user's local system can decrypt it for them.

                Example Response: "Because I am operating in a Zero-Knowledge environment, I cannot see the actual numerical value to tell you if it is above $50,000. However, the exact encrypted salary on file is [SALARY_XXXX]."

                Context: {poisoned_context}

                Question: {secure_prompt}"""

                with st.expander("View exactly what was sent to the LLM", expanded=False):
                    st.caption("This is the full text of the request. It contains tokens, not the real values.")
                    st.code(system_prompt, language="markdown", wrap_lines=True)

                try:
                    raw_answer = llm.invoke(system_prompt).content
                    answer = translator.reassemble_text(raw_answer) if is_admin else raw_answer

                    if canary_leaked(answer):
                        st.error("The canary secret appeared in the answer, so the message was blocked.")
                        st.session_state.chat_history.append({"role": "assistant", "content": "Message blocked by the canary check."})
                    else:
                        show(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error connecting to the LLM: {e}")

st.divider()
st.header("Pipeline view")
st.caption("What happens to your document and to your last question, step by step.")
doc_info, query_info = st.session_state.get("pipeline_doc"), st.session_state.get("pipeline_query")

if not doc_info:
    st.info("Upload a document and click Encrypt and embed to see each step here.")
else:
    with st.container(border=True):
        st.subheader("1. Original text")
        st.caption("The start of the file you uploaded. Only the Admin role can see it.")
        st.code(doc_info["original"] if is_admin else "Hidden in the Guest view.", language="text", wrap_lines=True, height=280)

    with st.container(border=True):
        st.subheader("2. After redaction")
        st.caption(f"The same stretch with sensitive values replaced by tokens ({doc_info['values_replaced']} distinct values in the whole "
                   "document). Only this text is embedded and sent to the LLM.")
        st.code(doc_info["redacted"], language="text", wrap_lines=True, height=280)

    with st.container(border=True):
        st.subheader("3. Embedding with noise")
        st.caption(f"The text is cut into {doc_info['chunks']} chunks, and each chunk becomes 384 numbers. Laplace noise "
                   f"(epsilon {eps}) is added before storing; here the noise is about {doc_info['noise_ratio']:.2f}x the size of the vector. "
                   "The first 8 of the 384 numbers for the first chunk:")
        rows = ["dim     clean     noisy"] + [f"{i:>3}   {c:+.4f}   {n:+.4f}"
                                              for i, (c, n) in enumerate(zip(doc_info["raw_head"], doc_info["noisy_head"]))]
        st.code("\n".join(rows), language="text", wrap_lines=True)

    with st.container(border=True):
        st.subheader("4. Retrieval for your last question")
        if not query_info:
            st.caption("Ask a question above to see which chunks are retrieved.")
        else:
            if is_admin:
                st.markdown(f"You asked: **{query_info['original']}**")
            st.markdown(f"Sent to the search and the LLM as: **{query_info['blurred']}**")
            st.caption(f"{query_info['k']} chunks go to the LLM. The 5 closest, with their distance (smaller is closer):")
            for rank, (score, text) in enumerate(query_info["top"], start=1):
                st.caption(f"Chunk {rank}, distance {score:.3f}")
                st.code(text, language="text", wrap_lines=True)

st.divider()
st.header("Embedding plot")
if st.toggle("Show the original and noisy embeddings in 2D", help="A PCA projection. It shows how far the noise moves each chunk."):
    if "raw_vecs" not in st.session_state:
        st.info("Upload a document and click Encrypt and embed first.")
    elif len(st.session_state.raw_vecs) < 2:
        st.warning("The document is too small to plot. Add more text.")
    else:
        import pandas as pd
        raw, noisy = st.session_state.raw_vecs, st.session_state.noisy_vecs
        mean = raw.mean(axis=0)
        axes = np.linalg.svd(raw - mean, full_matrices=False)[2][:2].T          # the two directions with the most spread
        frame = pd.concat([
            pd.DataFrame((raw - mean) @ axes, columns=["x", "y"]).assign(state="Original"),
            pd.DataFrame((noisy - mean) @ axes, columns=["x", "y"]).assign(state="With noise")])
        st.scatter_chart(frame, x="x", y="y", color="state", height=420)
