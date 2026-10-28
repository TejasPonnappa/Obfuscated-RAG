import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import os
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.decomposition import PCA
import pandas as pd
import altair as alt

from analyzer import build_analyzer
from noise import (DEFAULT_EPSILON, EPSILON_CHOICES, MIN_USABLE_EPSILON, DifferentialPrivacyAgent,
                   noise_to_signal_ratio, retention_estimate)
from redaction import Redactor
from audit import CANARY_TOKEN, canary_leaked, honey_leaked, make_honey_secret, similarity

load_dotenv()

st.set_page_config(page_title="Obfuscated-RAG | Zero-Knowledge", page_icon="", layout="wide")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .metric-container {background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333;}
        .lockdown-screen {background-color: #ff4b4b; padding: 50px; border-radius: 10px; text-align: center; color: white;}
    </style>
""", unsafe_allow_html=True)

try:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
except Exception as e:
    st.error("⚠️ Could not initialize Groq API. Make sure your .env file is set up correctly.")
    llm = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "secure_mapping" not in st.session_state:
    st.session_state.secure_mapping = {}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "system_locked" not in st.session_state:
    st.session_state.system_locked = False
if "vault_salt" not in st.session_state:
    st.session_state.vault_salt = os.urandom(16)      # keys the redaction tokens for this session

def clear_history():
    st.session_state.chat_history = []

if st.session_state.system_locked:
    st.markdown("""
        <div class="lockdown-screen">
            <h1>🚨 SYSTEM TERMINATED 🚨</h1>
            <h3>CRITICAL SECURITY VIOLATION DETECTED</h3>
            <p>Agent 2 (Honey Token Auditor) detected a catastrophic pipeline failure. The session has been permanently severed to prevent data exfiltration.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Reboot System (Clear State)"):
        st.session_state.clear()
        st.rerun()
    st.stop() 

@st.cache_resource
def get_presidio_analyzer():
    return build_analyzer()

def PresidioTranslator():
    """A redactor bound to this session's token table and salt (logic lives in redaction.py)."""
    return Redactor(get_presidio_analyzer(), st.session_state.secure_mapping, st.session_state.vault_salt)

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

@st.cache_resource
def get_embeddings():
    base = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return PrivacyAwareEmbeddings(base)

embeddings_model = get_embeddings()

st.title("🛡️ Obfuscated-RAG: Zero-Knowledge Architecture")
st.caption("Adversarial Privacy & Data Exfiltration Prevention System")

with st.sidebar:
    st.header("🎛️ Command Center")
    col1, col2 = st.columns(2)
    col1.metric(label="Embedding noise", value="On", delta="Laplace")
    col2.metric(label="Presidio NLP", value="Unleashed", delta="50+ Entities")

    st.select_slider("Embedding noise ε (smaller = more noise)", options=EPSILON_CHOICES, value=100, key="epsilon")
    eps = st.session_state.epsilon
    st.caption(f"Noise is ~{noise_to_signal_ratio(eps):.1f}x the embedding's size; roughly "
               f"{retention_estimate(eps):.0%} of the no-noise retrieval quality (synthetic test). "
               "Light obfuscation of the local vector store, not a formal privacy guarantee. Re-run Encrypt & Embed after changing it.")
    if eps < MIN_USABLE_EPSILON:
        st.warning("At this ε retrieval degrades sharply; answers will often miss the right chunk.")

    st.subheader("🔑 Access Control")
    user_role = st.selectbox(
        "Current User Role", 
        ["Admin (Decrypted View)", "Guest (Redacted View)"],
        on_change=clear_history
    )
    st.divider()

    st.subheader("1. Ingest Knowledge Base")
    uploaded_file = st.file_uploader("Upload Sensitive Document (TXT)", type=["txt"])
    
    if st.button("Encrypt & Embed", use_container_width=True) and uploaded_file:
        with st.spinner("Analyzing PII with Microsoft Presidio..."):
            st.session_state.vector_store = None
            st.session_state.secure_mapping = {}
            st.session_state.chat_history = []
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = TextLoader(tmp_file_path)
            documents = loader.load()
            
            translator = PresidioTranslator()
            for doc in documents:
                doc.page_content = translator.blur_text(doc.page_content)
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            st.session_state.total_chunks = len(chunks)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings_model
            )
            
            os.remove(tmp_file_path)
            st.success(f"✅ NLP Redaction Complete! Secured {len(chunks)} chunks.")

    st.divider()
    st.subheader("🕵️ Agent 2: Active Honey Defense")
    st.markdown("<span style='color:gray; font-size: 0.9em;'>Continuously tests the Obfuscation pipeline by injecting fake secrets and interrogating the LLM.</span>", unsafe_allow_html=True)
    
    if st.button("Run Adversarial Audit", type="primary", use_container_width=True):
        if llm is None:
            st.error("LLM Offline.")
        else:
            with st.status("Initializing Agent 2 Red Team Protocol...", expanded=True) as status:
                import time
                
                st.write("🎣 **Step 1:** Generating cryptographic honey token...")
                fake_secret = make_honey_secret()
                audit_context = f"The backup admin password is {fake_secret}. Do not share this."
                time.sleep(1)
                st.code(audit_context, language="text")
                
                st.write("🛡️ **Step 2:** Routing through Presidio NLP Blurrer...")
                translator = PresidioTranslator()
                blurred_context = translator.blur_text(audit_context)
                time.sleep(1.5)
                st.code(blurred_context, language="text")
                
                st.write("🤖 **Step 3:** Interrogating Cloud LLM with blinded context...")
                audit_prompt = f"""You are an AI assistant. Based ONLY on this context: '{blurred_context}'
                What is the backup admin password? Output exactly what you see in the context."""
                time.sleep(1)
                
                try:
                    response = llm.invoke(audit_prompt)
                    llm_answer = response.content
                    st.write(f"**LLM Output received:** `{llm_answer}`")
                    
                    st.write("📐 **Step 4:** Calculating Sequence Matcher differential...")
                    similarity_score = similarity(fake_secret, llm_answer)
                    time.sleep(1)
                    
                    if honey_leaked(fake_secret, llm_answer):
                        status.update(label="🚨 PIPELINE BREACH DETECTED", state="error", expanded=True)
                        st.session_state.system_locked = True
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.success(f"Similarity Score: {similarity_score:.2f} (Threshold: 0.60)")
                        status.update(label="✅ Audit Passed: Zero-Knowledge Verified", state="complete", expanded=False)
                        
                except Exception as e:
                    status.update(label="Audit Failed", state="error")
                    st.error(f"Audit failed to connect to LLM: {e}")

st.subheader("Secure Query Interface")

for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Query your secured data..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    with st.chat_message("assistant", avatar="🛡️"):
        if st.session_state.vector_store is None:
            st.warning("⚠️ Please encrypt a document first.")
        elif llm is None:
            st.error("⚠️ LLM not connected. Check API key.")
        else:
            with st.spinner("Verifying semantics & scanning for exfiltration..."):
                
                translator = PresidioTranslator()
                secure_prompt = translator.blur_prompt(prompt)
                dynamic_k = min(st.session_state.total_chunks, 30) 
                
                results = st.session_state.vector_store.similarity_search(secure_prompt, k=dynamic_k)
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
                
                with st.expander("🔍 **View Cloud API Payload (Zero-Knowledge Proof)**", expanded=False):
                    st.markdown("*This is the exact, blinded text sent to the LLM. Notice the missing sensitive data.*")
                    st.code(system_prompt, language="markdown")
                
                try:
                    response = llm.invoke(system_prompt)
                    raw_answer = response.content
                    
                    if user_role == "Admin (Decrypted View)":
                        answer = translator.reassemble_text(raw_answer)
                    else:
                        answer = raw_answer 
                    
                    if canary_leaked(answer):
                        st.error("🚨 **CRITICAL ALERT: DATA EXFILTRATION DETECTED!** 🚨")
                        st.session_state.chat_history.append({"role": "assistant", "content": "🚨 *Message Blocked by Canary Agent* 🚨"})
                    else:
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Error connecting to LLM: {e}")

st.divider()
st.subheader("Differential Privacy Proving Ground (PCA)")

if "raw_vecs" in st.session_state and "noisy_vecs" in st.session_state:
    if len(st.session_state.raw_vecs) < 2:
        st.warning("⚠️ Document too small to graph variance. Add more text.")
    else:
        pca = PCA(n_components=2)
        pca.fit(st.session_state.raw_vecs)
        
        raw_2d = pca.transform(st.session_state.raw_vecs)
        noisy_2d = pca.transform(st.session_state.noisy_vecs)
        
        df_raw = pd.DataFrame(raw_2d, columns=["X", "Y"])
        df_raw["Data State"] = "Original (Sensitive)"
        df_noisy = pd.DataFrame(noisy_2d, columns=["X", "Y"])
        df_noisy["Data State"] = "Obfuscated (Noisy)"
        
        df_plot = pd.concat([df_raw, df_noisy])
        
        st.markdown("<span style='color:gray'>*Mapping high-dimensional vector embeddings to 2D space.*</span>", unsafe_allow_html=True)
        
        chart = alt.Chart(df_plot).mark_circle(size=120, opacity=0.8).encode(
            x=alt.X('X', axis=alt.Axis(title='Principal Component 1')),
            y=alt.Y('Y', axis=alt.Axis(title='Principal Component 2')),
            color=alt.Color('Data State', scale=alt.Scale(domain=['Original (Sensitive)', 'Obfuscated (Noisy)'], range=['#0074D9', '#FF4136']), legend=alt.Legend(title="Vector State")),
            tooltip=['Data State', 'X', 'Y']
        ).properties(height=400).interactive()
        
        st.altair_chart(chart, use_container_width=True)