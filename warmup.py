"""
Loads the slow parts (LangChain's vector-store code, the embedding model, Presidio with its spaCy model) in a background thread,
so the page can appear at once. get() waits only if something is asked for before it has finished loading.
"""
import threading

_lock = threading.Lock()
_thread = None
_loaded = {}


def _load():
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter      # alone takes ~11 s to import (it pulls in torch)

        from analyzer import build_analyzer
        _loaded["analyzer"] = build_analyzer()
        _loaded["embeddings"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _loaded["Chroma"] = Chroma
        _loaded["Splitter"] = RecursiveCharacterTextSplitter
    except Exception as e:                       # shown when something asks for it, not lost inside the thread
        _loaded["error"] = e


def start():
    global _thread
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_load, daemon=True)
            _thread.start()


def ready():
    return _thread is not None and not _thread.is_alive() and "error" not in _loaded


def get(name):
    start()
    _thread.join()
    if "error" in _loaded:
        raise RuntimeError(f"Could not load the models: {_loaded['error']}")
    return _loaded[name]
