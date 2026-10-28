"""
An optional extra detector: GLiNER, a small zero-shot entity model that runs locally (the weights are downloaded once;
afterwards it needs no network, e.g. set HF_HUB_OFFLINE=1). It is added to Presidio's findings, not a replacement for them.
"""
import os
import time
from types import SimpleNamespace

DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"
THRESHOLD = 0.5
WINDOW_CHARS = 1000                 # GLiNER reads about 384 tokens at a time, so long text is cut at line breaks

# what we ask the model to find -> the entity type the redactor uses for the token
LABELS = {
    "person name": "PERSON",
    "email address": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "salary": "SALARY",
    "street address": "ADDRESS",
    "postal code": "ADDRESS",
    "password": "PASSWORD",
    "credit card number": "CREDIT_CARD",
    "social security number": "NATIONAL_ID",
    "date of birth": "DATE_OF_BIRTH",
    "bank account number": "BANK_ACCOUNT",
    "identification number": "ID_NUMBER",
}


def windows(text, size=WINDOW_CHARS):
    """(offset, piece) pairs that end at line breaks where possible."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            end = cut + 1 if cut > start else end
        yield start, text[start:end]
        start = end


def _load(GLiNER, model_name):
    """Load from the local cache with HuggingFace's offline mode on (the tokenizer would otherwise make a metadata lookup);
    only if that fails (first use) allow the one-time download."""
    import huggingface_hub.constants as hub
    was_offline, env_before = hub.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE")
    hub.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return GLiNER.from_pretrained(model_name, local_files_only=True), True
    except Exception:
        pass
    finally:
        hub.HF_HUB_OFFLINE = was_offline
        if env_before is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = env_before
    return GLiNER.from_pretrained(model_name), False


class GlinerAnalyzer:
    """Same analyze() interface as Presidio's engine: its results plus GLiNER's."""

    def __init__(self, base, model_name=DEFAULT_MODEL, threshold=THRESHOLD):
        from gliner import GLiNER                    # imported here so the app and CI work without it
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self.base = base
        self.model, self.loaded_offline = _load(GLiNER, model_name)
        self.threshold = threshold
        self.seconds = 0.0

    def analyze(self, text, language="en"):
        results = list(self.base.analyze(text=text, language=language))
        t0 = time.time()
        for offset, piece in windows(text):
            for hit in self.model.predict_entities(piece, list(LABELS), threshold=self.threshold):
                if not any(c.isalnum() for c in hit["text"]):          # separators such as "===" are not values
                    continue
                results.append(SimpleNamespace(start=offset + hit["start"], end=offset + hit["end"],
                                               entity_type=LABELS[hit["label"]], score=float(hit["score"])))
        self.seconds += time.time() - t0
        return results
