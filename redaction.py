"""
PII redaction with reversible tokens, independent of Streamlit so it can be tested.

The analyzer is injected (Presidio in the app, a fake in tests); the token -> real value table lives in a plain dict the
caller owns. Tokens are deterministic per (entity type, value) within a session: the same value always gets the same
token, so the LLM can tell two mentions are the same entity. They are keyed with a per-session secret salt, so a token
cannot be guessed from the value.
"""
import hashlib
import hmac
import re


class Redactor:
    def __init__(self, analyzer, mapping, salt):
        """analyzer: object with analyze(text=..., language=...) -> results with start, end, entity_type.
        mapping: dict token -> real value (mutated). salt: bytes."""
        self.analyzer = analyzer
        self.mapping = mapping
        self.salt = salt

    def token_for(self, entity_type, value):
        """Deterministic token for a value; lengthens the hash on the rare collision with a different value."""
        digest = hmac.new(self.salt, f"{entity_type}|{value}".encode("utf-8"), hashlib.sha256).hexdigest().upper()
        for length in range(4, len(digest) + 1):
            token = f"[{entity_type}_{digest[:length]}]"
            if self.mapping.get(token, value) == value:
                self.mapping[token] = value
                return token
        raise RuntimeError("could not allocate a unique token")

    def blur_text(self, text):
        """Replace every detected entity with its token (longest non-overlapping match first)."""
        results = self.analyzer.analyze(text=text, language="en")
        ordered = sorted(results, key=lambda r: (r.start, -(r.end - r.start)))
        kept, last_end = [], -1
        for r in ordered:
            if r.start >= last_end:
                kept.append(r)
                last_end = r.end
        out = text
        for r in sorted(kept, key=lambda r: r.start, reverse=True):
            out = out[:r.start] + self.token_for(r.entity_type, text[r.start:r.end]) + out[r.end:]
        return out

    def blur_prompt(self, prompt_text):
        """Swap any known real value (or, for multi-word values, a long word of it) in a user prompt for its token."""
        blurred = prompt_text
        for token, real_value in sorted(self.mapping.items(), key=lambda item: len(item[1]), reverse=True):
            pattern = re.compile(re.escape(real_value), re.IGNORECASE)
            if pattern.search(blurred):
                blurred = pattern.sub(token, blurred)
                continue
            parts = real_value.split()
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 3:
                        part_pattern = re.compile(r"\b" + re.escape(part) + r"\b", re.IGNORECASE)
                        if part_pattern.search(blurred):
                            blurred = part_pattern.sub(token, blurred)
        return blurred

    def reassemble_text(self, llm_response):
        """Put real values back for display (bold), and undo markdown escaping of brackets around tokens."""
        final_text = llm_response.replace("\\[", "[").replace("\\]", "]")
        for token, real_value in self.mapping.items():
            if token in final_text:
                final_text = final_text.replace(token, f"**{real_value}**")
        return final_text
