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
from types import SimpleNamespace


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
        """Replace every detected entity with its token, then hide any name found once everywhere it appears."""
        kept = self._resolve_overlaps(text, self.analyzer.analyze(text=text, language="en"))
        out = text
        for r in sorted(kept, key=lambda r: r.start, reverse=True):
            end = self._trim_person_end(text, r) if r.entity_type == "PERSON" else r.end
            out = out[:r.start] + self.token_for(r.entity_type, text[r.start:end]) + out[end:]
        return self._propagate_names(out)

    @staticmethod
    def _resolve_overlaps(text, results):
        """Most confident detections first; a weaker overlapping span is cut back to the part that overlaps nothing.
        (The name model often returns "Daniel O'Connor - daniel@x.com" as one name; the email must still be found.)"""
        kept = []
        for r in sorted(results, key=lambda r: (-getattr(r, "score", 1.0), -(r.end - r.start), r.start)):
            pieces = [(r.start, r.end)]
            for k in kept:
                free = []
                for s, e in pieces:
                    if e <= k.start or s >= k.end:
                        free.append((s, e))
                    else:
                        if s < k.start:
                            free.append((s, k.start))
                        if e > k.end:
                            free.append((k.end, e))
                pieces = free
            for s, e in pieces:
                while s < e and text[s] in " \t\n-,;:()":
                    s += 1
                while e > s and text[e - 1] in " \t\n-,;:()":
                    e -= 1
                if e - s >= 2:
                    kept.append(SimpleNamespace(start=s, end=e, entity_type=r.entity_type))
        return kept

    _NAME = re.compile(r"[A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'-]+)+")

    def _propagate_names(self, text):
        """A name found once is hidden everywhere: later mentions the detector missed, and first or last names on their own."""
        names = [v for t, v in self.mapping.items() if t.startswith("[PERSON_") and self._NAME.fullmatch(v)]
        parts = sorted({w for v in names for w in v.split(" ") if len(w) >= 3}, key=len, reverse=True)
        for value in sorted(names, key=len, reverse=True) + parts:
            text = re.sub(r"(?<![\w\[])" + re.escape(value) + r"(?![\w\]])", lambda m, v=value: self.token_for("PERSON", v), text)
        return text

    @staticmethod
    def _trim_person_end(text, result):
        """The NER model sometimes swallows the next word ("Norma Fisher works"); a name does not end in a lowercase word."""
        span = text[result.start:result.end]
        words = span.split(" ")
        while len(words) > 1 and words[-1][:1].islower():
            words.pop()
        return result.start + len(" ".join(words))

    def blur_prompt(self, prompt_text):
        """Swap any known real value (or, for multi-word values, a long word of it) in a user prompt for its token."""
        blurred = prompt_text
        by_length = sorted(self.mapping.items(), key=lambda item: len(item[1]), reverse=True)
        unmatched = []
        for token, real_value in by_length:                  # whole values first, so "Norma Fisher" wins over its parts
            pattern = re.compile(re.escape(real_value), re.IGNORECASE)
            if pattern.search(blurred):
                blurred = pattern.sub(token, blurred)
            else:
                unmatched.append((token, real_value))
        for token, real_value in unmatched:
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
