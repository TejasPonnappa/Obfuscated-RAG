"""The Presidio analyzer used by the app and the evaluation scripts (built once, outside Streamlit)."""
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

# "password is X", "password: X" and also "password for <someone> is X" (up to 40 characters between the word and the verb)
PASSWORD_REGEX = r"(?i)\b(?:password|passcode|secret)\b[^.\n:=]{0,40}?(?:[:=]|\bis\b)\s*([^\s\n.,]+)"
SALARY_REGEX = r"\$[\d,]+(?:\.\d{2})?"
# Presidio's NER model misses names in very short sentences ("Review of Norma Fisher: ..."), so names that follow a label are
# matched by context. These labels come from the synthetic HR corpus; real documents will need their own.
_NAME = r"[A-Z][a-z]+(?:[ ][A-Z][a-z'-]+){1,2}"
NAME_AFTER_LABEL_REGEXES = [rf"(?<={label}[ ]){_NAME}" for label in ("Review of", "Employee:", "Employee", "Name:", "Name", "Profile of")]


def build_analyzer():
    """Presidio's default analyzer plus two custom recognizers: passwords and dollar amounts."""
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="PASSWORD", patterns=[Pattern(name="password_pattern", regex=PASSWORD_REGEX, score=0.9)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="SALARY", patterns=[Pattern(name="salary_pattern", regex=SALARY_REGEX, score=0.9)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="PERSON",
        patterns=[Pattern(name=f"name_after_label_{i}", regex=rx, score=0.85) for i, rx in enumerate(NAME_AFTER_LABEL_REGEXES)]))
    return engine
