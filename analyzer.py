"""The Presidio analyzer used by the app and the evaluation scripts (built once, outside Streamlit)."""
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

PASSWORD_REGEX = r"(?i)(?:password|secret)\s*(?:[:=]|\s+is\s+)\s*([^\s\n.,]+)"
SALARY_REGEX = r"\$[\d,]+(?:\.\d{2})?"


def build_analyzer():
    """Presidio's default analyzer plus two custom recognizers: passwords and dollar amounts."""
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="PASSWORD", patterns=[Pattern(name="password_pattern", regex=PASSWORD_REGEX, score=0.9)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="SALARY", patterns=[Pattern(name="salary_pattern", regex=SALARY_REGEX, score=0.9)]))
    return engine
