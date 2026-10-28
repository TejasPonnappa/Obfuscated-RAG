"""The Presidio analyzer used by the app and the evaluation scripts (built once, outside Streamlit)."""

# "password is X", "password: X", "pw: X", "the password he used was X" and "password for <someone> is X"
# (up to 40 characters between the keyword and the verb). The value may contain dots ("blue.fox.77"); a final dot ends the sentence.
PASSWORD_REGEX = r"(?i)\b(?:password|passcode|passwd|pwd|pw|secret)\b[^.\n:=]{0,40}?(?:[:=]|->|\b(?:is|was)\b)\s*([^\s,]*[^\s,.])"

# money in the usual written forms: $104k, $97,250, USD 91,500, 82,000 USD, 78 thousand dollars
_AMOUNT = r"\d(?:[\d,]*\d)?(?:\.\d+)?"
SALARY_REGEX = (rf"\$\s?{_AMOUNT}(?:[kKmM]\b)?"
                rf"|\b(?:USD|EUR|GBP)\s?{_AMOUNT}(?:[kKmM]\b)?"
                rf"|\b{_AMOUNT}\s?(?:thousand|million|k|m)?\s?(?:USD|EUR|GBP|dollars|euros|pounds)\b")

# "2218 Alder Street", "14 Maple Avenue, Apt 3B"
ADDRESS_REGEX = (r"\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b"
                 r"(?:,?\s*(?:Apt|Apartment|Suite|Unit|#)\.?\s*\w+)?")

# Presidio's NER model misses names in very short sentences ("Review of Norma Fisher: ..."), so names that follow a label are
# matched by context. These labels come from the synthetic HR corpus; real documents will need their own.
_NAME = r"[A-Z][a-z]+(?:[ ][A-Z][a-z'-]+){1,2}"
NAME_AFTER_LABEL_REGEXES = [rf"(?<={label}[ ]){_NAME}" for label in ("Review of", "Employee:", "Employee", "Name:", "Name", "Profile of")]


def build_analyzer(gliner=False):
    """Presidio's default analyzer plus custom recognizers: passwords, money, street addresses and names after a label.
    gliner=True adds the GLiNER model on top (gliner_layer.py)."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer      # imported here so the regexes above need no NLP stack

    engine = AnalyzerEngine()
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="PASSWORD", patterns=[Pattern(name="password_pattern", regex=PASSWORD_REGEX, score=0.9)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="SALARY", patterns=[Pattern(name="salary_pattern", regex=SALARY_REGEX, score=0.9)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="ADDRESS", patterns=[Pattern(name="address_pattern", regex=ADDRESS_REGEX, score=0.8)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="PERSON",
        patterns=[Pattern(name=f"name_after_label_{i}", regex=rx, score=0.85) for i, rx in enumerate(NAME_AFTER_LABEL_REGEXES)]))
    if gliner:
        from gliner_layer import GlinerAnalyzer
        return GlinerAnalyzer(engine)
    return engine
