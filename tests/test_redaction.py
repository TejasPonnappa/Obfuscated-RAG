import re
from types import SimpleNamespace

import pytest

from redaction import Redactor


class FakeAnalyzer:
    """Finds exact substrings listed per entity type, like a tiny stand-in for Presidio."""

    def __init__(self, entities):
        self.entities = entities                      # {"PERSON": ["John Smith"], "SALARY": ["$85,000"]}

    def analyze(self, text, language="en"):
        out = []
        for entity_type, values in self.entities.items():
            for value in values:
                for m in re.finditer(re.escape(value), text):
                    out.append(SimpleNamespace(start=m.start(), end=m.end(), entity_type=entity_type))
        return out


@pytest.fixture
def redactor():
    analyzer = FakeAnalyzer({"PERSON": ["John Smith", "Smith"], "SALARY": ["$85,000"], "EMAIL_ADDRESS": ["j@x.com"]})
    return Redactor(analyzer, {}, salt=b"unit-test-salt")


TEXT = "John Smith earns $85,000. John Smith can be reached at j@x.com."


def test_blur_then_reassemble_restores_every_real_value(redactor):
    blurred = redactor.blur_text(TEXT)
    for real in ("John Smith", "$85,000", "j@x.com"):
        assert real not in blurred                    # nothing sensitive survives
    restored = redactor.reassemble_text(blurred)
    assert restored == TEXT.replace("John Smith", "**John Smith**").replace("$85,000", "**$85,000**") \
        .replace("j@x.com", "**j@x.com**")


def test_the_same_value_always_gets_the_same_token(redactor):
    blurred = redactor.blur_text(TEXT)
    person_tokens = re.findall(r"\[PERSON_[0-9A-F]+\]", blurred)
    assert len(person_tokens) == 2 and person_tokens[0] == person_tokens[1]


def test_different_values_get_different_tokens_and_the_format_is_kept(redactor):
    blurred = redactor.blur_text(TEXT)
    tokens = set(re.findall(r"\[[A-Z_]+_[0-9A-F]{4,}\]", blurred))
    assert len(tokens) == 3
    assert any(t.startswith("[SALARY_") for t in tokens)


def test_tokens_depend_on_the_session_salt():
    a = Redactor(FakeAnalyzer({"SALARY": ["$1"]}), {}, salt=b"one").token_for("SALARY", "$1")
    b = Redactor(FakeAnalyzer({"SALARY": ["$1"]}), {}, salt=b"two").token_for("SALARY", "$1")
    assert a != b


def test_overlapping_detections_keep_the_longest_match_that_starts_first(redactor):
    blurred = redactor.blur_text("John Smith")         # "John Smith" and the inner "Smith" both match
    assert blurred.count("[") == 1 and "Smith" not in blurred


def test_blur_prompt_replaces_known_values_and_name_parts(redactor):
    redactor.blur_text(TEXT)
    assert "[PERSON_" in redactor.blur_prompt("what does john smith earn?")      # case-insensitive full value
    assert "Smith" not in redactor.blur_prompt("is Smith paid well?")           # a long word of a multi-word value


def test_blur_prompt_leaves_unrelated_text_alone(redactor):
    redactor.blur_text(TEXT)
    assert redactor.blur_prompt("what is the capital of France?") == "what is the capital of France?"


def test_reassemble_undoes_markdown_escaped_brackets(redactor):
    token = redactor.token_for("SALARY", "$85,000")
    escaped = token.replace("[", r"\[").replace("]", r"\]")
    assert redactor.reassemble_text(f"They get {escaped}.") == "They get **$85,000**."


def test_a_name_span_that_swallowed_the_next_word_is_trimmed():
    analyzer = FakeAnalyzer({"PERSON": ["Norma Fisher works"]})        # what the NER model really returned
    r = Redactor(analyzer, {}, salt=b"s")
    blurred = r.blur_text("Employee Norma Fisher works in Finance.")
    assert blurred.endswith(" works in Finance.")                      # the verb is left alone
    assert list(r.mapping.values()) == ["Norma Fisher"]


def test_the_whole_name_in_a_question_beats_a_longer_value_sharing_its_words():
    r = Redactor(FakeAnalyzer({}), {"[PERSON_AAAA]": "Norma Fisher", "[PERSON_BBBB]": "Norma Fisher works"}, salt=b"s")
    assert r.blur_prompt("How much is Norma Fisher paid?") == "How much is [PERSON_AAAA] paid?"
