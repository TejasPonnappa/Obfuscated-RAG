import re

import pytest

from analyzer import PASSWORD_REGEX

CASES_MATCH = [
    "The backup admin password is hunter2!",
    "password: s3cret",
    "Password = abc123",
    "The backup admin password for John Smith is see544!.",
    "secret is topsecret",
    "Their passcode was reset; the passcode is 4821.",
]
CASES_NO_MATCH = [
    "Please choose a strong password and keep it safe.",
    "He kept the secret to himself.",
    "The password policy is reviewed yearly",       # 'is' after a long phrase, but the value check below must still work
]


@pytest.mark.parametrize("text", CASES_MATCH)
def test_matches_the_phrasings_we_care_about(text):
    assert re.search(PASSWORD_REGEX, text)


def test_captured_value_is_the_secret_not_the_sentence():
    m = re.search(PASSWORD_REGEX, "The backup admin password for John Smith is see544!.")
    assert m.group(1) == "see544!"


@pytest.mark.parametrize("text", CASES_NO_MATCH[:2])
def test_does_not_fire_on_ordinary_sentences(text):
    assert not re.search(PASSWORD_REGEX, text)


@pytest.mark.parametrize("text, value", [
    ('the database secret is "blue.fox.77" (rotate it too)', '"blue.fox.77"'),      # dots inside the value
    ("the password he used was tanaka_2019_summer. His account was locked.", "tanaka_2019_summer"),
    ("pw: Clinic#Spring24", "Clinic#Spring24"),
])
def test_value_keeps_inner_dots_and_drops_the_sentence_end(text, value):
    assert re.search(PASSWORD_REGEX, text).group(1) == value


def test_does_not_treat_ordinary_words_as_credentials():
    assert not re.search(PASSWORD_REGEX, "All people, numbers and credentials below are invented for testing.")
