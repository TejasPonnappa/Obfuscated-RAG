import re

from analyzer import NAME_AFTER_LABEL_REGEXES


def _matches(text):
    return [m.group(0) for rx in NAME_AFTER_LABEL_REGEXES for m in re.finditer(rx, text)]


def test_name_after_review_label_is_matched_without_the_label():
    assert _matches("Review of Norma Fisher: reliable and punctual.") == ["Norma Fisher"]


def test_unlabelled_capitalised_words_are_not_matched():
    assert _matches("The Backup Admin reviewed the Quarterly Report.") == []
