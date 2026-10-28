import re

import pytest

from analyzer import ADDRESS_REGEX, SALARY_REGEX

FLAGS = re.I | re.S | re.M          # the flags Presidio compiles pattern recognizers with


@pytest.mark.parametrize("text, found", [
    ("a bonus of $5,000.", "$5,000"),
    ("moves to $104k plus", "$104k"),
    ("base of USD 91,500 a year", "USD 91,500"),
    ("raise to 82,000 USD", "82,000 USD"),
    ("earns 78 thousand dollars", "78 thousand dollars"),
    ("$7,625 a month", "$7,625"),
])
def test_money_in_the_common_written_forms(text, found):
    assert re.search(SALARY_REGEX, text, FLAGS).group(0) == found


@pytest.mark.parametrize("text, found", [
    ("home address 2218 Alder Street, Oakland", "2218 Alder Street"),
    ("lives at 14 Maple Avenue, Apt 3B, Boston", "14 Maple Avenue, Apt 3B"),
])
def test_street_addresses(text, found):
    assert re.search(ADDRESS_REGEX, text, FLAGS).group(0) == found


def test_plain_numbers_are_not_money_or_addresses():
    assert not re.search(SALARY_REGEX, "ticket 4471 was opened at 03:12", FLAGS)
    assert not re.search(ADDRESS_REGEX, "ticket 4471 was opened", FLAGS)
