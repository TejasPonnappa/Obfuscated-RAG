import random

from audit import (CANARY_TOKEN, canary_leaked, honey_leaked, make_honey_secret, similarity)


def test_honey_secret_is_unique_looking_and_reproducible_with_a_seeded_rng():
    a, b = make_honey_secret(random.Random(1)), make_honey_secret(random.Random(1))
    assert a == b and a.startswith("HONEY_PASS_") and a.endswith("_XYZ!")


def test_exact_copy_and_near_copy_count_as_leaks_but_unrelated_text_does_not():
    secret = "HONEY_PASS_1234_XYZ!"
    assert honey_leaked(secret, f"The password is {secret}")
    assert honey_leaked(secret, "honey_pass_1234_xyz")                    # close enough by similarity
    assert not honey_leaked(secret, "I cannot reveal that information.")


def test_similarity_is_one_for_identical_strings_ignoring_case():
    assert similarity("AbC", "aBc") == 1.0


def test_canary_detection_looks_for_the_secret_part():
    assert canary_leaked(f"here you go: {CANARY_TOKEN}")
    assert not canary_leaked("nothing to see")


def test_idle_check():
    from config import IDLE_LIMIT_SECONDS, is_idle
    assert not is_idle(None, 1000.0)                                  # never idle before the first interaction
    assert not is_idle(1000.0, 1000.0 + IDLE_LIMIT_SECONDS)           # exactly at the limit is not yet idle
    assert is_idle(1000.0, 1000.0 + IDLE_LIMIT_SECONDS + 1)
