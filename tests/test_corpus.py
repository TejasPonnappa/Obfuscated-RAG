from eval.corpus import make_people, make_queries, make_records


def test_corpus_is_reproducible_and_has_unique_names():
    a, b = make_people(30, seed=1), make_people(30, seed=1)
    assert a == b and len({p["name"] for p in a}) == 30


def test_every_sensitive_value_appears_in_its_own_text():
    for rec in make_records(make_people(20, seed=2)):
        assert rec["sensitive"] and all(value in rec["text"] for value, _ in rec["sensitive"])


def test_every_query_has_exactly_one_answering_record():
    people = make_people(10, seed=3)
    records = make_records(people)
    for _, person, kind in make_queries(people):
        assert sum(r["person"] == person and r["kind"] == kind for r in records) == 1
