"""
How well does the redaction step hide sensitive values?

    python -m eval.pii_detection [--people 60]

Reports, per kind of value, how often it survives redaction verbatim (a leak), and which entity types Presidio
produced. Ground truth comes from the synthetic corpus, so the numbers are exact for this corpus only.
"""
import argparse
import json
import os
from collections import Counter

from analyzer import build_analyzer
from eval.corpus import make_people, make_records
from redaction import Redactor

KINDS = ("name", "email", "phone", "salary", "password_plain", "password_natural")


def run(n_people=60, seed=0):
    people = make_people(n_people, seed)
    records = make_records(people)
    redactor = Redactor(build_analyzer(), {}, salt=b"eval")
    leaks, totals, entity_counts = Counter(), Counter(), Counter()
    for rec in records:
        blurred = redactor.blur_text(rec["text"])
        for value, kind in rec["sensitive"]:
            totals[kind] += 1
            if value.lower() in blurred.lower():
                leaks[kind] += 1
        entity_counts.update(t.split("_")[0] if t.count("_") == 1 else t[1:].rsplit("_", 1)[0]
                             for t in redactor.mapping if t in blurred)
    return dict(
        people=n_people, records=len(records),
        leak_rate={k: leaks[k] / totals[k] for k in KINDS if totals[k]},
        leaked={k: leaks[k] for k in KINDS if totals[k]}, checked={k: totals[k] for k in KINDS if totals[k]},
        overall_leak_rate=sum(leaks.values()) / sum(totals.values()),
        entity_types_produced=dict(entity_counts.most_common()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", type=int, default=60)
    args = parser.parse_args()
    result = run(args.people)
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/pii_detection.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
