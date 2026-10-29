"""
What does each improvement add?   python -m eval.ablation [--tab-docs 40]

Four configurations, each adding one thing to the one before:
  plain Presidio            Presidio's own detectors, the earliest overlapping span wins, no name propagation
  + overlap by confidence   the most confident overlapping span wins and weaker ones are cut back
  + custom rules            money, street addresses, passwords and names after a label
  + name propagation        a name found once is hidden everywhere it appears
Measured on (1) the three fictional sample documents and (2) a random sample of TAB court cases, a public human-annotated
anonymization corpus (download echr_test.json from github.com/NorskRegnesentral/text-anonymisation-benchmark into eval/data/).

A value counts as hidden if its text no longer appears in the redacted document. This is strict: a string that is hidden in one
place but left visible in another counts as visible. TAB's masked categories include organisations, places and dates, which a
personal-data redactor is not built to hide, so the per-type table matters more than the overall figure.
"""
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict

from analyzer import build_analyzer
from eval.sample_check import DOCS
from redaction import Redactor

TAB_PATH = "eval/data/echr_test.json"
VARIANTS = [                                   # (label, custom rules, overlap rule, propagate names)
    ("plain Presidio", False, "first", False),
    ("+ overlap by confidence", False, "confidence", False),
    ("+ custom rules", True, "confidence", False),
    ("+ name propagation", True, "confidence", True),
]


class Cached:
    """Returns analysis results computed once, so the four variants differ only in how they use them."""

    def __init__(self):
        self.results = {}

    def analyze(self, text, language="en"):
        return self.results[text]


def redact_variants(text, engines):
    """{label: redacted text} for all variants; each engine analyses the text once."""
    analysed = {custom: engines[custom].analyze(text=text, language="en") for custom in (False, True)}
    out = {}
    for label, custom, overlap, propagate in VARIANTS:
        cache = Cached()
        cache.results[text] = analysed[custom]
        out[label] = Redactor(cache, {}, b"ablation", propagate_names=propagate, overlap=overlap).blur_text(text)
    return out


def visible(value, text):
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I) is not None


def run_samples(engines):
    totals = {label: [0, 0] for label, *_ in VARIANTS}
    for path, truth in DOCS.items():
        text = open(path, encoding="utf-8").read()
        for label, out in redact_variants(text, engines).items():
            for values in truth.values():
                for v in values:
                    totals[label][1] += 1
                    totals[label][0] += v.lower() not in out.lower()
    return {label: dict(hidden=h, total=t, pct=h / t) for label, (h, t) in totals.items()}


def run_tab(engines, n_docs, seed=0):
    docs = json.load(open(TAB_PATH, encoding="utf-8"))
    random.Random(seed).shuffle(docs)
    docs = docs[:n_docs]
    stats = {label: defaultdict(lambda: [0, 0]) for label, *_ in VARIANTS}      # key -> [hidden, total]
    start = time.time()
    for i, doc in enumerate(docs, 1):
        text = doc["text"]
        mentions = next(iter(doc["annotations"].values()))["entity_mentions"]
        masked, unmasked = {}, set()
        for m in mentions:
            s = m["span_text"].strip()
            if len(s) < 2:
                continue
            if m["identifier_type"] in ("DIRECT", "QUASI"):
                masked[s] = (m["entity_type"], m["identifier_type"])
            else:
                unmasked.add(s)
        unmasked -= set(masked)
        for label, out in redact_variants(text, engines).items():
            for s, (etype, itype) in masked.items():
                hidden = not visible(s, out)
                for key in ("all masked", f"type {etype}", f"{itype} (all types)"):
                    stats[label][key][1] += 1
                    stats[label][key][0] += hidden
            for s in unmasked:
                stats[label]["NO_MASK values hidden (over-redaction)"][1] += 1
                stats[label]["NO_MASK values hidden (over-redaction)"][0] += not visible(s, out)
        if i % 5 == 0:
            print(f"  TAB {i}/{len(docs)} documents ({time.time() - start:.0f}s)", flush=True)
    return {label: {k: dict(hidden=h, total=t, pct=h / t if t else None) for k, (h, t) in s.items()} for label, s in stats.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tab-docs", type=int, default=40)
    args = parser.parse_args()
    engines = {False: build_analyzer(custom_rules=False), True: build_analyzer(custom_rules=True)}
    result = dict(samples=run_samples(engines))
    print("\nSample documents (3 fictional, 92 values): share of sensitive values hidden")
    for label, r in result["samples"].items():
        print(f"  {label:26s} {r['hidden']:3d}/{r['total']}  {r['pct']:.0%}")
    if os.path.exists(TAB_PATH):
        result["tab"] = run_tab(engines, args.tab_docs)
        result["tab_docs"] = args.tab_docs
        print(f"\nTAB court cases ({args.tab_docs} random test documents): share of masked identifiers hidden, strict value match")
        keys = ["all masked", "DIRECT (all types)", "QUASI (all types)", "type PERSON", "type CODE", "type DATETIME", "type LOC", "type ORG",
                "NO_MASK values hidden (over-redaction)"]
        print(f"  {'':40s}" + "".join(f"{label[:20]:>22s}" for label, *_ in VARIANTS))
        for k in keys:
            row = [result["tab"][label].get(k) for label, *_ in VARIANTS]
            if any(row):
                print(f"  {k:40s}" + "".join(f"{(r['pct'] if r else 0):>21.0%} " for r in row) + f"   (n={row[0]['total'] if row[0] else 0})")
    os.makedirs("eval/results", exist_ok=True)
    json.dump(result, open("eval/results/ablation.json", "w"), indent=2)
