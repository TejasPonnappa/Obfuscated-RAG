"""
Privacy versus retrieval quality.

    python -m eval.privacy_utility [--people 40]

Each person appears in three chunks (profile, payroll, review). A question such as "How much is X paid per year?" has
exactly one right chunk. We redact, embed, add Laplace noise to the chunk embeddings at several epsilons (smaller epsilon
= more noise), and measure how often the right chunk is in the top k by L2 distance (what Chroma does by default).
Also compares the app's old token scheme (a new random token per mention) with one deterministic token per value.
"""
import argparse
import json
import os
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer

from analyzer import build_analyzer
from eval.corpus import make_people, make_queries, make_records
from noise import DifferentialPrivacyAgent, optimize_privacy_budget
from redaction import Redactor

KS = (1, 3, 5, 10)


class RandomTokenRedactor(Redactor):
    """The app's behaviour before the fix: every mention of a value gets its own random token."""

    def token_for(self, entity_type, value):
        token = f"[{entity_type}_{uuid.uuid4().hex[:4].upper()}]"
        self.mapping[token] = value
        return token


class SequentialTokenRedactor(Redactor):
    """Short readable placeholders: [PERSON_1], [PERSON_2]... one per distinct value (deterministic within a session)."""

    def token_for(self, entity_type, value):
        for token, real in self.mapping.items():
            if real == value and token.startswith(f"[{entity_type}_"):
                return token
        n = 1 + sum(1 for t in self.mapping if t.startswith(f"[{entity_type}_"))
        token = f"[{entity_type}_{n}]"
        self.mapping[token] = value
        return token


def recall_at_k(chunk_vecs, query_vecs, gold, ks=KS):
    d = ((query_vecs[:, None, :] - chunk_vecs[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d, axis=1)
    return {k: float(np.mean([gold[i] in order[i, :k] for i in range(len(gold))])) for k in ks}


def embed_corpus(model, redactor_cls, records, queries, people, analyzer):
    redactor = redactor_cls(analyzer, {}, salt=b"eval")
    chunks = [redactor.blur_text(r["text"]) for r in records]
    questions = [redactor.blur_prompt(q) for q, _, _ in queries]
    gold = [next(j for j, r in enumerate(records) if r["person"] == person and r["kind"] == kind)
            for _, person, kind in queries]
    return model.encode(chunks), model.encode(questions), gold


def run_one(model, analyzer, n_people, corpus_seed, noise_seeds=3):
    """One corpus: {scheme: {label: {mode: {k: recall}}}} plus the epsilons the app's loop picks."""
    people = make_people(n_people, corpus_seed)
    records, queries = make_records(people), make_queries(people)
    out = {}
    for scheme, cls in (("deterministic_tokens", Redactor), ("random_tokens_old", RandomTokenRedactor),
                        ("sequential_tokens", SequentialTokenRedactor)):
        chunks, questions, gold = embed_corpus(model, cls, records, queries, people, analyzer)
        rows = {"no_noise": {"queries_plain": recall_at_k(chunks, questions, gold)}}
        loop_eps = [optimize_privacy_budget(chunks)[0] for _ in range(3)]
        rows["loop_epsilon"] = float(np.mean(loop_eps))
        for label, eps in [(f"eps_{e}", e) for e in (20, 10, 5, 2)] + [("eps_chosen_by_loop", float(np.mean(loop_eps)))]:
            agent = DifferentialPrivacyAgent(epsilon=eps)
            per_seed = []
            for s in range(noise_seeds):
                np.random.seed(s)
                per_seed.append(recall_at_k(agent.inject_laplace_noise(chunks), questions, gold))
            rows[label] = {"queries_plain": {k: float(np.mean([r[k] for r in per_seed])) for k in KS},
                           "noise_norm_over_signal_norm": float(
                               np.linalg.norm(agent.inject_laplace_noise(chunks) - chunks, axis=1).mean()
                               / np.linalg.norm(chunks, axis=1).mean())}
        out[scheme] = rows
    return out, len(records), len(queries)


def run(n_people=40, corpus_seeds=(0, 1, 2, 3)):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    analyzer = build_analyzer()
    runs = [run_one(model, analyzer, n_people, cs) for cs in corpus_seeds]
    n_chunks, n_queries = runs[0][1], runs[0][2]
    summary = dict(people=n_people, corpus_seeds=list(corpus_seeds), chunks=n_chunks, queries=n_queries,
                   random_baseline={k: k / n_chunks for k in KS})
    for scheme in ("deterministic_tokens", "random_tokens_old", "sequential_tokens"):
        summary[scheme] = {}
        for label in runs[0][0][scheme]:
            if label == "loop_epsilon":
                values = [r[0][scheme][label] for r in runs]
                summary[scheme][label] = dict(mean=float(np.mean(values)), per_corpus=values)
                continue
            summary[scheme][label] = {
                k: dict(mean=float(np.mean([r[0][scheme][label]["queries_plain"][k] for r in runs])),
                        std=float(np.std([r[0][scheme][label]["queries_plain"][k] for r in runs]))) for k in KS}
            if label != "no_noise":
                summary[scheme][label]["noise_norm_over_signal_norm"] = float(
                    np.mean([r[0][scheme][label]["noise_norm_over_signal_norm"] for r in runs]))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", type=int, default=40)
    args = parser.parse_args()
    result = run(args.people)
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/privacy_utility.json", "w") as f:
        json.dump(result, f, indent=2)
    b = result["random_baseline"]
    print(f"{result['chunks']} chunks, {result['queries']} queries, {len(result['corpus_seeds'])} corpora; "
          f"random-guess recall@1/@5/@10 = {b[1]:.3f} / {b[5]:.3f} / {b[10]:.3f}")
    for scheme in ("deterministic_tokens", "random_tokens_old", "sequential_tokens"):
        print(f"\n{scheme}: recall@1 / @5 / @10, mean over corpora (+/- std of @5); queries are NOT noised, as in the app")
        for label, row in result[scheme].items():
            if label == "loop_epsilon":
                print(f"  epsilon the app's loop picks (mean): {row['mean']:.2f}")
                continue
            extra = "" if label == "no_noise" else f"   noise/signal {row['noise_norm_over_signal_norm']:.1f}x"
            print(f"  {label:20s} {row[1]['mean']:.2f} / {row[5]['mean']:.2f} / {row[10]['mean']:.2f}  (+/- {row[5]['std']:.2f}){extra}")
