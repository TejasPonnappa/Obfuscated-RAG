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


def run(n_people=40, seeds=3):
    people = make_people(n_people)
    records, queries = make_records(people), make_queries(people)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    analyzer = build_analyzer()
    out = dict(people=n_people, chunks=len(records), queries=len(queries),
               random_baseline={k: k / len(records) for k in KS})
    for scheme, cls in (("deterministic_tokens", Redactor), ("random_tokens_old", RandomTokenRedactor)):
        chunks, questions, gold = embed_corpus(model, cls, records, queries, people, analyzer)
        rows = {"no_noise": recall_at_k(chunks, questions, gold)}
        loop_eps = [optimize_privacy_budget(chunks)[0] for _ in range(3)]
        for eps in (20, 10, 5, 2, 1, 0.5, round(float(np.mean(loop_eps)), 3)):
            agent = DifferentialPrivacyAgent(epsilon=eps)
            per_seed = {"queries_plain": [], "queries_noised": []}
            for s in range(seeds):
                np.random.seed(s)
                noisy_chunks = agent.inject_laplace_noise(chunks)
                per_seed["queries_plain"].append(recall_at_k(noisy_chunks, questions, gold))
                per_seed["queries_noised"].append(recall_at_k(noisy_chunks, agent.inject_laplace_noise(questions), gold))
            label = f"eps_{eps}" + ("_chosen_by_loop" if eps == round(float(np.mean(loop_eps)), 3) else "")
            rows[label] = {mode: {k: float(np.mean([r[k] for r in lst])) for k in KS} for mode, lst in per_seed.items()}
            rows[label]["noise_norm_over_signal_norm"] = float(
                np.linalg.norm(agent.inject_laplace_noise(chunks) - chunks, axis=1).mean() / np.linalg.norm(chunks, axis=1).mean())
        out[scheme] = rows
        out[scheme]["epsilon_chosen_by_loop"] = loop_eps
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", type=int, default=40)
    args = parser.parse_args()
    result = run(args.people)
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/privacy_utility.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"chunks {result['chunks']}, queries {result['queries']}, random baseline recall@5 = {result['random_baseline'][5]:.3f}")
    for scheme in ("deterministic_tokens", "random_tokens_old"):
        print(f"\n{scheme}  (recall@1 / @5 / @10)")
        for label, row in result[scheme].items():
            if label == "epsilon_chosen_by_loop":
                print(f"  epsilon picked by the app's loop on this data: {[round(e, 3) for e in row]}")
            elif label == "no_noise":
                print(f"  {label:26s} {row[1]:.2f} / {row[5]:.2f} / {row[10]:.2f}")
            else:
                for mode in ("queries_plain", "queries_noised"):
                    r = row[mode]
                    print(f"  {label:26s} {mode:15s} {r[1]:.2f} / {r[5]:.2f} / {r[10]:.2f}   (noise/signal {row['noise_norm_over_signal_norm']:.1f}x)")
