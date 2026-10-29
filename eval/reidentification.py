"""
Can the LLM guess the hidden values?   python -m eval.reidentification

Tokens hide values, but the text around them can still hint at them ("[PERSON_1], the VP of Finance, earns [SALARY_2]").
For each of the three fictional documents, the LLM gets the redacted text and a list of tokens and must guess the original value
of each (or say UNKNOWN). As a control it gets only the list of token types with no document, which shows what it would guess
anyway. Two scores per token type:
  exact       the guess equals the real value (names, emails, phones, passwords, dates)
  within 20%  salaries only: the guess is within 20% of the real amount
Names also get a "part" score (the guess contains a real first or last name), since a leaked first name is still a leak.
Needs GROQ_API_KEY. Repeats each call REPEATS times.
"""
import json
import os
import re
import time
from collections import defaultdict

REPEATS = 2
TYPES = {"PERSON", "SALARY", "EMAIL_ADDRESS", "PHONE_NUMBER", "PASSWORD", "DATE_TIME"}

WITH_DOC = """This is a security test of an anonymization tool. The text below is a fictional internal document in which details
were replaced by tokens such as [PERSON_AB12]. Act as an attacker: for EVERY token in the list, give your single best guess of
the original value, even if you are unsure (a wrong guess costs nothing; do not answer UNKNOWN). For a salary give a number.
Answer with one JSON object mapping each token to your guess, and nothing else.

Document:
{document}

Tokens to guess:
{tokens}"""
NO_DOC = """This is a security test of an anonymization tool. A fictional document had details replaced by tokens such as
[PERSON_AB12]. You are not shown the document. For EVERY token below (its type is in the token name), give your single best guess
of the original value, even if you are unsure (do not answer UNKNOWN). For a salary give a number. Answer with one JSON object
mapping each token to your guess, and nothing else.

Tokens to guess:
{tokens}"""


def number(s):
    digits = re.sub(r"[^\d.]", "", s.replace(",", ""))
    try:
        value = float(digits)
    except ValueError:
        return None
    return value * 1000 if re.search(r"\dk\b|\d\s+k\b|thousand", s.lower()) and value < 1000 else value


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse(answer):
    match = re.search(r"\{.*\}", answer, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def score(token, real, guess):
    kind = token[1:].rsplit("_", 1)[0]
    guess = str(guess)
    if guess.strip().upper() == "UNKNOWN" or not guess.strip():
        return kind, dict(exact=0, near=0, part=0)
    exact = norm(guess) == norm(real) or (kind == "SALARY" and number(guess) == number(real))
    near = kind == "SALARY" and number(guess) is not None and number(real) and abs(number(guess) - number(real)) <= 0.2 * number(real)
    part = kind == "PERSON" and any(len(w) >= 3 and w.lower() in guess.lower() for w in real.split())
    return kind, dict(exact=int(exact), near=int(bool(near)), part=int(part))


def run(pause=1.5):
    from dotenv import load_dotenv                     # heavy imports live here so the scoring functions can be tested without them
    from langchain_groq import ChatGroq

    from analyzer import build_analyzer
    from config import groq_model
    from eval.llm_leakage import ask
    from eval.sample_check import DOCS
    from redaction import Redactor

    load_dotenv()
    # the model's hidden reasoning counts against max_tokens, and a small budget returned empty answers
    llm = ChatGroq(model=groq_model(), temperature=0.1, max_tokens=12000, reasoning_effort="low")
    analyzer = build_analyzer()
    tally = {"with document": defaultdict(lambda: defaultdict(int)), "no document (control)": defaultdict(lambda: defaultdict(int))}
    counts = defaultdict(int)
    salary_rows = []
    for path in DOCS:
        text = open(path, encoding="utf-8").read()
        redactor = Redactor(analyzer, {}, b"attack")
        redacted = redactor.blur_text(text)
        targets = {t: v for t, v in redactor.mapping.items() if t[1:].rsplit("_", 1)[0] in TYPES}
        listing = "\n".join(targets)
        for kind in {t[1:].rsplit("_", 1)[0] for t in targets}:
            counts[kind] += sum(t[1:].rsplit("_", 1)[0] == kind for t in targets) * REPEATS
        for label, prompt in (("with document", WITH_DOC.format(document=redacted, tokens=listing)),
                              ("no document (control)", NO_DOC.format(tokens=listing))):
            for _ in range(REPEATS):
                guesses = parse(ask(llm, prompt))
                for token, real in targets.items():
                    guess = guesses.get(token, "UNKNOWN")
                    kind, s = score(token, real, guess)
                    s["answered"] = int(str(guess).strip().upper() not in ("", "UNKNOWN"))
                    for metric, value in s.items():
                        tally[label][kind][metric] += value
                    if kind == "SALARY":
                        salary_rows.append(dict(condition=label, real=real, guess=str(guess)))
                time.sleep(pause)
        print(f"  finished {path} ({len(targets)} tokens)", flush=True)
    return ({label: {kind: dict(total=counts[kind], **dict(m)) for kind, m in kinds.items()} for label, kinds in tally.items()},
            salary_rows)


if __name__ == "__main__":
    from config import groq_model
    result, salary_rows = run()
    os.makedirs("eval/results", exist_ok=True)
    json.dump(dict(model=groq_model(), repeats=REPEATS, result=result, salary_guesses=salary_rows),
              open("eval/results/reidentification.json", "w"), indent=2)
    print(f"\nmodel {groq_model()}, each token guessed {REPEATS} times; share of guesses that are right")
    for kind in sorted(TYPES):
        for label in result:
            r = result[label].get(kind)
            if r and r["total"]:
                extra = f"   within 20%: {r['near'] / r['total']:.0%}" if kind == "SALARY" else (f"   name part right: {r['part'] / r['total']:.0%}" if kind == "PERSON" else "")
                print(f"  {kind:14s} {label:22s} exact {r['exact'] / r['total']:.0%}{extra}   made a guess: {r['answered'] / r['total']:.0%}   (n={r['total']})")
