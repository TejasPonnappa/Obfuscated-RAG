# Evaluation

What was measured, how, and what it does and does not show. Every number comes from a script in `eval/`; the raw results are in `docs/results/` and are also shown in the app (Evaluation view).

The short version: the redaction is good at names and dates, weak at identifiers it has no pattern for, and the LLM could not recover any hidden value by guessing. All the LLM tests use small fictional documents that I wrote, so they are indicative, not proof.

## 1. How much sensitive text gets hidden (`python -m eval.ablation`)

Four configurations, each adding one thing to the one before. A value counts as hidden only if its text appears **nowhere** in the redacted output (strict).

**Three fictional documents, 92 sensitive values**

| Configuration | Hidden |
|---|---|
| Plain Presidio | 63% |
| + overlap by confidence | 65% |
| + custom rules (money, street addresses, passwords, names after a label) | 90% |
| + name propagation | 91% |

The big jump comes from the custom rules, which I wrote while looking at these same documents. That makes it the most flattering number here.

**TAB: 127 real court cases (European Court of Human Rights), 4,383 human-annotated identifiers**

| Category | Hidden |
|---|---|
| Names | 95% |
| Dates | 99% |
| Places | 71% |
| Organisations | 29% |
| Case and application numbers (e.g. `30326/03`) | 6% |
| All masked identifiers | 78% |

The custom rules change almost nothing here (77% to 78%), because they were written for HR-style text, not court cases. Of the **direct** identifiers, names are 99% hidden and the case numbers are 99% visible. That one format is the whole gap: Presidio has no pattern for it, and a generic `digits/digits` rule would also catch dates and fractions. It is a clear case where a domain needs its own rule.

The redactor also hides about 34% of the values the annotators said did **not** need hiding (over-redaction). Most of that is dates and places.

TAB is public (MIT licence): github.com/NorskRegnesentral/text-anonymisation-benchmark. The data is downloaded into `eval/data/` and not committed.

## 2. What privacy costs in answer quality (`python -m eval.qa_utility`)

27 questions about the three fictional documents, each with a known answer; model `openai/gpt-oss-20b`. An answer is correct if it contains the right value after the real values are restored locally. In the redacted conditions 4 chunks are retrieved (the app would pass these small documents whole, which would hide the effect of noise).

| Condition | Correct |
|---|---|
| Raw document (no privacy, upper bound) | 27 / 27 |
| Redacted, no embedding noise | 22 / 27 |
| Redacted, noise epsilon 100 (the default) | 23 / 27 |
| Redacted, noise epsilon 20 (heavy) | 20 / 27 |

Redaction costs about 5 of 27 answers here. Epsilon 100 looks as good as no noise; the one-answer difference is within the run-to-run variation. The failures were mostly the model refusing to repeat a password tag ("I can't help with that") or saying the value was not available, not wrong detection.

I then changed the prompt (`ANSWER_PROMPT` in `config.py`) to say plainly that tags are harmless placeholders to repeat. **That change is in the app but not yet measured**: the run that compares the two prompts was cut off at 125 of 189 answers by Groq's free daily token limit (200,000 for this model). Re-run `python -m eval.qa_utility` after the limit resets; it resumes from its cache.

## 3. Embedding noise and finding the right chunk (`python -m eval.privacy_utility`)

120 chunks, 80 questions, 4 corpora. Share of questions where the right chunk is in the top 5 (random guessing: 4%), noise added to both the stored chunks and the question:

| Epsilon | none | 500 | 200 | 100 | 50 | 20 | 10 | 2 |
|---|---|---|---|---|---|---|---|---|
| Found | 79% | 79% | 79% | 76% | 63% | 24% | 9% | 6% |

At epsilon 100 the noise is about 0.3 times the size of the embedding and about 95% of the no-noise retrieval survives; at 50 about 80%, at 20 about 30%. This is light obfuscation, not a differential-privacy guarantee. The original design picked epsilon with a "GAN" loop, which drove it below 1 (about 0.5 in the last run), where retrieval is close to random, so it was replaced with the fixed value.

## 4. Can the LLM guess the hidden values? (`python -m eval.reidentification`)

The LLM gets the redacted document and must give its best guess for **every** token (it may not answer UNKNOWN). As a control, it gets only the token list with no document, which shows what it would guess anyway. 2 repeats, 3 documents.

| Hidden value | Guesses | Right, with the document | Right, no document (control) |
|---|---|---|---|
| Names (exact) | 92 | 0% | 0% |
| Emails, phones, passwords, dates (exact) | 102 | 0% | 0% |
| Salaries (within 20%) | 30 | 47% | 40% |

Nothing exact was recovered with the document. (With no document, 2 of 30 salary guesses happened to be exactly right, round numbers like 5,000.) A salary guess lands within 20% about 40% of the time with no document at all, so the redacted text adds only a few points on top of what the model assumes anyway (n = 30, so that gap is not significant). The names are fictional, so this cannot test whether a famous person could be recognised from context.

My first version of this test said "0% guessed" for the document case. That was wrong: the model's hidden reasoning had used up the output budget and every answer came back empty. The numbers above are from the fixed run (larger budget, forced guesses), and the script now saves the salary guesses so they can be inspected.

## 5. Does the LLM ever receive a real value? (`python -m eval.llm_leakage`)

45 trials with the original prompt (15 ordinary questions, 15 attempts to make the LLM print the real values, 15 runs of the app's self-audit): 0 returned a real value. This mostly confirms that detection covered every value in these fictional documents, since the LLM can only repeat what it was given. It says nothing about values the detector misses in your own document, which is what section 1 measures. A re-run with the new prompt is pending for the same reason as in section 2.

## What this does not show

- No test on a real, private document of the kind people would actually upload; the fictional documents and the HR-style rules share an author.
- One LLM (`gpt-oss-20b`), small samples, no repeated runs for the QA test.
- Re-identification of real people from public knowledge, linkage across documents, and attackers with side information.
- Scanned PDFs (no text layer) are not supported, and detection is best-effort: check the Pipeline view for what actually went out.
