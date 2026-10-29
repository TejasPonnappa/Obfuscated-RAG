"""
What does privacy cost in answer quality?   python -m eval.qa_utility

27 questions over the three fictional sample documents, each with a known answer. The same questions are asked under several
conditions and an answer counts as correct if it contains the known value (after the app's own local restoration of tokens;
spaces and punctuation are ignored when comparing):
  raw document            the LLM sees the whole unredacted document (no privacy; the upper bound)
  redacted, ...           redacted text, the 4 closest chunks, restored locally; with no noise, the app's default epsilon 100,
                          and heavy noise (epsilon 20)
  ... (new prompt)        the same, with the improved prompt in config.py instead of the original one
Retrieval is identical across the two prompts, so the difference between them is the prompt alone. Four chunks are retrieved
because the app would pass all of these small documents, which would hide the effect of noise.
Needs GROQ_API_KEY. Results are cached, so an interrupted run resumes.
"""
import json
import os
import re
import time
import zlib

import numpy as np

K = 4
CACHE = "eval/results/qa_cache.json"
RAW_PROMPT = "Answer the question using only the context. Be brief.\n\nContext:\n{context}\n\nQuestion: {question}"

QA = {   # question, known answer, kind
    "samples/northwind_hr_records.txt": [
        ("What is Aisha Bello's starting salary?", "$97,250", "sensitive value"),
        ("What is the password for the backup server login?", "Wint3r!Harbor92", "sensitive value"),
        ("What is Tomasz Kowalski's phone number?", "617-555-0133", "sensitive value"),
        ("What is Aisha Bello's email address?", "aisha.bello@northwind-logistics.com", "sensitive value"),
        ("At what time did the monitoring system flag the failed logins?", "03:12", "sensitive value"),
        ("Whose laptop were the failed logins traced to?", "Bob Tanaka", "sensitive value"),
        ("What is Elena Vasquez's salary?", "$152,000", "sensitive value"),
        ("What is the IP address of the payroll database?", "10.20.4.37", "sensitive value"),
        ("What did the incident report conclude about payroll data?", "no payroll data was accessed", "other"),
    ],
    "samples/harbor_clinic_emails.txt": [
        ("What is Sofia Marchetti's annual base salary?", "91,500", "sensitive value"),
        ("What is Sofia Marchetti's phone number?", "415.555.0182", "sensitive value"),
        ("What is Sofia Marchetti's date of birth?", "3 July 1988", "sensitive value"),
        ("What is the front desk laptop password?", "Clinic#Spring24", "sensitive value"),
        ("What is Omar Haddad's new cell number?", "(617) 555 0147", "sensitive value"),
        ("What is Marco Delgado's email address?", "marco.delgado@outlook.com", "sensitive value"),
        ("What is Sofia Marchetti's employee ID?", "HV-20418", "sensitive value"),
        ("What is Eleanor Brooks's insurance member number?", "W482-993-117", "sensitive value"),
        ("What job is Sofia Marchetti joining as?", "nurse practitioner", "other"),
    ],
    "samples/riverbend_customer_letters.txt": [
        ("What is Piotr Jankowski's account number?", "40817263", "sensitive value"),
        ("What is Piotr Jankowski's sort code?", "20-45-67", "sensitive value"),
        ("What is Gareth Pugh's salary?", "31,500", "sensitive value"),
        ("What is the staff rota password?", "Rivers!Bend2024", "sensitive value"),
        ("What is Adaeze Okonkwo's national insurance number?", "QQ 12 34 56 C", "sensitive value"),
        ("What is Emeka Okonkwo's email address?", "emeka.okonkwo@gmail.com", "sensitive value"),
        ("What is Santiago Herrera-Lopez's temporary passphrase?", "Orchid-Lamp-3391", "sensitive value"),
        ("Which engineer is mentioned as fixing the boiler fault?", "Gareth Pugh", "sensitive value"),
        ("What was the cause of the boiler fault?", "pressure valve", "other"),
    ],
}
CONDITIONS = [   # name, epsilon (None = no noise), prompt ("old", "new" or None for the raw document)
    ("raw document", None, None),
    ("redacted, no noise", None, "old"), ("redacted, eps 100", 100.0, "old"), ("redacted, eps 20", 20.0, "old"),
    ("redacted, no noise (new prompt)", None, "new"), ("redacted, eps 100 (new prompt)", 100.0, "new"),
    ("redacted, eps 20 (new prompt)", 20.0, "new"),
]


def norm(s):
    """Lower-case letters and digits only, so '(617) 555 0147' matches '617-555-0147' and bold markers do not matter."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def run(pause=1.5):
    from dotenv import load_dotenv                 # heavy imports live here so norm() can be tested without them
    from langchain_groq import ChatGroq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    from analyzer import build_analyzer
    from config import ANSWER_PROMPT, groq_model
    from eval.llm_leakage import SYSTEM, ask
    from noise import DifferentialPrivacyAgent
    from redaction import Redactor

    load_dotenv()
    prompts = {"old": SYSTEM, "new": ANSWER_PROMPT}
    llm = ChatGroq(model=groq_model(), temperature=0.1)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    analyzer = build_analyzer()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    results = []
    for path, questions in QA.items():
        text = open(path, encoding="utf-8").read()
        redactor = Redactor(analyzer, {}, b"qa")
        redacted = redactor.blur_text(text)
        chunks = [d.page_content for d in RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([redacted])]
        vectors = embedder.encode(chunks)
        for q, gold, kind in questions:
            for cond, eps, prompt in CONDITIONS:
                key = f"{path}|{q}|{cond}|{groq_model()}"
                if key not in cache:
                    if prompt is None:
                        answer = ask(llm, RAW_PROMPT.format(context=text, question=q))
                    else:
                        np.random.seed(zlib.crc32(f"{path}|{q}".encode()) % 2**31)       # same noise for both prompts
                        blurred_q = redactor.blur_prompt(q)
                        cv, qv = vectors, embedder.encode([blurred_q])[0]
                        if eps:
                            agent = DifferentialPrivacyAgent(epsilon=eps)
                            cv, qv = agent.inject_laplace_noise(vectors), agent.inject_laplace_noise(qv)
                        top = np.argsort(((cv - qv) ** 2).sum(1))[:K]
                        context = "\n\n".join(chunks[i] for i in top)
                        answer = redactor.reassemble_text(ask(llm, prompts[prompt].format(context=context, question=blurred_q)))
                    cache[key] = answer
                    json.dump(cache, open(CACHE, "w"))
                    time.sleep(pause)
                results.append(dict(doc=path, q=q, kind=kind, cond=cond, correct=norm(gold) in norm(cache[key])))
        print(f"  finished {path}", flush=True)
    return results


def summarize(results):
    out = {}
    for cond, _, _ in CONDITIONS:
        rows = [r for r in results if r["cond"] == cond]
        out[cond] = dict(correct=sum(r["correct"] for r in rows), total=len(rows),
                         sensitive=dict(correct=sum(r["correct"] for r in rows if r["kind"] == "sensitive value"),
                                        total=sum(r["kind"] == "sensitive value" for r in rows)),
                         other=dict(correct=sum(r["correct"] for r in rows if r["kind"] == "other"),
                                    total=sum(r["kind"] == "other" for r in rows)))
    return out


if __name__ == "__main__":
    from config import groq_model
    results = run()
    summary = summarize(results)
    os.makedirs("eval/results", exist_ok=True)
    json.dump(dict(model=groq_model(), k=K, summary=summary, detail=results), open("eval/results/qa_utility.json", "w"), indent=2)
    print(f"\nmodel {groq_model()}, {sum(len(v) for v in QA.values())} questions per condition (4 chunks retrieved in the redacted conditions)")
    for cond, s in summary.items():
        print(f"  {cond:34s} {s['correct']:2d}/{s['total']}   sensitive-value questions {s['sensitive']['correct']}/{s['sensitive']['total']}"
              f"   other {s['other']['correct']}/{s['other']['total']}")
