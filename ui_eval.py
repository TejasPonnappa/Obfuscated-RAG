"""The Evaluation view: shows the measured results stored in docs/results/ (regenerate them with the scripts in eval/)."""
import json
from pathlib import Path

import streamlit as st

RESULTS = Path(__file__).parent / "docs" / "results"
VARIANTS = ["plain Presidio", "+ overlap by confidence", "+ custom rules", "+ name propagation"]


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def missing(script):
    st.info(f"No results stored yet. Run `python -m eval.{script}` and copy its JSON from eval/results/ to docs/results/.")


def bar(frame, x, y, order, color=None, height=280, fmt=".0%"):
    import altair as alt
    base = alt.Chart(frame).encode(x=alt.X(f"{x}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
                                   y=alt.Y(f"{y}:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%"), title=None))
    bars = base.mark_bar().encode(color=alt.Color(f"{color}:N") if color else alt.value("#4c78a8"), tooltip=[x, alt.Tooltip(y, format=fmt)])
    labels = base.mark_text(dy=-6).encode(text=alt.Text(f"{y}:Q", format=fmt))
    st.altair_chart((bars + labels).properties(height=height), use_container_width=True)


def render():
    import pandas as pd
    st.header("Evaluation")
    st.caption("Everything here was measured with the scripts in the `eval/` folder. Read the caveat under each result: most of it uses "
               "small fictional documents I wrote, so treat the numbers as indicative, not as proof.")

    # 1. how much is hidden
    st.subheader("1. How much sensitive text gets hidden")
    data = load("ablation.json")
    if not data:
        missing("ablation")
    else:
        s = data["samples"]
        st.markdown("**Three fictional documents (92 sensitive values).** Each step adds one improvement to the one before.")
        bar(pd.DataFrame({"step": VARIANTS, "hidden": [s[v]["pct"] for v in VARIANTS]}), "step", "hidden", VARIANTS)
        st.caption("Caveat: I wrote these documents and the extra rules at the same time, so the jump from the second to the third step "
                   "is the most flattering number in this app.")
        tab = data.get("tab")
        if tab:
            st.markdown(f"**TAB, a public benchmark of real court cases** ({data['tab_docs']} documents, human-annotated identifiers). "
                        "Share of identifiers hidden, by category:")
            keys = [("type PERSON", "Names"), ("type DATETIME", "Dates"), ("type LOC", "Places"), ("type ORG", "Organisations"),
                    ("type CODE", "Case and ID numbers"), ("DIRECT (all types)", "Direct identifiers (all)"), ("all masked", "All masked identifiers")]
            rows = [{"Category": label, "Count": tab[VARIANTS[0]][k]["total"], **{v: tab[v][k]["pct"] for v in VARIANTS}}
                    for k, label in keys if k in tab[VARIANTS[0]]]
            st.dataframe(pd.DataFrame(rows).style.format({v: "{:.0%}" for v in VARIANTS}), hide_index=True, width="stretch")
            over = tab[VARIANTS[-1]]["NO_MASK values hidden (over-redaction)"]
            st.caption(f"The extra rules help little here (they were written for HR-style text). The redactor is good at names and dates, weak at "
                       f"case numbers and organisations, which it was not built for. It also hides {over['pct']:.0%} of the values the annotators "
                       "said did not need hiding (over-redaction). A value counts as hidden only if it appears nowhere in the output.")

    # 2. utility
    st.subheader("2. What privacy costs in answer quality")
    qa = load("qa_utility.json")
    if not qa:
        missing("qa_utility")
    else:
        import altair as alt
        summ = qa["summary"]
        total = summ["raw document"]["total"]
        rows = [{"setting": "No privacy (raw document)", "prompt": "Raw document", "correct": summ["raw document"]["correct"] / total}]
        for label, name in (("No noise", "redacted, no noise"), ("Noise 100 (default)", "redacted, eps 100"), ("Noise 20", "redacted, eps 20")):
            for prompt, suffix in (("Original prompt", ""), ("Improved prompt", " (new prompt)")):
                if name + suffix in summ:
                    rows.append({"setting": label, "prompt": prompt, "correct": summ[name + suffix]["correct"] / total})
        frame = pd.DataFrame(rows)
        order = list(dict.fromkeys(frame["setting"]))
        base = alt.Chart(frame).encode(
            x=alt.X("setting:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=170)),
            xOffset=alt.XOffset("prompt:N"), y=alt.Y("correct:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%"), title=None))
        bars = base.mark_bar().encode(color=alt.Color("prompt:N", title=None, sort=["Raw document", "Original prompt", "Improved prompt"]),
                                      tooltip=["setting", "prompt", alt.Tooltip("correct", format=".0%")])
        labels = base.mark_text(dy=-6).encode(text=alt.Text("correct:Q", format=".0%"))
        st.altair_chart((bars + labels).properties(height=300), use_container_width=True)
        st.caption(f"{total} questions about the three fictional documents, model {qa['model']}. The first bar is the LLM seeing the whole "
                   f"unredacted document (no privacy). The others use redaction, restore the real values locally, and retrieve {qa['k']} chunks "
                   "with different amounts of embedding noise. The improved prompt tells the model that tags are harmless placeholders to "
                   "repeat; the original made it refuse some password questions. An answer is correct if it contains the right value.")

    # 3. retrieval vs noise
    st.subheader("3. Embedding noise and finding the right chunk")
    pu = load("privacy_utility.json")
    if not pu:
        missing("privacy_utility")
    else:
        rows = pu["deterministic_tokens"]
        labels = [("no_noise", "none"), ("eps_500", "500"), ("eps_200", "200"), ("eps_100", "100"), ("eps_50", "50"), ("eps_20", "20"), ("eps_10", "10"), ("eps_2", "2")]
        values = [rows["no_noise"]["5"]["mean"]] + [rows[k]["queries_noised"]["5"] for k, _ in labels[1:]]
        order = [name for _, name in labels]
        bar(pd.DataFrame({"epsilon": order, "found": values}), "epsilon", "found", order)
        st.caption(f"Share of questions where the right chunk is among the top 5 of {pu['chunks']} (random guessing would be "
                   f"{pu['random_baseline']['5']:.0%}). Smaller epsilon means more noise. The app's default, 100, keeps about 95% of it; "
                   "at 50 it keeps about 80% and at 20 about 30%. This is why the noise is fixed at 100 and not chosen automatically.")

    # 4. guessing attack
    st.subheader("4. Can the LLM guess the hidden values?")
    atk = load("reidentification.json")
    if not atk:
        missing("reidentification")
    else:
        names = {"PERSON": "Names", "SALARY": "Salaries", "EMAIL_ADDRESS": "Emails", "PHONE_NUMBER": "Phone numbers", "PASSWORD": "Passwords", "DATE_TIME": "Dates and times"}
        rows = []
        for kind, label in names.items():
            doc, ctl = atk["result"]["with document"].get(kind), atk["result"]["no document (control)"].get(kind)
            if not doc:
                continue
            metric = "near" if kind == "SALARY" else "exact"
            rows.append({"Hidden value": label + (" (within 20%)" if kind == "SALARY" else " (exact)"),
                         "Guesses": doc["total"], "LLM sees the redacted document": doc[metric] / doc["total"],
                         "LLM sees no document (control)": ctl[metric] / ctl["total"] if ctl else None})
        st.dataframe(pd.DataFrame(rows).style.format({"LLM sees the redacted document": "{:.0%}", "LLM sees no document (control)": "{:.0%}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Model {atk['model']}. The control shows what it would guess anyway; only the gap between the two columns is information "
                   "the document gave away. Tokens hide values but not everything around them, so a number the model can reason about "
                   "(like a salary for a stated job) can still be estimated.")

    # 5. leakage
    st.subheader("5. Does the LLM ever receive a real value?")
    leak = load("llm_leakage.json")
    if not leak:
        missing("llm_leakage")
    else:
        n = leak["ordinary"]["trials"] + leak["extraction"]["trials"] + leak["honeytoken"]["trials"]
        bad = leak["ordinary"]["with_a_real_value"] + leak["extraction"]["with_a_real_value"] + leak["honeytoken"]["leaked"]
        st.markdown(f"**{bad} of {n} trials** returned a real value (ordinary questions, attempts to make the LLM print the real values, "
                    "and the self-audit's planted secret).")
        st.caption("Measured with the original prompt, before the prompt in section 2 was improved; a re-run is pending. "
                   "This mostly confirms that detection covered every value in these fictional documents: the LLM can only repeat what it was "
                   "given. It says nothing about values the detector misses in your own document, which is what section 1 measures.")
