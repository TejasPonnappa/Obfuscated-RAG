"""
How much of each sample document does redaction hide?   python -m eval.sample_check

The value lists are the ground truth for the fictional documents in samples/. A value counts as hidden if it no longer appears
verbatim in the redacted text. Keep one document unseen while changing the detectors, so the score on it is an honest test.
"""
import re

from analyzer import build_analyzer
from redaction import Redactor

DOMAIN = "northwind-logistics.com"
DOCS = {
    "samples/northwind_hr_records.txt": {
        "name": ["Priya Raghunathan", "Marcus Thompson", "Elena Vasquez", "Tomasz Kowalski", "Aisha Bello", "Daniel O'Connor",
                 "Mei-Ling Chen", "Maureen O'Connor", "Bob Tanaka", "Robert Tanaka"],
        "salary": ["$118,500", "$126,000", "$97,250", "82,000 USD", "$104k", "$5,000", "$152,000", "$98,000"],
        "email": [f"{u}@{DOMAIN}" for u in ("aisha.bello", "daniel.oconnor", "mei-ling.chen", "tomasz.kowalski", "bob.tanaka")],
        "phone": ["(415) 555-0142", "212-555-0178", "(212) 555-0199", "+1 312 555 0116", "617-555-0133"],
        "ssn": ["512-44-7391", "498-27-1156"],
        "password": ["Wint3r!Harbor92", "Lantern-Coast-58", "blue.fox.77", "tanaka_2019_summer"],
        "card": ["4111 1111 1111 1111"],
        "ip": ["10.20.4.37", "192.168.14.9"],
        "address": ["2218 Alder Street"],
    },
    "samples/harbor_clinic_emails.txt": {
        "name": ["Anika Rao", "Liam Fitzgerald", "Sofia Marchetti", "Omar Haddad", "Priya Nair", "Hannah Whitfield",
                 "Marco Delgado", "Eleanor Brooks"],
        "salary": ["USD 91,500", "$7,625", "78 thousand dollars", "$85,000"],
        "email": ["anika.rao@harborviewclinic.org", "liam.f@harborviewclinic.org", "sofia.marchetti@gmail.com",
                  "marco.delgado@outlook.com"],
        "phone": ["415.555.0182", "(617) 555 0147"],
        "address": ["14 Maple Avenue", "22 Pine Street"],
        "date_of_birth": ["3 July 1988", "11/02/1954"],
        "bank_account": ["GB29 NWBK 6016 1331 9268 19"],
        "password": ["Clinic#Spring24", "Sunrise-Dock-4412", "Blue$Heron2025"],
        "card": ["5500 0000 0000 0004"],
        "ssn": ["221-58-9034"],
        "id_number": ["HV-20418", "W482-993-117"],
    },
}


def score(path, truth, redactor):
    text = open(path, encoding="utf-8").read()
    out = redactor.blur_text(text)
    hidden = total = 0
    print(f"\n{path}")
    for kind, values in truth.items():
        missed = [v for v in values if v.lower() in out.lower()]
        if kind in ("password", "id_number", "bank_account", "card", "ssn", "phone"):      # a distinctive piece (has a digit) left behind
            partial = [v for v in values if v not in missed
                       and any(p in out for p in re.findall(r"[A-Za-z0-9]*\d[A-Za-z0-9]*", v) if len(p) >= 4)]
            if partial:
                print(f"  {kind:13s} partly visible: {partial}")
        hidden += len(values) - len(missed)
        total += len(values)
        print(f"  {kind:13s} {len(values) - len(missed)}/{len(values)} hidden" + (f"   MISSED: {missed}" if missed else ""))
    firsts = {n.split()[0] for n in truth.get("name", [])}
    left = sorted(w for w in firsts if re.search(r"\b" + re.escape(w) + r"\b", out))
    print(f"  overall hidden: {hidden}/{total} ({hidden / total:.0%});  first names still visible: {left}")
    return hidden, total


if __name__ == "__main__":
    analyzer = build_analyzer()
    for path, truth in DOCS.items():
        score(path, truth, Redactor(analyzer, {}, b"x"))
