"""
How much of samples/northwind_hr_records.txt does redaction hide?   python -m eval.sample_check
The value lists below are the ground truth for that fictional document.
"""
from analyzer import build_analyzer
from redaction import Redactor
text = open("samples/northwind_hr_records.txt", encoding="utf-8").read()
r = Redactor(build_analyzer(), {}, b"x"); out = r.blur_text(text)
truth = {
 "names": ["Priya Raghunathan","Marcus Thompson","Elena Vasquez","Tomasz Kowalski","Aisha Bello","Daniel O'Connor","Mei-Ling Chen","Maureen O'Connor","Bob Tanaka","Robert Tanaka"],
 "salary": ["$118,500","$126,000","$97,250","82,000 USD","$104k","$5,000","$152,000","$98,000"],
 "email": ["aisha.bello@northwind-logistics.com","daniel.oconnor@northwind-logistics.com","mei-ling.chen@northwind-logistics.com","tomasz.kowalski@northwind-logistics.com","bob.tanaka@northwind-logistics.com"],
 "phone": ["(415) 555-0142","212-555-0178","(212) 555-0199","+1 312 555 0116","617-555-0133"],
 "ssn": ["512-44-7391","498-27-1156"],
 "password": ["Wint3r!Harbor92","Lantern-Coast-58","blue.fox.77","tanaka_2019_summer"],
 "card": ["4111 1111 1111 1111"],
 "ip": ["10.20.4.37","192.168.14.9"],
 "address": ["2218 Alder Street"],
}
tot = leak = 0
for kind, vals in truth.items():
    missed = [v for v in vals if v.lower() in out.lower()]
    tot += len(vals); leak += len(missed)
    print(f"{kind:9s} {len(vals)-len(missed)}/{len(vals)} hidden" + (f"   MISSED: {missed}" if missed else ""))
print(f"\noverall hidden: {tot-leak}/{tot}")
# partial leaks: first names / last names left behind
import re
firsts = {"Priya","Marcus","Elena","Tomasz","Aisha","Daniel","Mei-Ling","Maureen","Bob","Robert"}
left = sorted(w for w in firsts if re.search(r"\b"+re.escape(w)+r"\b", out))
print("first names still visible somewhere:", left)
print("entity types produced:", sorted({t.split('_')[0].strip('[') for t in r.mapping}))
