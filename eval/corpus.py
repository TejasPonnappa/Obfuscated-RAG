"""A synthetic HR-style corpus with known sensitive values, so redaction and retrieval can be measured."""
import random

from faker import Faker

AREA_CODES = [212, 312, 415, 617, 202, 305, 206, 512, 303, 404]     # real US area codes (Faker invents unassigned ones)
DEPARTMENTS = ["Finance", "Engineering", "Legal", "Marketing", "Logistics", "Research", "Security", "Support"]


def _phone(rng):
    return f"({rng.choice(AREA_CODES)}) {rng.randint(201, 989)}-{rng.randint(1000, 9999)}"


def make_people(n, seed=0):
    fake = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)
    people = []
    used = set()
    while len(people) < n:
        name = fake.name()
        if name in used:
            continue
        used.add(name)
        people.append(dict(
            name=name, email=fake.email(), phone=_phone(rng), dept=rng.choice(DEPARTMENTS),
            salary=f"${rng.randint(45, 210) * 1000:,}", password=f"{fake.word()}{rng.randint(100, 999)}!"))
    return people


def make_records(people):
    """Three lines per person (profile, payroll, review) so the same person is mentioned in several chunks.
    Returns [{"text", "person", "kind", "sensitive": [(value, label)]}]; labels name the kind of value, and passwords are
    split by sentence form: 'password_plain' ("password is X", the form the app's regex expects) and 'password_natural'
    ("password for NAME is X")."""
    records = []
    for i, p in enumerate(people):
        records.append(dict(person=i, kind="profile",
                            sensitive=[(p["name"], "name"), (p["email"], "email"), (p["phone"], "phone")],
                            text=f"Employee {p['name']} works in the {p['dept']} department. "
                                 f"Contact them at {p['email']} or {p['phone']}."))
        records.append(dict(person=i, kind="payroll", sensitive=[(p["name"], "name"), (p["salary"], "salary")],
                            text=f"Payroll note: {p['name']} is paid {p['salary']} per year."))
        plain = i % 2 == 0
        phrase = f"The backup admin password is {p['password']}." if plain else             f"The backup admin password for {p['name']} is {p['password']}."
        records.append(dict(person=i, kind="review",
                            sensitive=[(p["name"], "name"), (p["password"], "password_plain" if plain else "password_natural")],
                            text=f"Review of {p['name']}: reliable and punctual. {phrase}"))
    return records


def make_queries(people):
    """(question, person index, kind of the record that answers it)."""
    out = []
    for i, p in enumerate(people):
        out.append((f"How much is {p['name']} paid per year?", i, "payroll"))
        out.append((f"Which department does {p['name']} work in?", i, "profile"))
    return out
