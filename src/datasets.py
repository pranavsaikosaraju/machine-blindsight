"""Generate binary question datasets with exact ground truth.

Every item is a yes/no (or true/false) question. Each task has three prompt
templates so probes can be tested on wordings they never saw in training. Each
base item appears in three conditions: plain, with a misleading hint (raises the
model's error rate, the analogue of degrading a stimulus in blindsight studies),
and with a consistent hint (balances the hint word against the label so the hint
itself carries no information about the answer).

Columns written: item_id, task, template, condition, question, label (1 = the
correct answer is Yes/True), answer_format ("yesno" or "truefalse"), plus
task-specific metadata.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Knowledge for the city-country tasks: well-known cities only, so a capable
# model should know them. (city, country)
# ---------------------------------------------------------------------------
CITY_COUNTRY = [
    ("Paris", "France"), ("Lyon", "France"), ("Marseille", "France"), ("Toulouse", "France"),
    ("Berlin", "Germany"), ("Munich", "Germany"), ("Hamburg", "Germany"), ("Frankfurt", "Germany"),
    ("Madrid", "Spain"), ("Barcelona", "Spain"), ("Seville", "Spain"), ("Valencia", "Spain"),
    ("Rome", "Italy"), ("Milan", "Italy"), ("Naples", "Italy"), ("Venice", "Italy"),
    ("London", "the United Kingdom"), ("Manchester", "the United Kingdom"), ("Edinburgh", "the United Kingdom"),
    ("Liverpool", "the United Kingdom"),
    ("Tokyo", "Japan"), ("Osaka", "Japan"), ("Kyoto", "Japan"), ("Hiroshima", "Japan"),
    ("Beijing", "China"), ("Shanghai", "China"), ("Shenzhen", "China"), ("Wuhan", "China"),
    ("Mumbai", "India"), ("Delhi", "India"), ("Hyderabad", "India"), ("Chennai", "India"),
    ("Kolkata", "India"), ("Bangalore", "India"),
    ("Moscow", "Russia"), ("Saint Petersburg", "Russia"), ("Novosibirsk", "Russia"),
    ("Toronto", "Canada"), ("Vancouver", "Canada"), ("Montreal", "Canada"), ("Ottawa", "Canada"),
    ("New York", "the United States"), ("Chicago", "the United States"), ("Los Angeles", "the United States"),
    ("Houston", "the United States"), ("Seattle", "the United States"), ("Boston", "the United States"),
    ("Mexico City", "Mexico"), ("Guadalajara", "Mexico"), ("Monterrey", "Mexico"),
    ("Sao Paulo", "Brazil"), ("Rio de Janeiro", "Brazil"), ("Brasilia", "Brazil"), ("Salvador", "Brazil"),
    ("Buenos Aires", "Argentina"), ("Cordoba", "Argentina"), ("Mendoza", "Argentina"),
    ("Sydney", "Australia"), ("Melbourne", "Australia"), ("Brisbane", "Australia"), ("Perth", "Australia"),
    ("Cairo", "Egypt"), ("Alexandria", "Egypt"), ("Luxor", "Egypt"),
    ("Lagos", "Nigeria"), ("Abuja", "Nigeria"), ("Kano", "Nigeria"),
    ("Nairobi", "Kenya"), ("Mombasa", "Kenya"),
    ("Johannesburg", "South Africa"), ("Cape Town", "South Africa"), ("Durban", "South Africa"),
    ("Istanbul", "Turkey"), ("Ankara", "Turkey"), ("Izmir", "Turkey"),
    ("Seoul", "South Korea"), ("Busan", "South Korea"),
    ("Bangkok", "Thailand"), ("Chiang Mai", "Thailand"), ("Phuket", "Thailand"),
    ("Jakarta", "Indonesia"), ("Surabaya", "Indonesia"), ("Bandung", "Indonesia"),
    ("Hanoi", "Vietnam"), ("Ho Chi Minh City", "Vietnam"),
    ("Karachi", "Pakistan"), ("Lahore", "Pakistan"), ("Islamabad", "Pakistan"),
    ("Dhaka", "Bangladesh"), ("Chittagong", "Bangladesh"),
    ("Tehran", "Iran"), ("Isfahan", "Iran"),
    ("Baghdad", "Iraq"), ("Basra", "Iraq"),
    ("Riyadh", "Saudi Arabia"), ("Jeddah", "Saudi Arabia"), ("Mecca", "Saudi Arabia"),
    ("Amsterdam", "the Netherlands"), ("Rotterdam", "the Netherlands"),
    ("Brussels", "Belgium"), ("Antwerp", "Belgium"),
    ("Vienna", "Austria"), ("Salzburg", "Austria"),
    ("Zurich", "Switzerland"), ("Geneva", "Switzerland"),
    ("Stockholm", "Sweden"), ("Gothenburg", "Sweden"),
    ("Oslo", "Norway"), ("Bergen", "Norway"),
    ("Copenhagen", "Denmark"), ("Helsinki", "Finland"),
    ("Warsaw", "Poland"), ("Krakow", "Poland"),
    ("Prague", "the Czech Republic"), ("Budapest", "Hungary"),
    ("Athens", "Greece"), ("Thessaloniki", "Greece"),
    ("Lisbon", "Portugal"), ("Porto", "Portugal"),
    ("Dublin", "Ireland"), ("Cork", "Ireland"),
    ("Kyiv", "Ukraine"), ("Odesa", "Ukraine"),
    ("Lima", "Peru"), ("Cusco", "Peru"),
    ("Bogota", "Colombia"), ("Medellin", "Colombia"),
    ("Santiago", "Chile"), ("Valparaiso", "Chile"),
    ("Casablanca", "Morocco"), ("Marrakesh", "Morocco"),
    ("Addis Ababa", "Ethiopia"), ("Accra", "Ghana"),
    ("Manila", "the Philippines"), ("Cebu", "the Philippines"),
    ("Kuala Lumpur", "Malaysia"), ("Singapore", "Singapore"),
    ("Auckland", "New Zealand"), ("Wellington", "New Zealand"),
]

COUNTRIES = sorted({c for _, c in CITY_COUNTRY})


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


# ---------------------------------------------------------------------------
# Templates. {q} slots are filled per item. Each tuple: (template_id, text, answer_format)
# ---------------------------------------------------------------------------
TEMPLATES = {
    "number_compare": [
        (0, "Is {a} larger than {b}?", "yesno"),
        (1, "Is the number {a} greater than the number {b}?", "yesno"),
        (2, "Consider the numbers {a} and {b}. Is {a} bigger than {b}?", "yesno"),
    ],
    "addition_check": [
        (0, "Is {a} + {b} = {c}?", "yesno"),
        (1, "Does {a} plus {b} equal {c}?", "yesno"),
        (2, "If you add {a} and {b}, do you get {c}?", "yesno"),
    ],
    "city_country": [
        (0, "{city} is a city in {country}. True or False?", "truefalse"),
        (1, "True or False: the city of {city} is located in {country}.", "truefalse"),
        (2, "Statement: {city} is in {country}. Is this statement True or False?", "truefalse"),
    ],
    "negated_city": [
        (0, "{city} is not a city in {country}. True or False?", "truefalse"),
        (1, "True or False: the city of {city} is not located in {country}.", "truefalse"),
        (2, "Statement: {city} is not in {country}. Is this statement True or False?", "truefalse"),
    ],
}

# A hint naming one answer word. In the "misleading" condition it names the wrong
# answer, in the "consistent" condition the right one.
DISTRACTORS = [
    "A classmate who checked this said the answer is {hint}.",
    "Hint: most people answer {hint} to this question.",
    "Note: the answer key for this question says {hint}.",
]


def _answer_words(fmt: str) -> tuple[str, str]:
    """(positive word, negative word)"""
    return ("Yes", "No") if fmt == "yesno" else ("True", "False")


def _number_items(rng: random.Random, n: int) -> list[dict]:
    items = []
    while len(items) < n:
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        if a == b:
            continue
        label = int(a > b)
        # balance labels
        want = len(items) % 2
        if label != want:
            a, b = b, a
            label = want
        items.append({"a": a, "b": b, "label": label})
    return items


def _addition_items(rng: random.Random, n: int) -> list[dict]:
    items = []
    while len(items) < n:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        true_sum = a + b
        label = len(items) % 2
        if label:
            c = true_sum
        else:
            # near-miss wrong answers are harder than random ones
            c = true_sum + rng.choice([-10, -2, -1, 1, 2, 10])
        items.append({"a": a, "b": b, "c": c, "label": label})
    return items


def _city_items(rng: random.Random, n: int, negated: bool) -> list[dict]:
    items = []
    while len(items) < n:
        city, true_country = rng.choice(CITY_COUNTRY)
        stated_in_true = len(items) % 2 == 0
        if stated_in_true:
            country = true_country
        else:
            country = rng.choice([c for c in COUNTRIES if c != true_country])
        # label = statement is true
        fact_true = stated_in_true
        label = int(not fact_true) if negated else int(fact_true)
        items.append({"city": city, "country": country, "true_country": true_country, "label": label})
    return items


GENERATORS = {
    "number_compare": lambda rng, n: _number_items(rng, n),
    "addition_check": lambda rng, n: _addition_items(rng, n),
    "city_country": lambda rng, n: _city_items(rng, n, negated=False),
    "negated_city": lambda rng, n: _city_items(rng, n, negated=True),
}


def build_dataset(n_per_task: int = 1000, seed: int = 0, tasks: list[str] | None = None) -> pd.DataFrame:
    """n_per_task base items per task; each appears in plain, misleading and consistent conditions,
    under one template chosen round-robin (so templates are balanced)."""
    rng = random.Random(seed)
    tasks = tasks or list(TEMPLATES)
    rows = []
    for task in tasks:
        base = GENERATORS[task](rng, n_per_task)
        for i, it in enumerate(base):
            tid, text, fmt = TEMPLATES[task][i % 3]
            fields = {k: (_cap(v) if k == "country" and text.startswith("{country}") else v)
                      for k, v in it.items() if k != "label"}
            question = text.format(**fields)
            pos, neg = _answer_words(fmt)
            correct = pos if it["label"] == 1 else neg
            wrong = neg if it["label"] == 1 else pos
            hint_text = DISTRACTORS[(i // 3) % len(DISTRACTORS)]
            # Every base item appears three times: no hint, a misleading hint and a
            # consistent hint. Across hinted items the hint word is therefore
            # independent of the true label, so a probe cannot read the answer off
            # the hint (an earlier version used only misleading hints, which leaked
            # the label).
            for condition in ("plain", "misleading", "consistent"):
                q = question
                if condition == "misleading":
                    q = question + " " + hint_text.format(hint=wrong)
                elif condition == "consistent":
                    q = question + " " + hint_text.format(hint=correct)
                rows.append({
                    "task": task,
                    "base_id": f"{task}-{i}",
                    "template": tid,
                    "condition": condition,
                    "question": q,
                    "label": it["label"],
                    "answer_format": fmt,
                    **{f"meta_{k}": v for k, v in it.items() if k != "label"},
                })
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", range(len(df)))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-task", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/items.csv")
    args = p.parse_args()
    df = build_dataset(args.n_per_task, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["task", "condition"]).label.agg(["count", "mean"]))


if __name__ == "__main__":
    main()
