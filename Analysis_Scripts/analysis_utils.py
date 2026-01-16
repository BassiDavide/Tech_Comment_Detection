# -*- coding: utf-8 -*-

TIME_GRAIN_DEFAULT = "trimester"

GRAIN_SPECS = {
    "semester": {"months": 6, "prefix": "S", "label": "Semester"},
    "trimester": {"months": 3, "prefix": "T", "label": "Trimester"},
    "quadrimester": {"months": 4, "prefix": "Qm", "label": "Quadrimester"},
    "quarter": {"months": 3, "prefix": "Q", "label": "Quarter"},
}

LABEL_LIST = [
    "appeal to authority",
    "appeal to fear, prejudice",
    "appeal to hypocrisy (to quoque)",
    "appeal to pity",
    "appeal to popularity (bandwagon)",
    "appeal to time",
    "appeal to values/flag waving",
    "black-and-white fallacy",
    "causal oversimplification",
    "distraction",
    "exaggeration or minimisation",
    "intentional vagueness",
    "loaded language",
    "name calling",
    "reductio ad hitlerum",
    "repetition",
    "slogans",
    "smears/doubt",
    "thought-terminating clich\u00e9",
]

PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#1100ffcd",
    "#e7298a",
    "#66a61e",
    "#fcdd11dd",
    "#a6761d",
    "#666666",
    "#1f78b4",
    "#33a02c",
    "#bb00ffdb",
    "#ff7f00",
    "#6a3d9a",
    "#b15928",
    "#a6cee3",
    "#b2df8a",
    "#e31a1c",
    "#12af00",
    "#cab2d6",
]

LABEL_ALIASES = {
    "appeal to hypocrisy": "appeal to hypocrisy (to quoque)",
}

DEFAULT_COLOR = "#000000"
LABEL_SET = set(LABEL_LIST)
TECHNIQUE_COLORS = dict(zip(LABEL_LIST, PALETTE))


def normalize_label(label):
    if not isinstance(label, str):
        return ""
    return " ".join(label.strip().lower().split())


def canonical_label(label):
    norm = normalize_label(label)
    if norm in LABEL_SET:
        return norm
    return LABEL_ALIASES.get(norm, norm)


def color_for_label(label):
    return TECHNIQUE_COLORS.get(canonical_label(label), DEFAULT_COLOR)


def resolve_grain(grain):
    if isinstance(grain, int):
        months = grain
        label = f"{months}-Month"
        prefix = f"M{months}P"
    elif isinstance(grain, str):
        norm = normalize_label(grain)
        if norm.isdigit():
            months = int(norm)
            label = f"{months}-Month"
            prefix = f"M{months}P"
        else:
            spec = GRAIN_SPECS.get(norm)
            if spec is None:
                raise ValueError(f"Unsupported TIME_GRAIN: {grain}")
            months = spec["months"]
            prefix = spec["prefix"]
            label = spec["label"]
    else:
        raise ValueError("TIME_GRAIN must be a string or integer months.")

    if months < 1 or months > 12:
        raise ValueError("TIME_GRAIN months must be between 1 and 12.")
    if 12 % months != 0:
        raise ValueError("TIME_GRAIN months must evenly divide 12.")
    return months, prefix, label


def get_period_label_from_month(year, month, months_per, prefix):
    period = (month - 1) // months_per + 1
    label = f"{year}-{prefix}{period}"
    return label, (year, period)


def period_label_from_timestamp(ts, months_per, prefix):
    if not isinstance(ts, str) or len(ts) < 7:
        return None
    try:
        year = int(ts[0:4])
        month = int(ts[5:7])
    except Exception:
        return None
    label, _ = get_period_label_from_month(year, month, months_per, prefix)
    return label


def shorten_period_label(label):
    if not isinstance(label, str):
        return label
    if len(label) >= 5 and label[4] == "-":
        return label[2:]
    return label
