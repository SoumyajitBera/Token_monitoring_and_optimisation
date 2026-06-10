import re
from typing import Dict, List

from .math_utils import CONSTRAINT_WORDS, NEGATIONS, STOPWORDS, words

NUMBER_RE = re.compile(r"(?:[$₹€£])?\b\d+(?:[,.]\d+)*(?:%|\b)")
CODE_RE = re.compile(r"\b[A-Z]{1,6}[-_]?[0-9]{1,8}[A-Z0-9-]*\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+){0,4})\b")


def extract_numbers(text: str) -> List[str]:
    return sorted(set(m.group(0) for m in NUMBER_RE.finditer(text or "")))


def extract_codes(text: str) -> List[str]:
    return sorted(set(m.group(0) for m in CODE_RE.finditer(text or "")))


def extract_entities(text: str) -> List[str]:
    raw = [m.group(0).strip() for m in ENTITY_RE.finditer(text or "")]
    out = []
    for item in raw:
        low = item.lower()
        if low in STOPWORDS or len(item) < 2:
            continue
        out.append(item)
    return sorted(set(out))


def extract_constraints(text: str) -> List[str]:
    toks = words(text or "", keep_stopwords=True)
    constraints = []
    for i, tok in enumerate(toks):
        if tok in CONSTRAINT_WORDS or tok in NEGATIONS:
            window = " ".join(toks[max(0, i - 3): i + 4])
            constraints.append(window)
    return sorted(set(constraints))


def _constraint_group(window: str) -> str:
    w = f" {window.lower()} "
    if any(x in w for x in [" not ", " no ", " never ", " without ", " cannot ", " can't ", " must not ", " should not ", " do not "]):
        return "negation"
    if any(x in w for x in [" only if ", " unless ", " provided ", " if ", " when ", " subject to "]):
        return "conditional"
    if any(x in w for x in [" at least ", " minimum ", " min ", ">=", " greater than ", " above ", " more than "]):
        return "threshold_min"
    if any(x in w for x in [" at most ", " maximum ", " max ", "<=", " less than ", " below ", " no more than "]):
        return "threshold_max"
    if any(x in w for x in [" json ", " schema ", " structured ", " output ", " return ", " format "]):
        return "format"
    if any(x in w for x in [" priority ", " prioritize ", " critical ", " important ", " focus "]):
        return "priority"
    if any(x in w for x in [" must ", " required ", " mandatory ", " need ", " needs ", " shall ", " have to ", " has to ", " essential "]):
        return "mandatory"
    return "generic"


def extract_constraint_atoms(text: str) -> List[Dict[str, object]]:
    """Extract logical constraint atoms from text.

    Each atom has a group, nearby target terms, numbers, and safety weight.
    This supports scoring semantic preservation without requiring exact wording.
    """
    toks = words(text or "", keep_stopwords=True)
    content_toks = words(text or "", keep_stopwords=False)
    atoms: List[Dict[str, object]] = []
    seen = set()

    for i, tok in enumerate(toks):
        trigger = tok in CONSTRAINT_WORDS or tok in NEGATIONS
        # phrase triggers using adjacent tokens
        prev_tok = toks[i - 1] if i > 0 else ""
        phrase = f"{prev_tok} {tok}".strip()
        if phrase in {"only if", "at least", "at most", "must not", "should not", "do not", "more than", "less than"}:
            trigger = True
        if not trigger:
            continue

        start, end = max(0, i - 6), min(len(toks), i + 9)
        window_toks = toks[start:end]
        window = " ".join(window_toks)
        group = _constraint_group(window)
        terms = []
        for t in window_toks:
            if len(t) >= 3 and t not in STOPWORDS and t not in CONSTRAINT_WORDS and t not in NEGATIONS:
                terms.append(t)
        nums = extract_numbers(window)
        # Extra target context: add strong terms from nearby sentence-ish window.
        if len(terms) < 3:
            for t in content_toks:
                if t not in terms and len(t) >= 5:
                    terms.append(t)
                if len(terms) >= 5:
                    break
        weight = 1.0
        if group in {"negation", "threshold_min", "threshold_max"}:
            weight = 1.35
        elif group in {"mandatory", "conditional"}:
            weight = 1.2
        key = (group, tuple(terms[:6]), tuple(nums))
        if key in seen:
            continue
        seen.add(key)
        atoms.append({"group": group, "terms": terms[:8], "numbers": nums, "window": window, "weight": weight})
    return atoms


def extract_keywords(text: str, max_keywords: int = 30) -> List[str]:
    toks = words(text or "")
    scored = {}
    for t in toks:
        if len(t) < 3:
            continue
        score = len(t)
        if any(ch.isdigit() for ch in t):
            score += 4
        if t.upper() == t and len(t) > 1:
            score += 3
        scored[t] = max(scored.get(t, 0), score)
    return [k for k, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:max_keywords]]
