from __future__ import annotations

import json
import re
from collections import Counter
from importlib import resources
from typing import Dict, Iterable, List, Tuple

from .models import AnalysisOptions, Level, Priority, ScoreBreakdown

SIGNAL_TERMS: Dict[str, Tuple[str, ...]] = {
    "clinical_frequency": (
        "common",
        "frequent",
        "routine",
        "typical",
        "first-line",
        "initial",
        "status epilepticus",
        "stroke",
        "sedation",
        "airway",
        "ventilation",
    ),
    "high_stakes": (
        "death",
        "mortality",
        "herniation",
        "irreversible",
        "emergency",
        "urgent",
        "injury",
        "cannot miss",
        "time-sensitive",
        "brain injury",
    ),
    "decision_density": (
        "if",
        "when",
        "versus",
        "escalate",
        "algorithm",
        "next",
        "refractory",
        "titrate",
        "consider",
        "should",
    ),
    "guideline_density": (
        "guideline",
        "recommend",
        "recommended",
        "should",
        "target",
        "dose",
        "classification",
        "contraindication",
        "board",
    ),
    "pitfall_density": (
        "pitfall",
        "pearl",
        "avoid",
        "contraindication",
        "warning",
        "mistake",
        "do not",
        "watch for",
    ),
    "rare_critical": (
        "rare",
        "salvage",
        "can't miss",
        "can’t miss",
        "super refractory",
        "ecmo",
        "malignant",
        "decompressive",
    ),
}

LEVEL_TERMS: Dict[Level, Tuple[str, ...]] = {
    "BASIC": ("basic", "fundamental", "definition", "recognition", "initial", "first-line"),
    "INTERMEDIATE": ("second-line", "nuance", "consult", "adjust", "titrate", "adjunct"),
    "ADVANCED": ("advanced", "refractory", "algorithm", "invasive", "multimodal", "ivig", "plex"),
    "EXPERT": ("expert", "ecmo", "impella", "salvage", "tertiary", "neuromonitoring"),
}

SPECIALTY_TERMS = {
    "Neuro ICU": ("seizure", "status epilepticus", "intracranial", "brain", "herniation", "cpp", "icp"),
    "General ICU": ("shock", "pressor", "ventilation", "sepsis", "antibiotic", "sedation"),
    "ECMO": ("ecmo", "extracorporeal", "cannula", "anticoagulation", "oxygenator"),
}


def load_scoring_weights() -> Dict[str, float]:
    with resources.files("cme_core").joinpath("scoring_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_text(text: str, options: AnalysisOptions) -> ScoreBreakdown:
    lower = text.lower()
    evidence_terms: List[str] = []
    weights = load_scoring_weights()
    signal_scores: Dict[str, float] = {}

    for signal_name, terms in SIGNAL_TERMS.items():
        hits = _count_term_hits(lower, terms)
        if hits:
            evidence_terms.extend(hit for hit in terms if hit in lower)
        signal_scores[signal_name] = min(1.0, hits / weights["normalizers"].get(signal_name, 3.0))

    specialty_hits = _count_term_hits(lower, SPECIALTY_TERMS.get(options.specialty_focus, ()))
    specialty_bonus = min(0.12, specialty_hits * 0.03)
    total = (
        signal_scores["clinical_frequency"] * weights["weights"]["clinical_frequency"]
        + signal_scores["high_stakes"] * weights["weights"]["high_stakes"]
        + signal_scores["decision_density"] * weights["weights"]["decision_density"]
        + signal_scores["guideline_density"] * weights["weights"]["guideline_density"]
        + signal_scores["pitfall_density"] * weights["weights"]["pitfall_density"]
        + signal_scores["rare_critical"] * weights["weights"]["rare_critical"]
        + specialty_bonus
    )
    total = min(1.0, total)
    unique_evidence = list(dict.fromkeys(evidence_terms))
    return ScoreBreakdown(
        clinical_frequency=signal_scores["clinical_frequency"],
        high_stakes=signal_scores["high_stakes"],
        decision_density=signal_scores["decision_density"],
        guideline_density=signal_scores["guideline_density"],
        pitfall_density=signal_scores["pitfall_density"],
        rare_critical=signal_scores["rare_critical"],
        specialty_bonus=specialty_bonus,
        total=total,
        evidence_terms=unique_evidence[:12],
    )


def priority_from_score(score: float) -> Priority:
    if score >= 0.63:
        return "HIGH"
    if score >= 0.34:
        return "MEDIUM"
    return "LOW"


def classify_level(text: str) -> Level:
    lower = text.lower()
    counts = Counter()
    for level, terms in LEVEL_TERMS.items():
        counts[level] = _count_term_hits(lower, terms)
    if counts["EXPERT"] >= 1 and counts["ADVANCED"] + counts["EXPERT"] >= 2:
        return "EXPERT"
    if counts["ADVANCED"] >= 1:
        return "ADVANCED"
    if counts["INTERMEDIATE"] >= 1:
        return "INTERMEDIATE"
    return "BASIC"


def score_explanation(breakdown: ScoreBreakdown, level: Level) -> str:
    drivers = []
    if breakdown.high_stakes >= 0.3:
        drivers.append("high-stakes neurologic or ICU consequences")
    if breakdown.decision_density >= 0.3:
        drivers.append("dense management branching")
    if breakdown.guideline_density >= 0.3:
        drivers.append("board-style recommendations or targets")
    if breakdown.pitfall_density >= 0.3:
        drivers.append("pearls, pitfalls, or contraindications")
    if breakdown.rare_critical >= 0.3:
        drivers.append("rare-but-critical rescue content")
    if not drivers:
        drivers.append("foundational clinical teaching points")
    evidence = ", ".join(breakdown.evidence_terms[:5]) if breakdown.evidence_terms else "text structure and keyword density"
    return f"Level {level}. Priority driven by {', '.join(drivers)}; evidence terms: {evidence}."


def _count_term_hits(text: str, terms: Iterable[str]) -> int:
    total = 0
    for term in terms:
        pattern = re.escape(term)
        total += len(re.findall(pattern, text))
    return total
