from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .models import Chunk

GENERIC_HEADINGS = {
    "overview",
    "body",
    "introduction",
    "background",
    "page 1",
    "page 2",
    "page 3",
    "page 4",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

TOPIC_LEXICON = {
    "Status Epilepticus": ["status epilepticus", "refractory status epilepticus", "seizure"],
    "Intracranial Pressure": ["intracranial pressure", "cerebral perfusion pressure", "icp", "herniation"],
    "Neurocritical Airway": ["airway", "intubation", "ventilation", "oxygenation", "hyperventilation"],
    "Brain Perfusion Targets": ["map", "cpp", "perfusion", "blood pressure target"],
    "Sedation and Analgesia": ["sedation", "analgesia", "propofol", "midazolam", "ketamine"],
    "ECMO and Neuromonitoring": ["ecmo", "extracorporeal membrane oxygenation", "neuromonitoring"],
}


@dataclass(frozen=True)
class TopicSeed:
    label: str
    chunks: List[Chunk]


def propose_topic_seeds(chunks: List[Chunk]) -> List[TopicSeed]:
    grouped: Dict[str, List[Chunk]] = {}
    labels: Dict[str, str] = {}
    for chunk in chunks:
        label = derive_topic_label(chunk)
        normalized = _normalize_label(label)
        grouped.setdefault(normalized, []).append(chunk)
        labels[normalized] = label
    return [TopicSeed(label=labels[key], chunks=value) for key, value in grouped.items()]


def derive_topic_label(chunk: Chunk) -> str:
    heading = re.sub(r"\s+", " ", chunk.heading).strip()
    if heading and heading.lower() not in GENERIC_HEADINGS and len(heading.split()) <= 10:
        return heading

    text_lower = chunk.text.lower()
    for canonical, aliases in TOPIC_LEXICON.items():
        if any(alias in text_lower for alias in aliases):
            return canonical

    sentence = _first_sentence(chunk.text)
    nounish = re.match(r"([A-Z][A-Za-z0-9/\- ]{3,70}?)(?: is| are| remains| requires| should| can| may)", sentence)
    if nounish:
        return nounish.group(1).strip()

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text_lower)
    candidates = Counter()
    for size in (2, 3):
        for index in range(len(tokens) - size + 1):
            ngram = tokens[index : index + size]
            if ngram[0] in STOPWORDS or ngram[-1] in STOPWORDS:
                continue
            phrase = " ".join(ngram)
            candidates[phrase] += 1
    if candidates:
        best, _ = max(candidates.items(), key=lambda item: (item[1], len(item[0])))
        return best.title()
    return "Key Topic"


def _first_sentence(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence[:120]


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
