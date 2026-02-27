from __future__ import annotations

import csv
import io
import json
import re
from typing import List, Sequence

from .models import NormalizedDocument, Topic

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_summary_bullets(label: str, text: str, desired_depth: str = "boards") -> List[str]:
    limit = 3 if desired_depth == "boards" else 4
    sentences = _clean_sentences(text)
    ranked = _rank_sentences(sentences)
    bullets = [sentence for sentence in ranked[:limit]]
    if not bullets:
        bullets = [f"{label} is a practical ICU topic with source-supported decision points."]
    return bullets


def build_what_you_should_know(label: str, text: str, priority: str, level: str) -> List[str]:
    sentences = _rank_sentences(_clean_sentences(text))
    bullets = []
    for sentence in sentences[:3]:
        bullets.append(f"{label}: {sentence}")
    if not bullets:
        bullets.append(f"{label}: {priority} priority, {level.lower()} level topic.")
    return bullets


def build_pitfalls(text: str, anchors) -> List[str]:
    candidates = [
        sentence
        for sentence in _clean_sentences(text)
        if any(term in sentence.lower() for term in ("avoid", "pitfall", "warning", "contraindication", "delay"))
    ]
    if candidates:
        return candidates[:3]
    anchor = anchors[0].label if anchors else "source text"
    return [f"Watch for delayed escalation or missed contraindications flagged near {anchor}."]


def build_key_decision_points(text: str, anchors) -> List[str]:
    candidates = [
        sentence
        for sentence in _clean_sentences(text)
        if any(term in sentence.lower() for term in ("should", "if", "when", "escalate", "target", "consider"))
    ]
    if candidates:
        return candidates[:3]
    anchor = anchors[0].label if anchors else "source text"
    return [f"Use the decision thresholds summarized around {anchor} to guide escalation."]


def export_topics_json(document: NormalizedDocument, topics: Sequence[Topic]) -> str:
    payload = {
        "document": document.to_dict(),
        "topics": [topic.to_dict() for topic in topics],
    }
    return json.dumps(payload, indent=2)


def export_topics_csv(topics: Sequence[Topic]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["topic", "priority", "level", "score", "anchors", "rationale"],
    )
    writer.writeheader()
    for topic in topics:
        writer.writerow(
            {
                "topic": topic.label,
                "priority": topic.priority,
                "level": topic.level,
                "score": f"{topic.score:.3f}",
                "anchors": "; ".join(topic.citations),
                "rationale": topic.rationale,
            }
        )
    return buffer.getvalue()


def export_topics_markdown(document: NormalizedDocument, topics: Sequence[Topic]) -> str:
    lines = [
        f"# {document.title}",
        "",
        "> Educational use only. Not medical advice.",
        "",
    ]
    for topic in topics:
        lines.extend(
            [
                f"## {topic.label}",
                f"- Priority: {topic.priority}",
                f"- Level: {topic.level}",
                f"- Anchors: {', '.join(topic.citations)}",
                f"- Rationale: {topic.rationale}",
                "",
                "### Summary",
                *[f"- {item}" for item in topic.summary_bullets],
                "",
                "### What You Should Know",
                *[f"- {item}" for item in topic.what_you_should_know],
                "",
                "### Pitfalls",
                *[f"- {item}" for item in topic.pitfalls],
                "",
                "### Key Decision Points",
                *[f"- {item}" for item in topic.key_decision_points],
                "",
            ]
        )
    return "\n".join(lines)


def export_anki_tsv(topics: Sequence[Topic]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(["front", "back", "anchor", "card_type"])
    for topic in topics:
        for flashcard in topic.flashcards:
            writer.writerow([flashcard.front, flashcard.back, flashcard.anchor_label, flashcard.card_type])
    return buffer.getvalue()


def _clean_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(normalized) if len(sentence.strip()) > 35]
    return sentences


def _rank_sentences(sentences: Sequence[str]) -> List[str]:
    def score(sentence: str) -> tuple[int, int]:
        lower = sentence.lower()
        signal_score = sum(
            term in lower
            for term in ("should", "target", "avoid", "urgent", "refractory", "contraindication", "escalate")
        )
        return (signal_score, len(sentence))

    return [sentence for sentence, _ in sorted(((sentence, score(sentence)) for sentence in sentences), key=lambda item: item[1], reverse=True)]
