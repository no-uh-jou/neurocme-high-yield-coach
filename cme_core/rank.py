from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence

from .llm_provider import LLMProvider, NullLLMProvider
from .models import AnalysisOptions, Chunk, Flashcard, NormalizedDocument, Topic
from .outputs import (
    build_key_decision_points,
    build_pitfalls,
    build_summary_bullets,
    build_what_you_should_know,
)
from .scoring import classify_level, priority_from_score, score_explanation, score_text
from .topics import TopicSeed, propose_topic_seeds


def rank_document(
    document: NormalizedDocument,
    chunks: Sequence[Chunk],
    options: Optional[AnalysisOptions] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> List[Topic]:
    config = options or AnalysisOptions()
    provider = llm_provider or NullLLMProvider()
    topics = rank_chunks(chunks=chunks, options=config)
    if config.use_llm and provider.is_available():
        topics = list(provider.enrich_topics(document=document, chunks=chunks, topics=topics, options=config))
    return topics


def rank_chunks(chunks: Sequence[Chunk], options: Optional[AnalysisOptions] = None) -> List[Topic]:
    config = options or AnalysisOptions()
    topic_seeds = propose_topic_seeds(list(chunks))
    topics = [_topic_from_seed(seed, config) for seed in topic_seeds]
    topics.sort(key=lambda topic: (-topic.score, topic.label))
    return topics[: config.max_topics]


def _topic_from_seed(seed: TopicSeed, options: AnalysisOptions) -> Topic:
    merged_text = "\n\n".join(chunk.text for chunk in seed.chunks)
    anchors = _dedupe_anchors([anchor for chunk in seed.chunks for anchor in chunk.anchors])
    breakdown = score_text(merged_text, options)
    level = classify_level(merged_text)
    priority = priority_from_score(breakdown.total)
    rationale = f"{score_explanation(breakdown, level)} Anchors: {', '.join(anchor.label for anchor in anchors[:3])}."
    summary_bullets = build_summary_bullets(seed.label, merged_text, options.desired_depth)
    what_you_should_know = build_what_you_should_know(seed.label, merged_text, priority, level)
    pitfalls = build_pitfalls(merged_text, anchors)
    key_decision_points = build_key_decision_points(merged_text, anchors)
    flashcards = _build_flashcards(seed.label, what_you_should_know, anchors)
    topic_id = hashlib.sha1(f"{seed.label}:{merged_text[:120]}".encode("utf-8")).hexdigest()[:12]
    return Topic(
        topic_id=topic_id,
        label=seed.label,
        priority=priority,
        level=level,
        score=breakdown.total,
        rationale=rationale,
        anchors=anchors,
        citations=[anchor.label for anchor in anchors],
        breakdown=breakdown,
        summary_bullets=summary_bullets,
        what_you_should_know=what_you_should_know,
        pitfalls=pitfalls,
        key_decision_points=key_decision_points,
        flashcards=flashcards,
        supporting_chunk_ids=[chunk.chunk_id for chunk in seed.chunks],
    )


def _dedupe_anchors(anchors):
    seen = set()
    unique = []
    for anchor in anchors:
        key = (anchor.page, anchor.paragraph, anchor.section)
        if key in seen:
            continue
        seen.add(key)
        unique.append(anchor)
    return unique


def _build_flashcards(label: str, bullets: Sequence[str], anchors) -> List[Flashcard]:
    if not anchors:
        return []
    anchor_label = anchors[0].label
    cards: List[Flashcard] = []
    first_bullet = bullets[0] if bullets else f"{label} matters because it changes ICU decision making."
    answer = re.sub(r"^[A-Z][^:]+:\s*", "", first_bullet)
    cards.append(
        Flashcard(
            front=f"What should you know first about {label}?",
            back=answer,
            anchor_label=anchor_label,
            card_type="qa",
        )
    )
    cards.append(
        Flashcard(
            front=f"{label} requires {{{{c1::{answer[:80]}}}}}.",
            back=f"Source anchor: {anchor_label}",
            anchor_label=anchor_label,
            card_type="cloze",
        )
    )
    return cards
