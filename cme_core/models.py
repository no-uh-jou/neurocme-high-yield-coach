from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


Priority = Literal["LOW", "MEDIUM", "HIGH"]
Level = Literal["BASIC", "INTERMEDIATE", "ADVANCED", "EXPERT"]
SourceType = Literal["pdf", "url", "text"]


@dataclass(frozen=True)
class SourceAnchor:
    page: Optional[int] = None
    paragraph: Optional[int] = None
    section: Optional[str] = None
    snippet: str = ""

    @property
    def label(self) -> str:
        parts: List[str] = []
        if self.page is not None:
            parts.append(f"Page {self.page}")
        if self.paragraph is not None:
            parts.append(f"Paragraph {self.paragraph}")
        if self.section:
            parts.append(self.section)
        return " | ".join(parts) if parts else "Source anchor"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["label"] = self.label
        return payload


@dataclass(frozen=True)
class Paragraph:
    text: str
    anchor: SourceAnchor
    section_heading: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["anchor"] = self.anchor.to_dict()
        return payload


@dataclass(frozen=True)
class NormalizedDocument:
    document_id: str
    title: str
    source_type: SourceType
    source_ref: str
    paragraphs: List[Paragraph]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    heading: str
    text: str
    anchors: List[SourceAnchor]
    paragraph_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "heading": self.heading,
            "text": self.text,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "paragraph_count": self.paragraph_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ScoreBreakdown:
    clinical_frequency: float
    high_stakes: float
    decision_density: float
    guideline_density: float
    pitfall_density: float
    rare_critical: float
    specialty_bonus: float
    total: float
    evidence_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Flashcard:
    front: str
    back: str
    anchor_label: str
    card_type: Literal["qa", "cloze"] = "qa"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Topic:
    topic_id: str
    label: str
    priority: Priority
    level: Level
    score: float
    rationale: str
    anchors: List[SourceAnchor]
    citations: List[str]
    breakdown: ScoreBreakdown
    summary_bullets: List[str]
    what_you_should_know: List[str]
    pitfalls: List[str]
    key_decision_points: List[str]
    flashcards: List[Flashcard]
    supporting_chunk_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "label": self.label,
            "priority": self.priority,
            "level": self.level,
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "citations": self.citations,
            "breakdown": self.breakdown.to_dict(),
            "summary_bullets": self.summary_bullets,
            "what_you_should_know": self.what_you_should_know,
            "pitfalls": self.pitfalls,
            "key_decision_points": self.key_decision_points,
            "flashcards": [flashcard.to_dict() for flashcard in self.flashcards],
            "supporting_chunk_ids": self.supporting_chunk_ids,
        }


@dataclass(frozen=True)
class AnalysisOptions:
    specialty_focus: Literal["Neuro ICU", "General ICU", "ECMO"] = "Neuro ICU"
    desired_depth: Literal["boards", "fellowship", "attending"] = "boards"
    output_type: Literal["outline", "pearls", "flashcards"] = "outline"
    use_llm: bool = False
    max_topics: int = 12

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
