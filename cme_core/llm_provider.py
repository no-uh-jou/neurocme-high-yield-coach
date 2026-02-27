from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .models import AnalysisOptions, Chunk, NormalizedDocument, Topic


class LLMProvider(ABC):
    """Provider boundary for optional topic enrichment."""

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def enrich_topics(
        self,
        document: NormalizedDocument,
        chunks: Sequence[Chunk],
        topics: Sequence[Topic],
        options: AnalysisOptions,
    ) -> Sequence[Topic]:
        raise NotImplementedError


class NullLLMProvider(LLMProvider):
    """Default provider that preserves the heuristic pipeline."""

    def is_available(self) -> bool:
        return False

    def enrich_topics(
        self,
        document: NormalizedDocument,
        chunks: Sequence[Chunk],
        topics: Sequence[Topic],
        options: AnalysisOptions,
    ) -> Sequence[Topic]:
        return list(topics)
