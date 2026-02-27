from __future__ import annotations

from pathlib import Path

from cme_core import extract, ingest, rank
from cme_core.models import AnalysisOptions


ROOT = Path(__file__).resolve().parents[1]


def test_high_yield_topics_are_ranked_and_classified() -> None:
    html = (ROOT / "sample_data" / "sample_article.html").read_text(encoding="utf-8")
    document = ingest.document_from_html(html=html, url="https://example.test/sample")
    chunks = extract.extract_chunks(document)
    topics = rank.rank_document(document=document, chunks=chunks, options=AnalysisOptions())

    labels = {topic.label: topic for topic in topics}
    assert "Status Epilepticus" in labels
    assert labels["Status Epilepticus"].priority == "HIGH"
    assert labels["Status Epilepticus"].level in {"BASIC", "INTERMEDIATE", "ADVANCED"}
    assert any(topic.level == "EXPERT" for topic in topics)
    assert any(topic.priority == "HIGH" for topic in topics)
