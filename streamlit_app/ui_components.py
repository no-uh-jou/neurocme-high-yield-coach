from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import streamlit as st

from cme_core.models import NormalizedDocument, Topic
from cme_core.outputs import export_anki_tsv, export_topics_csv, export_topics_json, export_topics_markdown


def topic_rows(topics: Sequence[Topic]) -> List[Dict[str, str]]:
    return [
        {
            "Topic": topic.label,
            "Priority": topic.priority,
            "Level": topic.level,
            "Score": f"{topic.score:.2f}",
            "Anchors": ", ".join(topic.citations[:3]),
        }
        for topic in topics
    ]


def filter_topics(topics: Sequence[Topic], priorities: Iterable[str], levels: Iterable[str]) -> List[Topic]:
    allowed_priorities = set(priorities)
    allowed_levels = set(levels)
    return [
        topic
        for topic in topics
        if topic.priority in allowed_priorities and topic.level in allowed_levels
    ]


def render_topic_details(topics: Sequence[Topic], output_type: str) -> None:
    for topic in topics:
        with st.expander(f"{topic.label} · {topic.priority} · {topic.level}", expanded=False):
            st.markdown(f"**Rationale**: {topic.rationale}")
            st.markdown(f"**Source anchors**: {', '.join(topic.citations)}")
            if output_type == "pearls":
                _render_list("Pitfalls", topic.pitfalls)
                _render_list("Key Decision Points", topic.key_decision_points)
                _render_list("What You Should Know", topic.what_you_should_know)
            elif output_type == "flashcards":
                _render_list("Summary", topic.summary_bullets)
                _render_flashcards(topic)
            else:
                _render_list("Summary", topic.summary_bullets)
                _render_list("What You Should Know", topic.what_you_should_know)
                _render_list("Pitfalls", topic.pitfalls)
                _render_list("Key Decision Points", topic.key_decision_points)


def render_export_buttons(document: NormalizedDocument, topics: Sequence[Topic]) -> None:
    json_payload = export_topics_json(document, topics)
    csv_payload = export_topics_csv(topics)
    markdown_payload = export_topics_markdown(document, topics)
    anki_payload = export_anki_tsv(topics)

    col1, col2, col3, col4 = st.columns(4)
    col1.download_button(
        "Download JSON",
        data=json_payload,
        file_name="neurocme_topics.json",
        mime="application/json",
        width="stretch",
    )
    col2.download_button(
        "Download CSV",
        data=csv_payload,
        file_name="neurocme_topics.csv",
        mime="text/csv",
        width="stretch",
    )
    col3.download_button(
        "Download Markdown",
        data=markdown_payload,
        file_name="neurocme_topics.md",
        mime="text/markdown",
        width="stretch",
    )
    col4.download_button(
        "Download Anki TSV",
        data=anki_payload,
        file_name="neurocme_flashcards.tsv",
        mime="text/tab-separated-values",
        width="stretch",
    )


def _render_list(title: str, items: Sequence[str]) -> None:
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


def _render_flashcards(topic: Topic) -> None:
    st.markdown("**Flashcards**")
    for flashcard in topic.flashcards:
        st.markdown(f"- `{flashcard.card_type}` | **Front**: {flashcard.front}")
        st.markdown(f"  **Back**: {flashcard.back}")
