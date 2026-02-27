from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from .models import NormalizedDocument, Paragraph, SourceAnchor

USER_AGENT = "NeuroCME-HighYieldCoach/0.1 (+educational-use)"


class UrlIngestError(RuntimeError):
    """Raised when URL ingestion fails cleanly."""


def fetch_html(url: str, timeout: int = 15) -> str:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - guarded by install docs
        raise UrlIngestError("requests is required for URL ingestion") from exc

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise UrlIngestError(f"Could not fetch URL: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and not response.text.lstrip().startswith("<"):
        raise UrlIngestError("URL did not return HTML content")
    return response.text


def ingest_url(url: str, timeout: int = 15) -> NormalizedDocument:
    html = fetch_html(url, timeout=timeout)
    return document_from_html(html=html, url=url)


def document_from_html(html: str, url: str, title: Optional[str] = None) -> NormalizedDocument:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - guarded by install docs
        raise UrlIngestError("beautifulsoup4 is required for URL ingestion") from exc

    readable_html = html
    readable_title = title
    try:
        from readability import Document as ReadabilityDocument

        readable = ReadabilityDocument(html)
        readable_html = readable.summary(html_partial=True)
        readable_title = readable_title or readable.short_title()
    except Exception:
        readable_html = html

    soup = BeautifulSoup(readable_html, "html.parser")
    if readable_title is None:
        title_tag = soup.find("title")
        readable_title = title_tag.get_text(" ", strip=True) if title_tag else "Untitled URL Document"

    paragraphs = _extract_paragraphs_from_soup(soup)
    if not paragraphs:
        full_soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs_from_soup(full_soup)
    if not paragraphs:
        text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
        raw_parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        paragraphs = [
            Paragraph(
                text=part,
                anchor=SourceAnchor(paragraph=index + 1, section="Body", snippet=_snippet(part)),
                section_heading="Body",
            )
            for index, part in enumerate(raw_parts)
        ]
    if not paragraphs:
        raise UrlIngestError("No readable article content found in URL")

    document_id = hashlib.sha1(f"url::{url}".encode("utf-8")).hexdigest()[:12]
    return NormalizedDocument(
        document_id=document_id,
        title=readable_title or "Untitled URL Document",
        source_type="url",
        source_ref=url,
        paragraphs=paragraphs,
        metadata={"paragraph_count": len(paragraphs)},
    )


def _extract_paragraphs_from_soup(soup) -> List[Paragraph]:
    current_heading = "Overview"
    paragraphs: List[Paragraph] = []
    paragraph_index = 0
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = node.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if node.name.startswith("h"):
            current_heading = text
            continue
        if len(text) < 35:
            continue
        paragraph_index += 1
        paragraphs.append(
            Paragraph(
                text=text,
                anchor=SourceAnchor(
                    paragraph=paragraph_index,
                    section=current_heading,
                    snippet=_snippet(text),
                ),
                section_heading=current_heading,
            )
        )
    return paragraphs


def _snippet(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
