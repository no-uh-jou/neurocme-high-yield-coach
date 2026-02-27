from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import List, Optional

from .models import NormalizedDocument, Paragraph, SourceAnchor


class PdfIngestError(RuntimeError):
    """Raised when PDF ingestion fails cleanly."""


def ingest_pdf_path(path: str | Path) -> NormalizedDocument:
    pdf_path = Path(path)
    return ingest_pdf_bytes(pdf_path.read_bytes(), source_name=pdf_path.name)


def ingest_pdf_bytes(pdf_bytes: bytes, source_name: str = "uploaded.pdf") -> NormalizedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - guarded by install docs
        raise PdfIngestError("pypdf is required for PDF ingestion") from exc

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise PdfIngestError(f"Could not read PDF: {exc}") from exc

    paragraphs: List[Paragraph] = []
    global_paragraph_index = 0
    extracted_title: Optional[str] = None

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.replace("\x00", " ").strip()
        if not page_text:
            continue
        if extracted_title is None:
            first_line = next((line.strip() for line in page_text.splitlines() if line.strip()), None)
            extracted_title = first_line or source_name
        page_paragraphs = _extract_page_paragraphs(page_text, page_number)
        for paragraph in page_paragraphs:
            global_paragraph_index += 1
            anchor = SourceAnchor(
                page=page_number,
                paragraph=global_paragraph_index,
                section=paragraph.section_heading,
                snippet=paragraph.anchor.snippet,
            )
            paragraphs.append(
                Paragraph(
                    text=paragraph.text,
                    anchor=anchor,
                    section_heading=paragraph.section_heading,
                )
            )

    if not paragraphs:
        raise PdfIngestError("No readable text extracted from PDF")

    document_id = hashlib.sha1(f"pdf::{source_name}::{len(pdf_bytes)}".encode("utf-8")).hexdigest()[:12]
    return NormalizedDocument(
        document_id=document_id,
        title=extracted_title or source_name,
        source_type="pdf",
        source_ref=source_name,
        paragraphs=paragraphs,
        metadata={"page_count": len(reader.pages), "paragraph_count": len(paragraphs)},
    )


def _extract_page_paragraphs(page_text: str, page_number: int) -> List[Paragraph]:
    current_heading = f"Page {page_number}"
    lines = [line.strip() for line in page_text.splitlines()]
    paragraphs: List[Paragraph] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        buffer.clear()
        if len(text) < 25:
            return
        paragraphs.append(
            Paragraph(
                text=text,
                anchor=SourceAnchor(page=page_number, section=current_heading, snippet=_snippet(text)),
                section_heading=current_heading,
            )
        )

    for line in lines:
        if not line:
            flush_buffer()
            continue
        if _is_heading(line):
            flush_buffer()
            current_heading = line
            continue
        buffer.append(line)
        joined = " ".join(buffer)
        if line.endswith((".", "?", "!")) and len(joined) >= 60:
            flush_buffer()
    flush_buffer()
    return paragraphs


def _is_heading(line: str) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    if len(clean) < 4 or len(clean) > 90:
        return False
    if clean.endswith("."):
        return False
    if clean.isupper():
        return True
    words = clean.split()
    if len(words) <= 8 and sum(word[:1].isupper() for word in words) >= max(1, len(words) - 1):
        return True
    return False


def _snippet(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
