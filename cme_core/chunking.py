from __future__ import annotations

import hashlib
from typing import List

from .models import Chunk, NormalizedDocument, Paragraph


def extract_chunks(
    document: NormalizedDocument,
    max_chars: int = 1100,
    min_chars: int = 280,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    buffer: List[Paragraph] = []
    current_heading = document.paragraphs[0].section_heading or "Overview"

    def flush_buffer() -> None:
        if not buffer:
            return
        text = "\n\n".join(paragraph.text for paragraph in buffer).strip()
        chunk_index = len(chunks) + 1
        chunk_id = hashlib.sha1(
            f"{document.document_id}:{chunk_index}:{current_heading}:{text[:120]}".encode("utf-8")
        ).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                heading=current_heading or "Overview",
                text=text,
                anchors=[paragraph.anchor for paragraph in buffer],
                paragraph_count=len(buffer),
                metadata={"source_type": document.source_type},
            )
        )
        buffer.clear()

    for paragraph in document.paragraphs:
        heading = paragraph.section_heading or current_heading or "Overview"
        proposed_size = len("\n\n".join(item.text for item in buffer + [paragraph]))
        heading_changed = buffer and heading != current_heading
        too_large = proposed_size > max_chars
        if buffer and (heading_changed or too_large) and len("\n\n".join(item.text for item in buffer)) >= min_chars:
            flush_buffer()
        if not buffer:
            current_heading = heading
        buffer.append(paragraph)
        if len("\n\n".join(item.text for item in buffer)) >= max_chars:
            flush_buffer()
    flush_buffer()
    return chunks
