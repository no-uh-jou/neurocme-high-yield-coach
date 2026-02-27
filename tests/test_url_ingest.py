from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path

from cme_core import ingest


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"


def test_ingest_url_extracts_paragraph_anchors() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SAMPLE_DIR))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/sample_article.html"
        document = ingest.ingest_url(url)
        server.shutdown()
        thread.join(timeout=2)

    assert document.source_type == "url"
    assert document.title == "Sample Neurocritical Care Article"
    assert len(document.paragraphs) >= 5
    assert document.paragraphs[0].anchor.paragraph == 1
    assert document.paragraphs[1].anchor.section == "Status Epilepticus"
