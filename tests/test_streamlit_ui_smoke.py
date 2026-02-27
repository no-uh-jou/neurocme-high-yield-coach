from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
APP_PATH = ROOT / "streamlit_app" / "app.py"


def test_streamlit_ui_smoke_url_flow_and_filters() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SAMPLE_DIR))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            sample_url = f"http://127.0.0.1:{server.server_address[1]}/sample_article.html"
            at = AppTest.from_file(str(APP_PATH), default_timeout=15)
            at.run()

            assert [tab.label for tab in at.tabs] == ["Upload PDF", "Paste URL"]
            assert len(at.tabs[0].get("file_uploader")) == 1

            at.text_input[0].set_value(sample_url)
            at.button[0].click().run()
            assert [message.value for message in at.success] == ["Fetched URL preview."]
            assert any("Sample Neurocritical Care Article" in markdown.value for markdown in at.markdown)

            at.button[1].click().run()
            assert [message.value for message in at.success] == ["URL analyzed."]
            assert len(at.dataframe) == 1

            topic_table = at.dataframe[0].value
            assert not topic_table.empty
            assert "Status Epilepticus" in set(topic_table["Topic"])
            assert len(at.expander) >= 2
            assert any("Status Epilepticus" in expander.label for expander in at.expander)

            download_labels = [element.proto.label for element in at.get("download_button")]
            assert download_labels == [
                "Download JSON",
                "Download CSV",
                "Download Markdown",
                "Download Anki TSV",
            ]

            initial_row_count = len(topic_table)
            at.multiselect[0].set_value(["HIGH"]).run()
            filtered_table = at.dataframe[0].value
            assert len(filtered_table) <= initial_row_count
            assert set(filtered_table["Priority"]) <= {"HIGH"}
        finally:
            server.shutdown()
            thread.join(timeout=2)
