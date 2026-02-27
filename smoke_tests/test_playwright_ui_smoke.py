from __future__ import annotations

import functools
import http.server
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:  # pragma: no cover - explicit failure path for smoke environments
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app" / "app.py"
SAMPLE_DIR = ROOT / "sample_data"
SAMPLE_PDF = SAMPLE_DIR / "sample_page.pdf"
SCREENSHOT_PATH = ROOT / "agent_artifacts" / "last_run" / "playwright_ui_smoke.png"


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if int(response.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for HTTP readiness: {url}")


def test_playwright_smoke_pdf_and_url_flows() -> None:
    if sync_playwright is None:
        pytest.fail(
            "Missing dependency: playwright. Install with "
            "`python3 -m pip install -e '.[smoke]' && python3 -m playwright install chromium`."
        )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SAMPLE_DIR))
    with ReusableTCPServer(("127.0.0.1", 0), handler) as sample_server:
        sample_thread = threading.Thread(target=sample_server.serve_forever, daemon=True)
        sample_thread.start()
        sample_url = f"http://127.0.0.1:{sample_server.server_address[1]}/sample_article.html"

        streamlit_port = _free_port()
        streamlit_url = f"http://127.0.0.1:{streamlit_port}"
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.address=127.0.0.1",
            f"--server.port={streamlit_port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            _wait_for_http(streamlit_url)
            SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1400})
                page.goto(streamlit_url, wait_until="domcontentloaded", timeout=60000)

                page.get_by_text("NeuroCME High-Yield Coach").first.wait_for(timeout=60000)
                page.get_by_role("tab", name="Paste URL").click()
                page.get_by_label("Open-access article URL").fill(sample_url)
                page.get_by_role("button", name="Fetch preview").click()
                page.get_by_text("Fetched URL preview.").wait_for(timeout=60000)
                page.get_by_text("Sample Neurocritical Care Article").first.wait_for(timeout=60000)

                page.get_by_role("button", name="Analyze URL").click()
                page.get_by_text("URL analyzed.").wait_for(timeout=60000)
                page.get_by_text("Results").wait_for(timeout=60000)
                page.get_by_text("Status Epilepticus").first.wait_for(timeout=60000)
                page.get_by_role("button", name="Download JSON").wait_for(timeout=60000)
                page.get_by_role("button", name="Download CSV").wait_for(timeout=60000)
                page.get_by_role("button", name="Download Markdown").wait_for(timeout=60000)
                page.get_by_role("button", name="Download Anki TSV").wait_for(timeout=60000)

                page.get_by_role("tab", name="Upload PDF").click()
                page.locator("input[type='file']").set_input_files(str(SAMPLE_PDF))
                page.get_by_text("Selected file:").wait_for(timeout=60000)
                page.get_by_role("button", name="Analyze PDF").click()
                page.get_by_text("PDF analyzed.").wait_for(timeout=60000)
                page.get_by_text("Neuro ICU High-Yield PDF").first.wait_for(timeout=60000)
                page.wait_for_function(
                    "text => document.body.innerHTML.includes(text)",
                    arg="Page 1 | Paragraph 1 | Status Epilepticus",
                    timeout=60000,
                )

                page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
                browser.close()
        finally:
            sample_server.shutdown()
            sample_thread.join(timeout=2)

            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
