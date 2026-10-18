"""Web search / fetch tools.

These are simple implementations using `urllib` for `web_fetch` and a
Google web search URL pattern for `web_search`. For a production-grade
deployment, plug in a real search backend (Google Custom Search, Brave, etc.).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List


class _LinkExtractor(HTMLParser):
    """Extract result links and titles from a Google results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[dict] = []
        self._in_link = False
        self._current_href: str = ""
        self._current_text: list = []
        self._seen: set = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value.startswith("/url?q="):
                    # Google wraps result links as /url?q=<actual-url>&sa=...
                    parsed = urllib.parse.urlparse(value)
                    qs = urllib.parse.parse_qs(parsed.query)
                    actual = (qs.get("q") or [""])[0]
                    if actual and actual not in self._seen:
                        self._in_link = True
                        self._current_href = actual
                        self._current_text = []
                        self._seen.add(actual)

    def handle_data(self, data):
        if self._in_link:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            title = "".join(self._current_text).strip()
            if title:
                self.results.append({"title": title, "url": self._current_href})


def web_search(query: str, max_results: int = 8) -> str:
    """Search the web for `query`.

    Uses a basic Google fetch (no API key). For a richer integration, replace
    this with a call to your favorite search API.
    """
    url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_bytes = resp.read()
    except Exception as e:
        return f"Error: web_search failed: {e}"

    try:
        html_text = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        return "Error: could not decode search results"

    parser = _LinkExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        pass

    results = parser.results[:max_results]
    if not results:
        return "(no results)"
    return "\n".join(f"{i+1}. {r['title']}\n   {r['url']}" for i, r in enumerate(results))


def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch the contents of a URL and return plain text."""
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; gemini-py/0.1; +https://gemini.google/)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            text = data.decode(charset, errors="replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # Strip HTML tags if it looks like HTML.
    if "<html" in text.lower():
        text = _strip_html(text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _strip_html(html: str) -> str:
    """Convert HTML to plain text by removing tags and collapsing whitespace."""
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Insert newline after block-level tags
    html = re.sub(r"</(p|div|li|h[1-6]|tr|br)>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", html)
    # Unescape entities
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Collapse whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
