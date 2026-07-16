"""Tests for livesearch.py — offline (network calls are monkeypatched out)."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

ls = importlib.import_module("livesearch")


def test_google_news_rss_url_encodes_query_and_recency():
    url = ls.google_news_rss("iran oil sanctions", when="7d")
    assert url.startswith("https://news.google.com/rss/search?")
    assert "when%3A7d" in url  # "when:7d" url-encoded
    assert "iran" in url


def test_google_news_rss_no_when():
    url = ls.google_news_rss("drones", when="")
    assert "when%3A" not in url


@pytest.mark.parametrize(
    "value",
    ["Tue, 15 Jul 2025 12:00:00 GMT", "2025-07-15T12:00:00Z", "2025-07-15T12:00:00+00:00"],
)
def test_parse_dt_accepts_rfc822_and_iso(value):
    dt = ls._parse_dt(value)
    assert dt is not None
    assert dt.year == 2025


@pytest.mark.parametrize("value", [None, "", "not a date"])
def test_parse_dt_rejects_garbage(value):
    assert ls._parse_dt(value) is None


def test_tag_strips_namespace():
    class E:
        tag = "{http://www.w3.org/2005/Atom}entry"

    assert ls._tag(E()) == "entry"


RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <item><title>First &amp; foremost</title><link>https://x.io/1</link>
    <pubDate>{d}</pubDate></item>
  <item><title>Second</title><link>https://x.io/2</link>
    <pubDate>{d}</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry><title>Atom item</title>
    <link href="https://a.io/1"/><updated>{d}</updated></entry>
</feed>"""


def _now_rfc822() -> str:
    return format_datetime(datetime.now(timezone.utc))


def test_fetch_feed_parses_rss(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: RSS.format(d=_now_rfc822()).encode())
    items = ls.fetch_feed("https://feed")
    assert len(items) == 2
    assert items[0]["title"] == "First & foremost"  # HTML entity unescaped
    assert items[0]["link"] == "https://x.io/1"
    assert items[0]["published"].endswith("Z")
    assert items[0]["source"] == "Test Feed"


def test_fetch_feed_parses_atom_link_href(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: ATOM.format(d=_now_rfc822()).encode())
    items = ls.fetch_feed("https://feed")
    assert items[0]["link"] == "https://a.io/1"


def test_fetch_feed_limit(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: RSS.format(d=_now_rfc822()).encode())
    assert len(ls.fetch_feed("https://feed", limit=1)) == 1


def test_fetch_feed_bad_xml_returns_empty(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: b"<not xml")
    assert ls.fetch_feed("https://feed") == []


def test_fetch_feed_network_error_returns_empty(monkeypatch):
    def boom(url):
        raise OSError("no network")

    monkeypatch.setattr(ls, "_get", boom)
    assert ls.fetch_feed("https://feed") == []


def test_web_search_stamps_query(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: RSS.format(d=_now_rfc822()).encode())
    items = ls.web_search("kinetic strike report", when="1d")
    assert items and all(it["query"] == "kinetic strike report" for it in items)
    assert all(it["source"] == "google-news" for it in items)


DDG_HTML = (
    '<a rel="nofollow" class="result__a" '
    'href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpost">Great <b>post</b></a>'
)


def test_ddg_search_extracts_real_url_and_strips_tags(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: DDG_HTML.encode())
    items = ls.ddg_search("query")
    assert items[0]["link"] == "https://example.com/post"
    assert items[0]["title"] == "Great post"
    assert items[0]["source"] == "duckduckgo"


def test_ddg_search_network_error(monkeypatch):
    def boom(url):
        raise OSError

    monkeypatch.setattr(ls, "_get", boom)
    assert ls.ddg_search("q") == []


def test_harvest_dedupes_and_sorts_newest_first(monkeypatch):
    older = format_datetime(datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = format_datetime(datetime(2026, 6, 1, tzinfo=timezone.utc))
    feed = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>F</title>
      <item><title>old</title><link>https://x.io/old</link><pubDate>{older}</pubDate></item>
      <item><title>new</title><link>https://x.io/new</link><pubDate>{newer}</pubDate></item>
      <item><title>dupe</title><link>https://x.io/new</link><pubDate>{newer}</pubDate></item>
    </channel></rss>"""
    monkeypatch.setattr(ls, "_get", lambda url: feed.encode())
    out = ls.harvest(["https://feed"], since_days=0, min_year=2026)
    links = [it["link"] for it in out]
    assert links == ["https://x.io/new", "https://x.io/old"]  # newest first, deduped


def test_harvest_filters_by_min_year(monkeypatch):
    old = format_datetime(datetime(2020, 1, 1, tzinfo=timezone.utc))
    feed = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>F</title>
      <item><title>ancient</title><link>https://x.io/a</link><pubDate>{old}</pubDate></item>
    </channel></rss>"""
    monkeypatch.setattr(ls, "_get", lambda url: feed.encode())
    assert ls.harvest(["https://feed"], since_days=0, min_year=2026) == []


def test_harvest_accepts_query_dicts(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: RSS.format(d=_now_rfc822()).encode())
    out = ls.harvest([{"query": "test topic"}], since_days=30)
    assert out and all(it["query"] == "test topic" for it in out)


def test_harvest_skips_unrecognized_source(monkeypatch):
    monkeypatch.setattr(ls, "_get", lambda url: RSS.format(d=_now_rfc822()).encode())
    out = ls.harvest([{"neither": "x"}], since_days=30)
    assert out == []
