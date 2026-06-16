#!/usr/bin/env python3
"""Scrape the cms-talk Statistics category into knowledge_base/forum/.

Writes both .txt (human-readable) and .json (structured per-post) for each
topic, plus a .manifest.json that records last_posted_at so re-runs only
fetch topics whose latest reply changed.

Auth (one of):
    DISCOURSE_COOKIE     Browser cookies from a logged-in cms-talk session.
                         How to grab them:
                           1. Log into https://cms-talk.web.cern.ch in your
                              browser (CERN SSO).
                           2. Open devtools -> Application -> Storage ->
                              Cookies -> https://cms-talk.web.cern.ch.
                           3. Copy the values of BOTH `_forum_session` (the
                              Rails session — this is what actually
                              authenticates API calls) and `_t` (the
                              remember-me token; without it, a refresh of
                              `_forum_session` won't auto-log-in).
                           4. Export them semicolon-separated:
                                export DISCOURSE_COOKIE='_forum_session=<v>; _t=<v>'
                         Notes:
                           - `_t` alone is NOT enough — Discourse .json
                             endpoints reject sessionless requests with a
                             403 "You need to be logged in to do that."
                           - The cookies expire when the CERN SSO session
                             does (typically days to weeks). If you start
                             getting 403s mid-run, re-grab and rerun; the
                             manifest is incremental.
                           - Quote the value in single quotes so the shell
                             doesn't choke on `;` or `=`.
    DISCOURSE_API_KEY + DISCOURSE_USERNAME
                         Proper API key from
                         https://cms-talk.web.cern.ch/admin/api/keys
                         (admin-only; ask a moderator if you need one).
                         More durable than a cookie — survives SSO timeouts.

Usage:
    uv run scripts/scrape_forum.py                # incremental
    uv run scripts/scrape_forum.py --full         # rescrape everything
    uv run scripts/scrape_forum.py --limit 5      # debug, first 5 topics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

import requests

BASE_URL = "https://cms-talk.web.cern.ch"
CATEGORY_PATH = "c/physics/cat/cat-stats"
CATEGORY_ID = 279
DEFAULT_OUTPUT = Path("knowledge_base/forum")
MANIFEST_NAME = ".manifest.json"
POSTS_PER_BATCH = 20  # Discourse batch limit for /t/{id}/posts.json

# Topics to never scrape (meta/admin threads that aren't real Q&A).
# Add IDs here as you find them.
SKIP_TOPIC_IDS: frozenset[int] = frozenset({
    19755,  # subscribers list
})


class _HTMLStrip(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []

    def handle_data(self, data: str) -> None:
        self._buf.append(data)

    def text(self) -> str:
        return "".join(self._buf)


def strip_html(html: str) -> str:
    s = _HTMLStrip()
    s.feed(html)
    raw = s.text()
    cleaned: list[str] = []
    prev_blank = False
    for ln in (line.strip() for line in raw.splitlines()):
        if ln:
            cleaned.append(ln)
            prev_blank = False
        elif not prev_blank:
            cleaned.append("")
            prev_blank = True
    return "\n".join(cleaned).strip()


def make_session() -> requests.Session:
    cookie = os.environ.get("DISCOURSE_COOKIE")
    api_key = os.environ.get("DISCOURSE_API_KEY")
    api_username = os.environ.get("DISCOURSE_USERNAME")
    if not cookie and not (api_key and api_username):
        raise SystemExit(
            "ERROR: set DISCOURSE_COOKIE, or DISCOURSE_API_KEY + DISCOURSE_USERNAME"
        )
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "combine-bot-scraper/0.1",
        }
    )
    if cookie:
        s.headers["Cookie"] = cookie
    else:
        s.headers["Api-Key"] = api_key
        s.headers["Api-Username"] = api_username
    return s


def get_json(
    session: requests.Session, url: str, sleep_s: float, max_retries: int = 5
) -> dict[str, Any]:
    for attempt in range(max_retries):
        r = session.get(url, timeout=30)
        if r.status_code == 429:
            wait = min(60, 2**attempt)
            print(f"  rate-limited, sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        time.sleep(sleep_s)
        return r.json()
    raise RuntimeError(f"too many retries for {url}")


def iter_topics(session: requests.Session, sleep_s: float) -> Iterator[dict[str, Any]]:
    page = 0
    while True:
        url = f"{BASE_URL}/{CATEGORY_PATH}/{CATEGORY_ID}.json?page={page}"
        data = get_json(session, url, sleep_s)
        topic_list = data.get("topic_list") or {}
        topics = topic_list.get("topics") or []
        if not topics:
            return
        for t in topics:
            yield t
        if not topic_list.get("more_topics_url"):
            return
        page += 1


def fetch_all_posts(
    session: requests.Session, topic_id: int, sleep_s: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (topic_meta, all_posts) — fetching extra pages if the topic is long."""
    data = get_json(session, f"{BASE_URL}/t/{topic_id}.json", sleep_s)
    if not data:
        return {}, []
    stream = (data.get("post_stream") or {}).get("stream") or []
    posts = list((data.get("post_stream") or {}).get("posts") or [])
    have = {p["id"] for p in posts}
    missing = [pid for pid in stream if pid not in have]
    while missing:
        batch, missing = missing[:POSTS_PER_BATCH], missing[POSTS_PER_BATCH:]
        qs = "&".join(f"post_ids[]={pid}" for pid in batch)
        extra = get_json(session, f"{BASE_URL}/t/{topic_id}/posts.json?{qs}", sleep_s)
        posts.extend((extra.get("post_stream") or {}).get("posts") or [])
    posts.sort(key=lambda p: p.get("post_number", 0))
    return data, posts


def find_accepted_post_number(
    topic: dict[str, Any], posts: list[dict[str, Any]]
) -> int | None:
    """Resolve the accepted-answer post number from the Discourse Solved plugin.

    Tries the topic-level `accepted_answer` blob first, then falls back to
    scanning posts for `accepted_answer: true`. Returns None if the topic
    isn't solved (or the plugin isn't active on this category).
    """
    acc = topic.get("accepted_answer")
    if isinstance(acc, dict) and acc.get("post_number"):
        return acc["post_number"]
    for p in posts:
        if p.get("accepted_answer"):
            return p.get("post_number")
    return None


def render_txt(topic: dict[str, Any], posts: list[dict[str, Any]]) -> str:
    topic_id = topic.get("id")
    accepted = find_accepted_post_number(topic, posts)
    lines = [
        f"TOPIC: {topic.get('title', '')}",
        f"URL: {BASE_URL}/t/{topic_id}",
        f"DATE: {topic.get('created_at', '')}",
        f"SOLVED: {'yes (reply ' + str(accepted) + ')' if accepted else 'no'}",
        "=" * 60,
    ]
    for post in posts:
        n = post.get("post_number", 0)
        if n == 1:
            role = "QUESTION"
        elif n == accepted:
            role = f"REPLY {n} [ACCEPTED ANSWER]"
        else:
            role = f"REPLY {n}"
        lines.append("")
        lines.append(f"{role} (by {post.get('username', '?')} @ {post.get('created_at', '')}):")
        lines.append(strip_html(post.get("cooked", "")))
        lines.append("-" * 40)
    return "\n".join(lines) + "\n"


def build_json(topic: dict[str, Any], posts: list[dict[str, Any]]) -> dict[str, Any]:
    topic_id = topic.get("id")
    accepted = find_accepted_post_number(topic, posts)
    return {
        "topic_id": topic_id,
        "title": topic.get("title"),
        "url": f"{BASE_URL}/t/{topic_id}",
        "category_id": topic.get("category_id"),
        "created_at": topic.get("created_at"),
        "last_posted_at": topic.get("last_posted_at"),
        "tags": topic.get("tags", []),
        "accepted_answer_post_number": accepted,
        "posts": [
            {
                "post_number": p.get("post_number"),
                "username": p.get("username"),
                "name": p.get("name"),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at"),
                "reply_to_post_number": p.get("reply_to_post_number"),
                "is_accepted_answer": p.get("post_number") == accepted,
                "text": strip_html(p.get("cooked", "")),
                "cooked_html": p.get("cooked", ""),
            }
            for p in posts
        ],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"scraped_at": None, "topics": {}}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["scraped_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output directory (default: {DEFAULT_OUTPUT})")
    p.add_argument("--full", action="store_true",
                   help="rescrape every topic, ignoring the manifest")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="seconds between API calls (default: 0.5)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N topics (debug)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / MANIFEST_NAME
    manifest = load_manifest(manifest_path)

    for tid in SKIP_TOPIC_IDS:
        manifest["topics"].pop(str(tid), None)
        for suffix in (".txt", ".json"):
            f = args.output / f"topic_{tid}{suffix}"
            if f.exists():
                f.unlink()

    known = {} if args.full else dict(manifest["topics"])

    session = make_session()
    seen = fetched = skipped = 0
    try:
        for summary in iter_topics(session, args.sleep):
            seen += 1
            if args.limit and seen > args.limit:
                break
            topic_id = summary["id"]
            if topic_id in SKIP_TOPIC_IDS:
                skipped += 1
                continue
            last_posted_at = summary.get("last_posted_at") or summary.get("bumped_at")
            key = str(topic_id)
            prior = known.get(key, {}).get("last_posted_at")
            if prior and prior == last_posted_at:
                skipped += 1
                continue
            print(f"[{seen}] topic {topic_id}: {(summary.get('title') or '')[:70]}")
            topic_meta, posts = fetch_all_posts(session, topic_id, args.sleep)
            if not posts:
                print("  -> empty or 404, skipping", file=sys.stderr)
                continue
            merged = {**summary, **topic_meta}
            (args.output / f"topic_{topic_id}.txt").write_text(render_txt(merged, posts))
            (args.output / f"topic_{topic_id}.json").write_text(
                json.dumps(build_json(merged, posts), indent=2)
            )
            manifest["topics"][key] = {
                "last_posted_at": last_posted_at,
                "title": summary.get("title"),
                "post_count": len(posts),
                "accepted_answer_post_number": find_accepted_post_number(merged, posts),
            }
            fetched += 1
            if fetched % 25 == 0:
                save_manifest(manifest_path, manifest)
    finally:
        save_manifest(manifest_path, manifest)

    print(f"\nDone. seen={seen} fetched={fetched} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
