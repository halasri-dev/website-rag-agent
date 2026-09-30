"""Crawl the Litestream docs (latest version only) into data/pages.jsonl"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE = "https://litestream.io"
USER_AGENT = "website-rag-assessment-bot/0.1 (educational project)"
OUT = Path("data/pages.jsonl")
DELAY_SECONDS = 1.0
SKIP_PARTS = ["/v0.3/", "/tags/", "/categories/"]  # old version + non-doc pages


def get(url):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    return r


def sitemap_urls(url):
    """Return all page URLs listed in a sitemap (handles sitemap indexes)."""
    soup = BeautifulSoup(get(url).text, "xml")
    urls = []
    for sm in soup.find_all("sitemap"):
        urls += sitemap_urls(sm.loc.text.strip())
    urls += [u.loc.text.strip() for u in soup.find_all("url")]
    return urls


def extract(html):
    """Return (title, clean_text) from a page's HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    h1 = main.find("h1")
    title = h1.get_text(strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")

    # remove the "#" heading-anchor links
    for a in main.find_all("a"):
        if a.get_text(strip=True) == "#":
            a.decompose()
    # merge inline tags (links, code, bold...) into the surrounding text
    for tag in main.find_all(["a", "code", "strong", "em", "b", "i", "span", "kbd", "abbr"]):
        tag.unwrap()
    main.smooth()  # join the text pieces so sentences stay whole

    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return title, text


def main():
    OUT.parent.mkdir(exist_ok=True)

    robots = RobotFileParser(BASE + "/robots.txt")
    robots.read()

    domain = urlparse(BASE).netloc
    urls = sorted({urldefrag(u)[0] for u in sitemap_urls(BASE + "/sitemap.xml")})
    urls = [u for u in urls if urlparse(u).netloc == domain and not any(p in u for p in SKIP_PARTS)]
    print(f"Found {len(urls)} candidate pages\n")

    saved = 0
    with OUT.open("w", encoding="utf-8") as f:
        for url in urls:
            if not robots.can_fetch(USER_AGENT, url):
                print(f"SKIPPED (robots.txt)  {url}")
                continue
            try:
                html = get(url).text
            except Exception as e:
                print(f"FAILED  {url}  {e}")
                continue
            title, text = extract(html)
            words = len(text.split())
            f.write(json.dumps({"url": url, "title": title, "text": text, "words": words},
                               ensure_ascii=False) + "\n")
            saved += 1
            print(f"{words:5d} words  {url}")
            time.sleep(DELAY_SECONDS)

    print(f"\nSaved {saved} pages to {OUT}")


if __name__ == "__main__":
    main()