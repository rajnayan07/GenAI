"""
Scraper for GitLab Handbook and Direction pages.

Fetches content from GitLab's public handbook and direction pages,
extracts meaningful text, and saves it as structured JSON for the
RAG pipeline.

Usage:
    python scripts/scrape_gitlab.py
"""

import json
import time
import re
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HANDBOOK_SEED_URLS = [
    "https://handbook.gitlab.com/handbook/values/",
    "https://handbook.gitlab.com/handbook/communication/",
    "https://handbook.gitlab.com/handbook/leadership/",
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/engineering/",
    "https://handbook.gitlab.com/handbook/engineering/development/",
    "https://handbook.gitlab.com/handbook/engineering/infrastructure/",
    "https://handbook.gitlab.com/handbook/engineering/quality/",
    "https://handbook.gitlab.com/handbook/engineering/security/",
    "https://handbook.gitlab.com/handbook/product/",
    "https://handbook.gitlab.com/handbook/product/product-principles/",
    "https://handbook.gitlab.com/handbook/product/ux/",
    "https://handbook.gitlab.com/handbook/sales/",
    "https://handbook.gitlab.com/handbook/marketing/",
    "https://handbook.gitlab.com/handbook/finance/",
    "https://handbook.gitlab.com/handbook/legal/",
    "https://handbook.gitlab.com/handbook/it/",
    "https://handbook.gitlab.com/handbook/security/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/guide/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/getting-started/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/tips/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/meetings/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/management/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/informal-communication/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/hybrid-remote/",
    "https://handbook.gitlab.com/handbook/total-rewards/compensation/",
    "https://handbook.gitlab.com/handbook/total-rewards/benefits/",
    "https://handbook.gitlab.com/handbook/hiring/",
    "https://handbook.gitlab.com/handbook/hiring/interviewing/",
    "https://handbook.gitlab.com/handbook/people-group/onboarding/",
    "https://handbook.gitlab.com/handbook/teamops/",
]

DIRECTION_SEED_URLS = [
    "https://about.gitlab.com/direction/",
    "https://about.gitlab.com/direction/maturity/",
    "https://about.gitlab.com/direction/plan/",
    "https://about.gitlab.com/direction/create/",
    "https://about.gitlab.com/direction/verify/",
    "https://about.gitlab.com/direction/package/",
    "https://about.gitlab.com/direction/secure/",
    "https://about.gitlab.com/direction/govern/",
    "https://about.gitlab.com/direction/release/",
    "https://about.gitlab.com/direction/configure/",
    "https://about.gitlab.com/direction/monitor/",
    "https://about.gitlab.com/direction/modelops/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 1.5  # seconds between requests to be respectful


def fetch_page(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def extract_handbook_content(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
    if not main:
        main = soup.find("body")
    if not main:
        return None

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.sub(r"\s*\|.*$", "", title).strip()

    h1 = main.find("h1")
    if h1 and not title:
        title = h1.get_text(strip=True)

    paragraphs = []
    for elem in main.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "blockquote"]):
        text = elem.get_text(separator=" ", strip=True)
        if len(text) > 20:
            paragraphs.append(text)

    content = "\n\n".join(paragraphs)

    if len(content) < 100:
        logger.info(f"Skipping {url}: content too short ({len(content)} chars)")
        return None

    section = categorize_url(url)

    return {
        "title": title or url.split("/")[-2].replace("-", " ").title(),
        "url": url,
        "section": section,
        "content": content,
    }


def categorize_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if "direction" in path:
        return "Direction"
    if "all-remote" in path or "culture" in path:
        return "Remote Work & Culture"
    if "engineering" in path:
        return "Engineering"
    if "product" in path:
        return "Product"
    if "security" in path:
        return "Security"
    if "people" in path or "hiring" in path or "onboarding" in path:
        return "People & Hiring"
    if "values" in path:
        return "Values"
    if "communication" in path:
        return "Communication"
    if "total-rewards" in path or "compensation" in path or "benefits" in path:
        return "Total Rewards"
    if "leadership" in path:
        return "Leadership"
    if "sales" in path:
        return "Sales"
    if "marketing" in path:
        return "Marketing"
    if "finance" in path or "legal" in path:
        return "Finance & Legal"
    return "General"


def discover_links(html: str, base_url: str, domain_filter: str) -> list[str]:
    """Find additional handbook links from a page."""
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if domain_filter in clean_url and clean_url.endswith("/"):
            links.add(clean_url)
    return list(links)


def scrape_all(max_pages: int = 80) -> list[dict]:
    all_urls = list(set(HANDBOOK_SEED_URLS + DIRECTION_SEED_URLS))
    visited = set()
    documents = []

    urls_to_visit = list(all_urls)

    while urls_to_visit and len(visited) < max_pages:
        url = urls_to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        logger.info(f"[{len(visited)}/{max_pages}] Fetching: {url}")
        html = fetch_page(url)
        if not html:
            continue

        doc = extract_handbook_content(html, url)
        if doc:
            documents.append(doc)
            logger.info(f"  -> Extracted: {doc['title']} ({len(doc['content'])} chars)")

        if len(visited) < max_pages // 2:
            domain = "handbook.gitlab.com" if "handbook" in url else "about.gitlab.com/direction"
            new_links = discover_links(html, url, domain)
            for link in new_links:
                if link not in visited and link not in urls_to_visit:
                    urls_to_visit.append(link)

        time.sleep(REQUEST_DELAY)

    logger.info(f"Scraping complete. Collected {len(documents)} documents from {len(visited)} pages.")
    return documents


def main():
    output_path = Path(__file__).parent.parent / "data" / "gitlab_docs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    documents = scrape_all(max_pages=80)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(documents)} documents to {output_path}")

    total_chars = sum(len(d["content"]) for d in documents)
    logger.info(f"Total content: {total_chars:,} characters across {len(documents)} documents")


if __name__ == "__main__":
    main()
