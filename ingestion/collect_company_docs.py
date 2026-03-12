from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


DEFAULT_KEYWORDS = (
    "whitepaper",
    "research",
    "resources",
    "docs",
    "documentation",
    "press",
    "news",
    "blog",
    "product",
    "technology",
    "solutions",
    "brochure",
    "case-study",
    "case-study",
    "pdf",
)

HTML_EXTENSIONS = {"", ".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}


@dataclass(frozen=True)
class CrawlConfig:
    company_name: str
    start_url: str
    output_dir: Path
    max_pages: int = 40
    timeout_sec: int = 20
    user_agent: str = "company-corpus-collector/0.1"
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS


def parse_args() -> CrawlConfig:
    parser = argparse.ArgumentParser(
        description="Collect public company-authored documents for company_corpus."
    )
    parser.add_argument("--company-name", required=True, help="Company slug or display name.")
    parser.add_argument("--start-url", required=True, help="Company homepage URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/company"),
        help="Base directory to store collected documents.",
    )
    parser.add_argument("--max-pages", type=int, default=40, help="Max HTML pages to crawl.")
    parser.add_argument("--timeout-sec", type=int, default=20, help="HTTP timeout in seconds.")

    args = parser.parse_args()
    return CrawlConfig(
        company_name=args.company_name,
        start_url=args.start_url,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        timeout_sec=args.timeout_sec,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args()
    session = build_session(config)
    collect_company_docs(config, session)


def build_session(config: CrawlConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": config.user_agent})
    return session


def collect_company_docs(config: CrawlConfig, session: requests.Session) -> None:
    company_dir = config.output_dir / slugify(config.company_name)
    docs_dir = company_dir / "docs"
    metadata_dir = company_dir / "metadata"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    start_netloc = urlparse(config.start_url).netloc
    queue = deque([config.start_url])
    visited: set[str] = set()
    saved_count = 0

    while queue and len(visited) < config.max_pages:
        url = queue.popleft()
        normalized = normalize_url(url)
        if normalized in visited:
            continue
        if urlparse(normalized).netloc != start_netloc:
            continue

        visited.add(normalized)
        logger.info("Crawling %s (%s/%s)", normalized, len(visited), config.max_pages)

        try:
            response = session.get(normalized, timeout=config.timeout_sec)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Request failed for %s: %s", normalized, exc)
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if is_pdf_url(normalized) or "application/pdf" in content_type:
            save_binary_document(
                config=config,
                url=normalized,
                content=response.content,
                content_type="pdf",
                metadata_dir=metadata_dir,
                docs_dir=docs_dir,
            )
            saved_count += 1
            continue

        if "text/html" not in content_type and not seems_html(normalized):
            logger.info("Skipping non-HTML resource: %s", normalized)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        save_html_document(
            config=config,
            url=normalized,
            soup=soup,
            metadata_dir=metadata_dir,
            docs_dir=docs_dir,
        )
        saved_count += 1

        for link in extract_candidate_links(soup, normalized, config.keywords):
            if urlparse(link).netloc == start_netloc and normalize_url(link) not in visited:
                queue.append(link)

    logger.info("Collected %s documents under %s", saved_count, company_dir)


def extract_candidate_links(
    soup: BeautifulSoup,
    base_url: str,
    keywords: Iterable[str],
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        absolute_url = normalize_url(urljoin(base_url, href))
        path = absolute_url.lower()
        anchor_text = anchor.get_text(" ", strip=True).lower()

        if not absolute_url.startswith(("http://", "https://")):
            continue
        if absolute_url in seen:
            continue
        if any(keyword in path or keyword in anchor_text for keyword in keywords):
            candidates.append(absolute_url)
            seen.add(absolute_url)

    return candidates


def save_html_document(
    *,
    config: CrawlConfig,
    url: str,
    soup: BeautifulSoup,
    metadata_dir: Path,
    docs_dir: Path,
) -> None:
    title = (soup.title.string or "").strip() if soup.title else ""
    text = soup.get_text("\n", strip=True)
    if not text:
        logger.info("Skipping empty HTML page: %s", url)
        return

    slug = slugify(urlparse(url).path or "index")
    doc_path = docs_dir / f"{slug}.html"
    doc_path.write_text(text, encoding="utf-8")

    metadata = {
        "company": config.company_name,
        "source_type": "company",
        "document_type": infer_document_type(url, title),
        "title": title or url,
        "url": url,
        "source_path": str(doc_path),
    }
    write_metadata(metadata_dir / f"{slug}.json", metadata)


def save_binary_document(
    *,
    config: CrawlConfig,
    url: str,
    content: bytes,
    content_type: str,
    metadata_dir: Path,
    docs_dir: Path,
) -> None:
    slug = slugify(urlparse(url).path)
    suffix = ".pdf" if content_type == "pdf" else ".bin"
    doc_path = docs_dir / f"{slug}{suffix}"
    doc_path.write_bytes(content)

    metadata = {
        "company": config.company_name,
        "source_type": "company",
        "document_type": infer_document_type(url, ""),
        "title": url.split("/")[-1] or url,
        "url": url,
        "source_path": str(doc_path),
    }
    write_metadata(metadata_dir / f"{slug}.json", metadata)


def infer_document_type(url: str, title: str) -> str:
    basis = f"{url} {title}".lower()
    if "whitepaper" in basis:
        return "whitepaper"
    if "press" in basis or "news" in basis:
        return "press_release"
    if "blog" in basis:
        return "tech_blog"
    if "deck" in basis or "pitch" in basis or "investor" in basis:
        return "pitch_deck"
    if "brochure" in basis or "solution" in basis:
        return "brochure"
    return "product_page"


def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean_path = parsed.path or "/"
    return parsed._replace(fragment="", query="", path=clean_path).geturl().rstrip("/")


def is_pdf_url(url: str) -> bool:
    return Path(urlparse(url).path).suffix.lower() in PDF_EXTENSIONS


def seems_html(url: str) -> bool:
    return Path(urlparse(url).path).suffix.lower() in HTML_EXTENSIONS


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = re.sub(r"[^\w]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "document"


if __name__ == "__main__":
    main()
