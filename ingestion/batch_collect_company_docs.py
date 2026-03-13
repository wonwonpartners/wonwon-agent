from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ingestion.collect_company_docs import CrawlConfig, build_session, collect_company_docs, slugify
from retrieval.config import PROJECT_ROOT


logger = logging.getLogger(__name__)

COMPANY_NAME_COLUMNS = ("company_name", "company", "기업명")
HOMEPAGE_URL_COLUMNS = ("homepage_url", "start_url", "url", "홈페이지", "homepage")


@dataclass(frozen=True)
class CompanyHomepageRow:
    company_name: str
    homepage_url: str


@dataclass(frozen=True)
class BatchCollectConfig:
    csv_path: Path
    output_dir: Path
    max_pages: int
    timeout_sec: int
    user_agent: str
    limit: int | None
    request_delay_sec: float
    dry_run: bool
    skip_existing: bool


def parse_args() -> BatchCollectConfig:
    parser = argparse.ArgumentParser(
        description="Batch-run company document collection from a CSV of homepage URLs."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=PROJECT_ROOT / "company_homepage_urls.csv",
        help="CSV file containing company_name/homepage_url columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "company",
        help="Base directory to store collected documents.",
    )
    parser.add_argument("--max-pages", type=int, default=40, help="Max HTML pages to crawl.")
    parser.add_argument("--timeout-sec", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--user-agent",
        default="company-corpus-collector/0.1",
        help="User-Agent header used for HTTP requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N valid CSV rows.",
    )
    parser.add_argument(
        "--request-delay-sec",
        type=float,
        default=0.0,
        help="Sleep between companies to reduce load on target sites.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work plan without making network requests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip companies that already have an output directory.",
    )

    args = parser.parse_args()
    return BatchCollectConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        timeout_sec=args.timeout_sec,
        user_agent=args.user_agent,
        limit=args.limit,
        request_delay_sec=args.request_delay_sec,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args()
    rows = load_company_rows(config.csv_path)
    if config.limit is not None:
        rows = rows[: config.limit]

    logger.info("Loaded %s companies from %s", len(rows), config.csv_path)
    run_batch_collection(rows, config)


def load_company_rows(csv_path: Path) -> list[CompanyHomepageRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return []

    company_column = resolve_column(reader.fieldnames, COMPANY_NAME_COLUMNS)
    homepage_column = resolve_column(reader.fieldnames, HOMEPAGE_URL_COLUMNS)

    parsed_rows: list[CompanyHomepageRow] = []
    for index, row in enumerate(rows, start=2):
        company_name = str(row.get(company_column, "")).strip()
        homepage_url = str(row.get(homepage_column, "")).strip()

        if not company_name:
            logger.warning("Skipping row %s with empty company name.", index)
            continue
        if not homepage_url:
            logger.warning("Skipping row %s for %s because homepage_url is empty.", index, company_name)
            continue

        parsed_rows.append(
            CompanyHomepageRow(company_name=company_name, homepage_url=homepage_url)
        )

    return parsed_rows


def resolve_column(fieldnames: Iterable[str] | None, aliases: tuple[str, ...]) -> str:
    normalized_map = {
        normalize_column_name(fieldname): fieldname for fieldname in (fieldnames or []) if fieldname
    }
    for alias in aliases:
        matched = normalized_map.get(normalize_column_name(alias))
        if matched:
            return matched

    available = ", ".join(fieldnames or [])
    expected = ", ".join(aliases)
    raise ValueError(f"Required CSV column not found. Expected one of [{expected}], found [{available}]")


def normalize_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def run_batch_collection(rows: list[CompanyHomepageRow], config: BatchCollectConfig) -> None:
    base_crawl_config = CrawlConfig(
        company_name="batch",
        start_url="https://example.com",
        output_dir=config.output_dir,
        max_pages=config.max_pages,
        timeout_sec=config.timeout_sec,
        user_agent=config.user_agent,
    )
    session = build_session(base_crawl_config)

    success_count = 0
    skipped_count = 0
    failed: list[str] = []

    for index, row in enumerate(rows, start=1):
        company_dir = config.output_dir / slugify(row.company_name)
        if config.skip_existing and company_dir.exists():
            skipped_count += 1
            logger.info("[%s/%s] Skipping existing company: %s", index, len(rows), row.company_name)
            continue

        logger.info("[%s/%s] %s -> %s", index, len(rows), row.company_name, row.homepage_url)
        if config.dry_run:
            continue

        crawl_config = CrawlConfig(
            company_name=row.company_name,
            start_url=row.homepage_url,
            output_dir=config.output_dir,
            max_pages=config.max_pages,
            timeout_sec=config.timeout_sec,
            user_agent=config.user_agent,
        )

        try:
            collect_company_docs(crawl_config, session)
            success_count += 1
        except Exception as exc:  # pragma: no cover - operational safety net
            failed.append(row.company_name)
            logger.exception("Failed to collect documents for %s: %s", row.company_name, exc)

        if config.request_delay_sec > 0:
            time.sleep(config.request_delay_sec)

    logger.info(
        "Batch collection finished. success=%s skipped=%s failed=%s",
        success_count,
        skipped_count,
        len(failed),
    )
    if failed:
        logger.warning("Failed companies: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
