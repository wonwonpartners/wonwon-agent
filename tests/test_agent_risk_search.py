from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing_extensions import NotRequired, TypedDict


# Load Environment
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("TRAVILY_API_KEY")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 또는 OPEN_AI_KEY가 필요합니다.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY가 필요합니다.")

if not all([POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD]):
    raise ValueError("POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD가 필요합니다.")


# State
class RiskDetectionState(TypedDict):
    legal_regulatory: str
    certification_status: list[str]
    red_flags: list[str]
    risk_summary: str


class CompanySearchProfile(TypedDict):
    company_id: str
    company_name: str
    product_name: NotRequired[str]
    description: NotRequired[str]
    invest_level: NotRequired[str]
    aliases: NotRequired[list[str]]
    categories: NotRequired[list[str]]


class WebSignal(TypedDict):
    source_type: str
    title: str
    url: str
    snippet: str
    published_at: str
    query: str


class SnippetScanResult(TypedDict):
    certification_mentions: list[str]
    patent_mentions: list[str]
    compliance_mentions: list[str]
    negative_mentions: list[str]


# Config
NEWS_TERMS = [
    "소송",
    "법적분쟁",
    "안전사고",
    "산재",
    "규제위반",
    "허위과장",
    "논란",
    "장애",
]

WEB_TERMS = [
    "ISO",
    "KC",
    "인증",
    "특허",
    "등록",
    "컴플라이언스",
    "보도자료",
]

CERT_PATTERNS = [r"ISO", r"KC", r"CE", r"인증", r"인증번호"]
PATENT_PATTERNS = [r"특허", r"특허등록", r"출원", r"등록번호"]
COMPLIANCE_PATTERNS = [r"컴플라이언스", r"품질", r"안전", r"모니터링", r"긴급정지", r"fail[- ]?safe"]
NEGATIVE_PATTERNS = [r"소송", r"법적분쟁", r"안전사고", r"산재", r"규제위반", r"허위과장", r"논란", r"장애"]

KOREAN_NEWS_DOMAINS = [
    "yna.co.kr",
    "newsis.com",
    "mk.co.kr",
    "zdnet.co.kr",
]

NEWS_DAYS = 180
NEWS_MAX_RESULTS = 2
WEB_MAX_RESULTS = 2


QUERY_COMPANY_SQL = """
SELECT
    c.company_id,
    c.company_name,
    c.product_name,
    c.description,
    c.invest_level,
    COALESCE(array_remove(array_agg(DISTINCT cat.category_name), NULL), '{}') AS categories
FROM companies c
LEFT JOIN company_categories cc ON cc.company_id = c.company_id
LEFT JOIN categories cat ON cat.category_id = cc.category_id
WHERE c.company_name = %(company_name)s
GROUP BY c.company_id, c.company_name, c.product_name, c.description, c.invest_level
LIMIT 1
"""


# Helpers
def get_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def dedupe_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_support_category(profile: CompanySearchProfile) -> str | None:
    categories = dedupe_keep_order(profile.get("categories", []))
    return categories[0] if categories else None


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY)


def get_news_search_tool() -> TavilySearch:
    return TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
        topic="news",
        max_results=NEWS_MAX_RESULTS,
        search_depth="basic",
    )


def get_web_search_tool() -> TavilySearch:
    return TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
        topic="general",
        max_results=WEB_MAX_RESULTS,
        search_depth="basic",
    )


# NODE - Company Profile
def lookup_company_profile(company_name: str) -> CompanySearchProfile:
    params = {
        "company_name": company_name,
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(QUERY_COMPANY_SQL, params)
            row = cur.fetchone()

    if row is None:
        raise ValueError(f"companies 테이블에서 '{company_name}' 회사를 찾지 못했습니다.")

    profile: CompanySearchProfile = {
        "company_id": row[0],
        "company_name": row[1],
        "aliases": dedupe_keep_order([row[1], company_name]),
        "categories": dedupe_keep_order(list(row[5] or [])),
    }

    if row[2]:
        profile["product_name"] = row[2]
        profile["aliases"] = dedupe_keep_order(profile["aliases"] + [row[2]])
    if row[3]:
        profile["description"] = row[3]
    if row[4]:
        profile["invest_level"] = row[4]

    return profile


# NODE - Query Builder
def build_news_queries(profile: CompanySearchProfile) -> list[str]:
    company_alias = profile["company_name"]
    support_category = build_support_category(profile)
    if support_category:
        queries = [f'"{company_alias}" "{support_category}" "{term}"' for term in NEWS_TERMS]
    else:
        queries = [f'"{company_alias}" "{term}"' for term in NEWS_TERMS]
    return dedupe_keep_order(queries)


def build_web_queries(profile: CompanySearchProfile) -> list[str]:
    company_alias = profile["company_name"]
    support_category = build_support_category(profile)
    if support_category:
        queries = [f'"{company_alias}" "{support_category}" "{term}"' for term in WEB_TERMS]
    else:
        queries = [f'"{company_alias}" "{term}"' for term in WEB_TERMS]
    return dedupe_keep_order(queries)


# NODE - Web Search
def normalize_tavily_results(results: list[dict[str, Any]], source_type: str, query: str) -> list[WebSignal]:
    signals: list[WebSignal] = []
    for item in results:
        signals.append(
            {
                "source_type": source_type,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "published_at": item.get("published_date", ""),
                "query": query,
            }
        )
    return signals


def run_tavily_news_search(queries: list[str]) -> list[WebSignal]:
    search = get_news_search_tool()
    signals: list[WebSignal] = []
    for query in queries:
        response = search.invoke(
            {
                "query": query,
                "days": NEWS_DAYS,
                "include_domains": KOREAN_NEWS_DOMAINS,
            }
        )
        signals.extend(normalize_tavily_results(response.get("results", []), "news", query))
    return signals


def run_tavily_web_search(queries: list[str]) -> list[WebSignal]:
    search = get_web_search_tool()
    signals: list[WebSignal] = []
    for query in queries:
        response = search.invoke({"query": query})
        signals.extend(normalize_tavily_results(response.get("results", []), "web", query))
    return signals


def is_company_relevant(signal: WebSignal, profile: CompanySearchProfile) -> bool:
    aliases = profile.get("aliases", [profile["company_name"]])
    haystack = " ".join([signal["title"], signal["snippet"], signal["url"]]).lower()
    return any(alias.lower() in haystack for alias in aliases)


def filter_company_relevant_signals(signals: list[WebSignal], profile: CompanySearchProfile) -> list[WebSignal]:
    return [signal for signal in signals if is_company_relevant(signal, profile)]


# NODE - Signal Scan
def find_pattern_mentions(signals: list[WebSignal], patterns: list[str]) -> list[str]:
    found = []
    for signal in signals:
        haystack = f"{signal['title']} {signal['snippet']}"
        for pattern in patterns:
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                found.append(f"{pattern}: {signal['title']}")
    return list(dict.fromkeys(found))


def scan_signals(news_signals: list[WebSignal], web_signals: list[WebSignal]) -> SnippetScanResult:
    return {
        "certification_mentions": find_pattern_mentions(web_signals, CERT_PATTERNS),
        "patent_mentions": find_pattern_mentions(web_signals, PATENT_PATTERNS),
        "compliance_mentions": find_pattern_mentions(web_signals, COMPLIANCE_PATTERNS),
        "negative_mentions": find_pattern_mentions(news_signals, NEGATIVE_PATTERNS),
    }


def normalize_signals(news_signals: list[WebSignal], web_signals: list[WebSignal]) -> list[WebSignal]:
    combined = news_signals + web_signals
    deduped: list[WebSignal] = []
    seen = set()
    for item in combined:
        key = item["url"] or item["title"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    news_items = [item for item in deduped if item["source_type"] == "news"]
    web_items = [item for item in deduped if item["source_type"] == "web"]
    news_items.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return news_items + web_items


# NODE - Prompt / LLM
def build_signal_context(signals: list[WebSignal], scan_result: SnippetScanResult, profile: CompanySearchProfile) -> str:
    lines = ["[COMPANY PROFILE]"]
    lines.append(f"company_id: {profile['company_id']}")
    lines.append(f"company_name: {profile['company_name']}")
    lines.append(f"product_name: {profile.get('product_name', '')}")
    lines.append(f"invest_level: {profile.get('invest_level', '')}")
    lines.append(f"description: {profile.get('description', '')}")
    lines.append(f"categories: {profile.get('categories', [])}")
    lines.append("")

    for idx, item in enumerate(signals, start=1):
        lines.append(f"[SIGNAL {idx}]")
        lines.append(f"source_type: {item['source_type']}")
        lines.append(f"query: {item['query']}")
        lines.append(f"title: {item['title']}")
        lines.append(f"published_at: {item['published_at']}")
        lines.append(f"url: {item['url']}")
        lines.append(f"snippet: {item['snippet']}")
        lines.append("")

    lines.append("[SCAN RESULT]")
    lines.append(f"certification_mentions: {scan_result['certification_mentions']}")
    lines.append(f"patent_mentions: {scan_result['patent_mentions']}")
    lines.append(f"compliance_mentions: {scan_result['compliance_mentions']}")
    lines.append(f"negative_mentions: {scan_result['negative_mentions']}")
    return "\n".join(lines)


def build_risk_prompt(company_name: str, context: str) -> str:
    return f"""
You are a risk detection agent for venture investment due diligence.

Company: {company_name}

Your task is to read Tavily news signals and general web snippet signals, then fill the state below.

Interpretation guide:
- `source_type: news` indicates public negative signals such as lawsuits, incidents, controversy, outages, or regulation-related reports.
- `source_type: web` indicates official or semi-official public traces such as certification mentions, patent mentions, and press pages.
- `SCAN RESULT` is a heuristic summary extracted from snippets and should be used conservatively.
- `COMPANY PROFILE` is internal metadata from the company RDB and should only be used to disambiguate the company and interpret signals.

Output rules:
- Use only the provided profile and signals.
- Do not invent legal outcomes or certifications.
- `legal_regulatory` should summarize legal or regulatory status in one concise paragraph.
- `certification_status` should summarize certification, patent, or compliance-related public mentions found in snippets, or say '확인 불가'.
- `red_flags` should list concrete negative signals.
- `risk_summary` should provide an overall summary in Korean.
- Return valid JSON only.

Return this exact schema:
{{
  "legal_regulatory": "string",
  "certification_status": ["string"],
  "red_flags": ["string"],
  "risk_summary": "string"
}}

Signals:
{context}
""".strip()


def run_risk_agent(company_name: str) -> RiskDetectionState:
    profile = lookup_company_profile(company_name)
    news_queries = build_news_queries(profile)
    web_queries = build_web_queries(profile)

    news_signals = filter_company_relevant_signals(run_tavily_news_search(news_queries), profile)
    web_signals = filter_company_relevant_signals(run_tavily_web_search(web_queries), profile)

    scan_result = scan_signals(news_signals, web_signals)
    signals = normalize_signals(news_signals, web_signals)
    context = build_signal_context(signals, scan_result, profile)
    prompt = build_risk_prompt(profile["company_name"], context)

    llm = get_llm()
    response = llm.invoke(prompt)
    return json.loads(response.content)


if __name__ == "__main__":
    company_name = "엑스와이지"
    risk_state = run_risk_agent(company_name)
    print(json.dumps(risk_state, ensure_ascii=False, indent=2))
