from __future__ import annotations

import logging
import re
from urllib.parse import urlparse
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, Field

from agents.agent_risk_search.common import (
    DOMAIN_NAME_MAP,
    KOREAN_NEWS_DOMAINS,
    NEWS_DAYS,
    get_fallback_chat_model,
    get_chat_model,
    get_news_search_tool,
    get_web_search_tool,
)
from agents.agent_risk_search.prompts import get_system_prompt, render_user_prompt
from agents.workflow_common import ResearchAgentState, get_company_id, get_company_name
from utils.openai_fallback import invoke_with_rate_limit_fallback

AGENT_NAME = "agent_risk_search"
logger = logging.getLogger(__name__)

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


class CompanyProfile(TypedDict):
    company_id: str
    company_name: str
    product_name: str
    description: str
    invest_level: str
    categories: list[str]
    aliases: list[str]


class WebSignal(TypedDict):
    source_type: str
    title: str
    url: str
    site_name: str
    snippet: str
    published_at: str
    query: str


class SnippetScanResult(TypedDict):
    certification_mentions: list[str]
    patent_mentions: list[str]
    compliance_mentions: list[str]
    negative_mentions: list[str]


class RiskDetectionState(TypedDict):
    legal_regulatory: str
    certification_status: list[str]
    red_flags: list[str]
    risk_summary: str


class RiskDetectionOutput(BaseModel):
    legal_regulatory: str = Field(default="확인 불가")
    certification_status: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    risk_summary: str = Field(default="확인 불가")


class StructuredOutputPayload(TypedDict):
    risk_state: RiskDetectionState
    company_profile: CompanyProfile
    search_queries: dict[str, list[str]]
    scan_result: SnippetScanResult
    support_category: NotRequired[str]
    signal_count: int


def run_agent_risk_search(
    selected_company: dict[str, Any] | None,
    previous_state: ResearchAgentState | None = None,
) -> ResearchAgentState:
    attempt_count = int((previous_state or {}).get("attempt_count", 0)) + 1
    company_id = get_company_id(selected_company)
    company_name = get_company_name(selected_company)

    if company_id is None:
        return {
            "agent_name": AGENT_NAME,
            "status": "skipped",
            "attempt_count": attempt_count,
            "input_company_id": None,
            "summary": "선정된 회사가 없어 리스크 탐지를 진행하지 못했습니다.",
            "findings": [
                "선행 단계에서 `selected_company`가 비어 있어 리스크 탐지를 건너뛰었습니다.",
            ],
            "sources": [],
            "structured_output": None,
        }

    company_profile = build_company_profile(selected_company)
    news_queries = build_news_queries(company_profile)
    web_queries = build_web_queries(company_profile)
    search_queries = {
        "news": news_queries,
        "web": web_queries,
    }

    signals: list[WebSignal] = []
    try:
        news_signals = filter_company_relevant_signals(
            run_tavily_news_search(news_queries),
            company_profile,
        )
        web_signals = filter_company_relevant_signals(
            run_tavily_web_search(web_queries),
            company_profile,
        )
        signals = normalize_signals(news_signals, web_signals)
        scan_result = scan_signals(news_signals, web_signals)
        risk_state = build_risk_state(company_profile, signals, scan_result)
        findings = build_findings(risk_state, scan_result)
        summary = risk_state["risk_summary"] or f"{company_name}의 리스크 탐지를 완료했습니다."
        structured_output: StructuredOutputPayload = {
            "risk_state": risk_state,
            "company_profile": company_profile,
            "search_queries": search_queries,
            "scan_result": scan_result,
            "support_category": build_support_category(company_profile) or "",
            "signal_count": len(signals),
        }
        return {
            "agent_name": AGENT_NAME,
            "status": "completed",
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": summary,
            "findings": findings,
            "sources": [dict(signal) for signal in signals],
            "structured_output": structured_output,
        }
    except Exception as exc:
        logger.exception(
            "[%s/error] company=%s(%s) message=%s",
            AGENT_NAME,
            company_name,
            company_id,
            exc,
        )
        return {
            "agent_name": AGENT_NAME,
            "status": "failed",
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": f"{company_name} ({company_id})의 리스크 탐지 중 오류가 발생했습니다.",
            "findings": [
                "Tavily 검색 또는 LLM structured extraction 단계에서 오류가 발생했습니다.",
                f"오류 메시지: {str(exc)}",
            ],
            "sources": [dict(signal) for signal in signals],
            "structured_output": {
                "risk_state": {
                    "legal_regulatory": "실행 오류로 확인하지 못했습니다.",
                    "certification_status": ["실행 오류로 확인하지 못했습니다."],
                    "red_flags": ["실행 오류 발생"],
                    "risk_summary": f"실행 오류로 리스크 탐지를 완료하지 못했습니다. 오류: {str(exc)}",
                },
                "company_profile": company_profile,
                "search_queries": search_queries,
                "scan_result": {
                    "certification_mentions": [],
                    "patent_mentions": [],
                    "compliance_mentions": [],
                    "negative_mentions": [],
                },
                "support_category": build_support_category(company_profile) or "",
                "signal_count": len(signals),
            },
        }


def build_company_profile(selected_company: dict[str, Any] | None) -> CompanyProfile:
    company_id = get_company_id(selected_company) or ""
    company_name = get_company_name(selected_company)
    product_name = str((selected_company or {}).get("product_name") or "").strip()
    description = str((selected_company or {}).get("description") or "").strip()
    invest_level = str((selected_company or {}).get("invest_level") or "").strip()
    raw_categories = (selected_company or {}).get("categories") or []
    categories = [str(category).strip() for category in raw_categories if str(category).strip()]
    aliases = [alias for alias in [company_name, product_name] if alias]
    aliases = list(dict.fromkeys(aliases))
    return {
        "company_id": company_id,
        "company_name": company_name,
        "product_name": product_name,
        "description": description,
        "invest_level": invest_level,
        "categories": categories,
        "aliases": aliases,
    }


def build_support_category(profile: CompanyProfile) -> str | None:
    return profile["categories"][0] if profile.get("categories") else None


def build_news_queries(profile: CompanyProfile) -> list[str]:
    company_name = profile["company_name"]
    support_category = build_support_category(profile)
    if support_category:
        return [f'"{company_name}" "{support_category}" "{term}"' for term in NEWS_TERMS]
    return [f'"{company_name}" "{term}"' for term in NEWS_TERMS]


def build_web_queries(profile: CompanyProfile) -> list[str]:
    company_name = profile["company_name"]
    support_category = build_support_category(profile)
    if support_category:
        return [f'"{company_name}" "{support_category}" "{term}"' for term in WEB_TERMS]
    return [f'"{company_name}" "{term}"' for term in WEB_TERMS]


def _extract_site_name(url: str) -> str:
    try:
        host = urlparse(url).netloc.removeprefix("www.")
        return DOMAIN_NAME_MAP.get(host, host)
    except Exception:
        return ""


def normalize_tavily_results(results: list[dict[str, Any]], source_type: str, query: str) -> list[WebSignal]:
    signals: list[WebSignal] = []
    for item in results:
        url = item.get("url", "")
        signals.append(
            {
                "source_type": source_type,
                "title": item.get("title", ""),
                "url": url,
                "site_name": _extract_site_name(url),
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
        raw_results = response if isinstance(response, list) else response.get("results", [])
        signals.extend(normalize_tavily_results(raw_results, "news", query))
    return signals


def run_tavily_web_search(queries: list[str]) -> list[WebSignal]:
    search = get_web_search_tool()
    signals: list[WebSignal] = []
    for query in queries:
        response = search.invoke({"query": query})
        raw_results = response if isinstance(response, list) else response.get("results", [])
        signals.extend(normalize_tavily_results(raw_results, "web", query))
    return signals


def is_company_relevant(signal: WebSignal, profile: CompanyProfile) -> bool:
    haystack = " ".join([signal["title"], signal["snippet"], signal["url"]]).lower()
    return any(alias.lower() in haystack for alias in profile.get("aliases", []))


def filter_company_relevant_signals(signals: list[WebSignal], profile: CompanyProfile) -> list[WebSignal]:
    return [signal for signal in signals if is_company_relevant(signal, profile)]


def find_pattern_mentions(signals: list[WebSignal], patterns: list[str]) -> list[str]:
    found: list[str] = []
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
    seen: set[str] = set()
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


def build_risk_state(
    company_profile: CompanyProfile,
    signals: list[WebSignal],
    scan_result: SnippetScanResult,
) -> RiskDetectionState:
    payload = [
        ("system", get_system_prompt()),
        (
            "user",
            render_user_prompt(
                company_profile=company_profile,
                signals=[dict(signal) for signal in signals],
                scan_result=scan_result,
            ),
        ),
    ]
    result = invoke_with_rate_limit_fallback(
        payload=payload,
        primary_factory=lambda: get_chat_model().with_structured_output(
            RiskDetectionOutput,
            method="json_schema",
        ),
        fallback_factory=lambda: get_fallback_chat_model().with_structured_output(
            RiskDetectionOutput,
            method="json_schema",
        ),
        logger=logger,
        operation_name="agent_risk_search.build_risk_state",
    )
    certification_status = result.certification_status or ["확인 불가"]
    return {
        "legal_regulatory": result.legal_regulatory,
        "certification_status": certification_status,
        "red_flags": result.red_flags,
        "risk_summary": result.risk_summary,
    }


def build_findings(
    risk_state: RiskDetectionState,
    scan_result: SnippetScanResult,
) -> list[str]:
    findings: list[str] = []
    findings.extend(risk_state.get("red_flags", []))
    if risk_state.get("certification_status"):
        findings.extend(risk_state["certification_status"])
    if not findings:
        findings.append("공개 검색 기준으로 즉각적인 법적/규제 리스크 신호를 확인하지 못했습니다.")
    if scan_result["negative_mentions"]:
        findings.append(f"negative_signals={len(scan_result['negative_mentions'])}건")
    return list(dict.fromkeys(findings))
