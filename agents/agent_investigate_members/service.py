from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, NotRequired, TypedDict
from urllib.parse import urlparse

from agents.agent_investigate_members.common import (
    MAX_KEY_MEMBERS,
    MAX_RESULTS_PER_QUERY_FAMILY,
    MAX_TOTAL_SIGNALS,
    QUERY_FAMILY_TERMS,
    ROLE_TAXONOMY,
    get_chat_model,
    get_web_search_tool,
)
from agents.agent_investigate_members.output import (
    InvestigateMemberProfile,
    InvestigateMembersPayload,
    InvestigateMembersRoleCoverage,
)
from agents.agent_investigate_members.prompts import get_system_prompt, render_user_prompt
from agents.agent_investigate_members.result import (
    InvestigateMemberExtraction,
    InvestigateMembersExtractionResult,
)
from agents.workflow_common import ResearchAgentState, get_company_id, get_company_name

logger = logging.getLogger(__name__)


AGENT_NAME = "investigate_members"

ROLE_LABELS = {
    "robot_hw": "로봇 HW",
    "robot_sw_ai": "로봇 SW/AI",
    "control_perception": "제어/인지",
    "system_integration": "시스템 통합",
    "productization_deployment": "제품화/현장 배치",
    "manufacturing_operations": "제조/운영",
    "business_development": "사업개발",
}

SOURCE_TYPE_PRIORITY = {
    "official": 4,
    "news": 3,
    "professional": 2,
    "general": 1,
}
PROFESSIONAL_DOMAIN_HINTS = (
    "linkedin.com",
    "rocketpunch.com",
    "wanted.co.kr",
    "thevc.kr",
)
NEWS_DOMAIN_HINTS = (
    "news",
    "press",
    "hankyung.com",
    "mk.co.kr",
    "sedaily.com",
    "etnews.com",
    "zdnet.co.kr",
    "platum.kr",
    "venturesquare.net",
    "donga.com",
    "joongang.co.kr",
    "chosun.com",
    "naver.com",
)
OFFICIAL_PATH_HINTS = (
    "/about",
    "/team",
    "/leadership",
    "/people",
    "/company",
    "/management",
    "/about-us",
)
ROLE_SIGNAL_TERMS = (
    "ceo",
    "대표",
    "창업자",
    "founder",
    "co-founder",
    "cofounder",
    "cto",
    "coo",
    "cpo",
    "cso",
    "head",
    "총괄",
    "임원",
    "이사",
    "리더십",
    "경영진",
    "핵심팀",
    "leadership",
    "executive",
    "management",
    "team",
)
OFFICIAL_PAGE_TERMS = (
    "회사 소개",
    "팀 소개",
    "리더십",
    "경영진",
    "핵심팀",
    "about",
    "leadership",
    "team",
    "people",
    "management",
)
NEWS_SIGNAL_TERMS = (
    "인터뷰",
    "기사",
    "보도자료",
    "news",
    "press",
    "article",
)
LOW_VALUE_TERMS = (
    "채용",
    "career",
    "careers",
    "jobs",
    "job opening",
    "recruit",
)
QUERY_FAMILY_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "ceo_founder": ("ceo", "대표", "창업자", "founder"),
    "leadership_team": ("리더십", "경영진", "핵심팀", "leadership", "team"),
    "executive_roles": ("cto", "coo", "cpo", "head", "총괄", "이사", "연구소장"),
    "robotics_expertise": ("로봇", "robot", "ai", "연구", "개발", "기술"),
    "deployment_business": ("운영", "사업", "제품", "deployment", "operations"),
}


class CompanyProfile(TypedDict):
    company_id: str
    company_name: str
    product_name: str
    description: str


class SearchSignal(TypedDict):
    title: str
    url: str
    snippet: str
    published_at: str
    query: str
    source_kind: str
    source_type: str
    query_family: str
    domain: str
    relevance_score: float
    source_id: NotRequired[str]


class SearchQueryPlan(TypedDict):
    query: str
    source_type: Literal["official", "news", "professional", "general"]
    topic: Literal["general", "news", "finance"]


def run_investigate_members(
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
            "summary": "선정된 회사가 없어 구성원 조사를 진행하지 못했습니다.",
            "findings": [
                "선행 단계에서 `selected_company`가 비어 있어 CEO 및 핵심팀 조사를 건너뛰었습니다.",
            ],
            "sources": [],
            "structured_output": None,
        }

    company_profile = build_company_profile(selected_company)
    search_queries = build_search_queries(company_profile)
    signals: list[SearchSignal] = []

    try:
        signals = collect_investigate_member_signals(company_profile, search_queries)
        extraction = (
            build_empty_extraction()
            if not signals
            else extract_investigate_members(company_profile, signals)
        )
        payload = build_investigate_members_payload(
            search_queries=search_queries,
            signals=signals,
            extraction=extraction,
        )
        status = determine_research_status(payload, signals)
        findings = build_findings(status, payload, signals)
        summary = build_summary(company_name, status, payload, signals)
        return {
            "agent_name": AGENT_NAME,
            "status": status,
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": summary,
            "findings": findings,
            "sources": [dict(signal) for signal in signals],
            "structured_output": payload,
        }
    except Exception as exc:
        logger.exception(
            "[%s/error] company=%s(%s) message=%s",
            AGENT_NAME,
            company_name,
            company_id,
            exc,
        )
        payload = build_error_payload(search_queries, str(exc))
        return {
            "agent_name": AGENT_NAME,
            "status": "failed",
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": (
                f"{company_name} ({company_id})의 CEO 및 핵심팀 조사를 실행하는 중 오류가 발생했습니다."
            ),
            "findings": [
                "웹 검색 또는 LLM structured extraction 단계에서 오류가 발생했습니다.",
                f"오류 메시지: {str(exc)}",
                "환경변수 또는 외부 API 상태를 확인한 뒤 재시도해야 합니다.",
            ],
            "sources": [dict(signal) for signal in signals],
            "structured_output": payload,
        }


def build_company_profile(selected_company: dict[str, Any] | None) -> CompanyProfile:
    company_id = get_company_id(selected_company) or ""
    company_name = get_company_name(selected_company)
    product_name = str((selected_company or {}).get("product_name") or "").strip()
    description = str((selected_company or {}).get("description") or "").strip()
    return {
        "company_id": company_id,
        "company_name": company_name,
        "product_name": product_name,
        "description": description,
    }


def build_search_queries(company_profile: CompanyProfile) -> dict[str, str]:
    aliases = build_company_aliases(company_profile)

    queries: dict[str, str] = {}
    for family, terms in QUERY_FAMILY_TERMS.items():
        query_parts = [f'"{alias}"' for alias in aliases]
        query_parts.extend(terms)
        queries[family] = " ".join(query_parts)
    return queries


def build_search_query_variants(
    company_profile: CompanyProfile,
    search_queries: dict[str, str],
) -> dict[str, list[SearchQueryPlan]]:
    company_name = company_profile["company_name"].strip()
    product_name = company_profile["product_name"].strip()
    variants: dict[str, list[SearchQueryPlan]] = {}
    for family, primary_query in search_queries.items():
        family_variants: list[SearchQueryPlan] = [
            {
                "query": primary_query,
                "source_type": "general",
                "topic": "general",
            }
        ]

        if family == "leadership_team":
            family_variants.extend(
                [
                    {
                        "query": f'"{company_name}" 회사 소개 팀 소개 리더십 경영진',
                        "source_type": "official",
                        "topic": "general",
                    },
                    {
                        "query": f'"{company_name}" leadership team about management',
                        "source_type": "official",
                        "topic": "general",
                    },
                    {
                        "query": f'"{company_name}" 핵심팀 임원 인터뷰 기사',
                        "source_type": "news",
                        "topic": "news",
                    },
                    {
                        "query": f'"{company_name}" 리더십 핵심팀 임원 site:linkedin.com',
                        "source_type": "professional",
                        "topic": "general",
                    },
                ]
            )
        elif family == "executive_roles":
            family_variants.extend(
                [
                    {
                        "query": f'"{company_name}" CTO COO CPO Head 총괄 팀 소개',
                        "source_type": "official",
                        "topic": "general",
                    },
                    {
                        "query": f'"{company_name}" CTO COO CPO 총괄 인터뷰 기사',
                        "source_type": "news",
                        "topic": "news",
                    },
                    {
                        "query": f'"{company_name}" CTO COO CPO 연구소장 이사 팀장 site:linkedin.com',
                        "source_type": "professional",
                        "topic": "general",
                    },
                ]
            )
        elif family == "deployment_business":
            family_variants.extend(
                [
                    {
                        "query": f'"{company_name}" 사업 운영 제품 총괄 팀 소개',
                        "source_type": "official",
                        "topic": "general",
                    },
                    {
                        "query": f'"{company_name}" 사업 운영 총괄 인터뷰 기사',
                        "source_type": "news",
                        "topic": "news",
                    },
                    {
                        "query": f'"{company_name}" 사업 운영 제품 총괄 팀 site:linkedin.com',
                        "source_type": "professional",
                        "topic": "general",
                    },
                ]
            )
        elif family == "robotics_expertise":
            family_variants.extend(
                [
                    {
                        "query": f'"{company_name}" 로봇 AI 연구 개발 팀 소개',
                        "source_type": "official",
                        "topic": "general",
                    },
                    {
                        "query": f'"{company_name}" 로봇 AI 연구개발 인터뷰 기사',
                        "source_type": "news",
                        "topic": "news",
                    },
                    {
                        "query": f'"{company_name}" robotics AI engineer researcher site:linkedin.com',
                        "source_type": "professional",
                        "topic": "general",
                    },
                ]
            )
        elif family == "ceo_founder":
            family_variants.extend(
                [
                    {
                        "query": f'"{company_name}" 회사 소개 대표 CEO',
                        "source_type": "official",
                        "topic": "general",
                    },
                    {
                        "query": f'"{company_name}" 대표 인터뷰 창업자 기사',
                        "source_type": "news",
                        "topic": "news",
                    },
                ]
            )
            if product_name:
                family_variants.append(
                    {
                        "query": f'"{company_name}" "{product_name}" 대표 창업자',
                        "source_type": "news",
                        "topic": "news",
                    }
                )

        variants[family] = dedupe_query_plans(family_variants)
    return variants


def collect_investigate_member_signals(
    company_profile: CompanyProfile,
    search_queries: dict[str, str],
) -> list[SearchSignal]:
    search = get_web_search_tool()
    collected: list[SearchSignal] = []
    seen_keys: set[str] = set()
    query_variants = build_search_query_variants(company_profile, search_queries)

    for family, plans in query_variants.items():
        family_candidates: list[SearchSignal] = []
        attempted_queries: list[str] = []

        for plan in plans:
            attempted_queries.append(f"{plan['source_type']}::{plan['query']}")
            response = search.invoke(
                {
                    "query": plan["query"],
                    "topic": plan["topic"],
                }
            )
            raw_results = extract_search_results(response)
            family_results = normalize_search_results(
                raw_results,
                query=plan["query"],
                query_family=family,
                source_type_hint=plan["source_type"],
                source_kind="web",
                company_profile=company_profile,
            )
            family_candidates.extend(
                signal
                for signal in family_results
                if is_relevant_signal(signal, company_profile)
            )

        selected_signals = select_family_signals(
            family_candidates,
            seen_keys=seen_keys,
            limit=min(
                MAX_RESULTS_PER_QUERY_FAMILY,
                MAX_TOTAL_SIGNALS - len(collected),
            ),
        )
        collected.extend(selected_signals)

        logger.info(
            "[%s/search] family=%s selected=%s source_types=%s queries=%s",
            AGENT_NAME,
            family,
            len(selected_signals),
            ", ".join(signal["source_type"] for signal in selected_signals) or "-",
            " | ".join(attempted_queries),
        )
        if len(collected) >= MAX_TOTAL_SIGNALS:
            break

    return attach_source_ids(collected[:MAX_TOTAL_SIGNALS])


def normalize_search_results(
    results: list[dict[str, Any]],
    *,
    query: str,
    query_family: str,
    source_type_hint: str,
    source_kind: str,
    company_profile: CompanyProfile,
) -> list[SearchSignal]:
    normalized: list[SearchSignal] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        snippet = build_signal_excerpt(item)
        source_type = classify_source_type(
            url=url,
            title=str(item.get("title") or ""),
            snippet=snippet,
            source_type_hint=source_type_hint,
        )
        signal: SearchSignal = {
            "title": str(item.get("title") or "").strip(),
            "url": url,
            "snippet": snippet,
            "published_at": str(item.get("published_date") or "").strip(),
            "query": query,
            "query_family": query_family,
            "source_kind": source_kind,
            "source_type": source_type,
            "domain": extract_domain(url),
            "relevance_score": 0.0,
        }
        signal["relevance_score"] = score_signal(signal, company_profile)
        normalized.append(
            signal
        )
    return normalized


def extract_search_results(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]

    if isinstance(response, dict):
        results = response.get("results", [])
        return results if isinstance(results, list) else []

    if isinstance(response, str):
        if "No search results found" in response:
            return []
        parsed = parse_json_safely(response)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            results = parsed.get("results", [])
            return results if isinstance(results, list) else []
        logger.warning("[%s/search] unexpected string response=%s", AGENT_NAME, response)
        return []

    logger.warning("[%s/search] unsupported response type=%s", AGENT_NAME, type(response).__name__)
    return []


def attach_source_ids(signals: list[SearchSignal]) -> list[SearchSignal]:
    enriched: list[SearchSignal] = []
    for index, signal in enumerate(signals, start=1):
        item = dict(signal)
        item["source_id"] = f"S{index}"
        enriched.append(item)
    return enriched


def is_relevant_signal(
    signal: SearchSignal,
    company_profile: CompanyProfile,
) -> bool:
    haystack = build_signal_haystack(signal)
    aliases = build_company_aliases(company_profile)
    if not any(alias.lower() in haystack for alias in aliases):
        return False

    if contains_any_term(haystack, QUERY_FAMILY_ROLE_HINTS.get(signal["query_family"], ())):
        return True
    if contains_any_term(haystack, ROLE_SIGNAL_TERMS):
        return True
    if signal["source_type"] in {"official", "news"} and contains_any_term(
        haystack,
        OFFICIAL_PAGE_TERMS + NEWS_SIGNAL_TERMS,
    ):
        return True
    return signal["relevance_score"] >= 6


def extract_domain(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower()


def build_company_aliases(company_profile: CompanyProfile) -> list[str]:
    aliases = [
        alias
        for alias in [
            company_profile["company_name"].strip(),
            company_profile["product_name"].strip(),
        ]
        if alias
    ]
    compact_aliases = [alias.replace(" ", "") for alias in aliases if " " in alias]
    return dedupe_keep_order([*aliases, *compact_aliases])


def build_signal_excerpt(item: dict[str, Any], *, max_chars: int = 900) -> str:
    for field_name in ("raw_content", "content", "snippet"):
        value = clean_text(str(item.get(field_name) or ""))
        if value:
            return truncate_text(value, max_chars=max_chars)
    return ""


def classify_source_type(
    *,
    url: str,
    title: str,
    snippet: str,
    source_type_hint: str,
) -> str:
    domain = extract_domain(url)
    path = urlparse(url).path.lower()
    haystack = " ".join([title, snippet, url]).lower()

    if any(hint in domain for hint in PROFESSIONAL_DOMAIN_HINTS):
        return "professional"
    if any(hint in domain for hint in NEWS_DOMAIN_HINTS) or contains_any_term(
        haystack,
        NEWS_SIGNAL_TERMS,
    ):
        return "news"
    if any(hint in path for hint in OFFICIAL_PATH_HINTS) or contains_any_term(
        haystack,
        OFFICIAL_PAGE_TERMS,
    ):
        return "official"
    return source_type_hint


def score_signal(
    signal: SearchSignal,
    company_profile: CompanyProfile,
) -> float:
    haystack = build_signal_haystack(signal)
    aliases = build_company_aliases(company_profile)
    score = 0.0

    if any(alias.lower() in signal["title"].lower() for alias in aliases):
        score += 3.0
    elif any(alias.lower() in haystack for alias in aliases):
        score += 2.0

    if contains_any_term(haystack, ROLE_SIGNAL_TERMS):
        score += 3.0
    if contains_any_term(haystack, QUERY_FAMILY_ROLE_HINTS.get(signal["query_family"], ())):
        score += 2.0
    if contains_any_term(haystack, OFFICIAL_PAGE_TERMS):
        score += 1.5
    if contains_any_term(haystack, NEWS_SIGNAL_TERMS):
        score += 1.0
    if contains_any_term(haystack, LOW_VALUE_TERMS) and not contains_any_term(
        haystack,
        ROLE_SIGNAL_TERMS,
    ):
        score -= 2.0

    score += SOURCE_TYPE_PRIORITY.get(signal["source_type"], 0)

    if signal["url"]:
        score += 0.5
    if signal["published_at"]:
        score += 0.25

    return round(score, 2)


def build_signal_haystack(signal: SearchSignal) -> str:
    return " ".join(
        [
            signal["title"],
            signal["snippet"],
            signal["url"],
        ]
    ).lower()


def contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term.lower() in text for term in terms)


def select_family_signals(
    signals: list[SearchSignal],
    *,
    seen_keys: set[str],
    limit: int,
) -> list[SearchSignal]:
    if limit <= 0:
        return []

    selected: list[SearchSignal] = []
    ranked = sorted(
        signals,
        key=lambda signal: (
            signal["source_type"] in {"official", "news"},
            signal["relevance_score"],
            SOURCE_TYPE_PRIORITY.get(signal["source_type"], 0),
            len(signal["snippet"]),
        ),
        reverse=True,
    )
    for signal in ranked:
        dedupe_key = signal["url"] or f"{signal['domain']}::{signal['title']}"
        if not dedupe_key or dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        selected.append(signal)
        if len(selected) >= limit:
            break

    return selected


def extract_investigate_members(
    company_profile: CompanyProfile,
    signals: list[SearchSignal],
) -> InvestigateMembersExtractionResult:
    extractor = get_chat_model().with_structured_output(
        InvestigateMembersExtractionResult,
        method="json_schema",
    )
    return extractor.invoke(
        [
            ("system", get_system_prompt()),
            ("user", render_user_prompt(company_profile, signals)),
        ]
    )


def build_empty_extraction() -> InvestigateMembersExtractionResult:
    return InvestigateMembersExtractionResult(
        strengths=[],
        evidence_gaps=[
            "웹 검색 결과에서 회사명과 직접 연결되는 CEO 및 핵심팀 공개 근거를 확보하지 못했습니다.",
        ],
        assessment_summary=(
            "현재 공개된 웹 검색 결과만으로는 창업자 및 핵심팀 신뢰도를 판단할 팀 구성 근거가 부족합니다."
        ),
        evidence_quality="공개 출처가 부족해 리더십 검증 품질이 낮습니다.",
    )


def build_investigate_members_payload(
    *,
    search_queries: dict[str, str],
    signals: list[SearchSignal],
    extraction: InvestigateMembersExtractionResult,
) -> InvestigateMembersPayload:
    source_map = {
        signal["source_id"]: signal
        for signal in signals
        if signal.get("source_id")
    }
    ceo = normalize_member_profile(extraction.ceo, source_map)
    key_members = normalize_key_members(extraction.key_members, ceo, source_map)
    role_coverage = build_role_coverage(ceo, key_members)
    strengths = clean_text_list(extraction.strengths)
    if not strengths:
        strengths = derive_strengths(ceo, key_members, role_coverage)

    evidence_gaps = merge_text_lists(
        clean_text_list(extraction.evidence_gaps),
        derive_evidence_gaps(ceo, key_members, signals, role_coverage),
    )

    assessment_summary = clean_text(extraction.assessment_summary) or build_assessment_summary(
        ceo,
        key_members,
        role_coverage,
        signals,
    )
    evidence_quality = clean_text(extraction.evidence_quality) or build_evidence_quality(
        signals,
        ceo,
        key_members,
    )

    return {
        "ceo": ceo,
        "key_members": key_members,
        "role_coverage": role_coverage,
        "strengths": strengths,
        "evidence_gaps": evidence_gaps,
        "assessment_summary": assessment_summary,
        "evidence_quality": evidence_quality,
        "search_queries": list(search_queries.values()),
    }


def build_error_payload(
    search_queries: dict[str, str],
    error_message: str,
) -> InvestigateMembersPayload:
    return {
        "ceo": None,
        "key_members": [],
        "role_coverage": build_empty_role_coverage(),
        "strengths": [],
        "evidence_gaps": [
            "웹 검색 또는 structured extraction 단계가 실패해 리더십 근거를 수집하지 못했습니다.",
            f"실행 오류: {error_message}",
        ],
        "assessment_summary": "조사 실행 오류로 인해 CEO 및 핵심팀 근거를 정리하지 못했습니다.",
        "evidence_quality": "외부 검색/모델 호출 실패로 판단 가능한 근거가 비어 있습니다.",
        "search_queries": list(search_queries.values()),
    }


def normalize_member_profile(
    member: InvestigateMemberExtraction | None,
    source_map: dict[str, SearchSignal],
) -> InvestigateMemberProfile | None:
    if member is None:
        return None

    name = clean_text(member.name)
    current_role = clean_text(member.current_role)
    source_ids = [
        source_id
        for source_id in dedupe_keep_order(member.source_ids)
        if source_id in source_map
    ]
    if not name or not current_role or not source_ids:
        return None

    experience_tags = [
        tag
        for tag in dedupe_keep_order(list(member.experience_tags))
        if tag in ROLE_TAXONOMY
    ]
    evidence_summary = clean_text(member.evidence_summary) or "공개 자료 기반 역할 및 경력 근거를 확인했습니다."

    return {
        "name": name,
        "current_role": current_role,
        "is_founder": bool(member.is_founder),
        "experience_tags": experience_tags,
        "evidence_summary": evidence_summary,
        "source_ids": source_ids,
        "confidence": clamp_confidence(member.confidence),
    }


def normalize_key_members(
    members: list[InvestigateMemberExtraction],
    ceo: InvestigateMemberProfile | None,
    source_map: dict[str, SearchSignal],
) -> list[InvestigateMemberProfile]:
    normalized: list[InvestigateMemberProfile] = []
    seen: set[tuple[str, str]] = set()
    ceo_name = (ceo or {}).get("name", "").strip().lower()

    for member in members:
        item = normalize_member_profile(member, source_map)
        if item is None:
            continue
        if item["name"].strip().lower() == ceo_name:
            continue

        dedupe_key = (
            item["name"].strip().lower(),
            item["current_role"].strip().lower(),
        )
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized.append(item)
        if len(normalized) >= MAX_KEY_MEMBERS:
            break

    return normalized


def build_role_coverage(
    ceo: InvestigateMemberProfile | None,
    key_members: list[InvestigateMemberProfile],
) -> InvestigateMembersRoleCoverage:
    coverage = build_empty_role_coverage()
    members = [member for member in [ceo, *key_members] if member]
    for member in members:
        for tag in member["experience_tags"]:
            if tag in coverage:
                coverage[tag] = True
    return coverage


def build_empty_role_coverage() -> InvestigateMembersRoleCoverage:
    return {
        "robot_hw": False,
        "robot_sw_ai": False,
        "control_perception": False,
        "system_integration": False,
        "productization_deployment": False,
        "manufacturing_operations": False,
        "business_development": False,
    }


def determine_research_status(
    payload: InvestigateMembersPayload,
    signals: list[SearchSignal],
) -> str:
    ceo = payload["ceo"]
    key_members = payload["key_members"]
    if ceo is None or not key_members:
        return "failed"

    source_map = {
        signal["source_id"]: signal
        for signal in signals
        if signal.get("source_id")
    }
    referenced_urls = {
        source_map[source_id]["url"]
        for source_id in collect_referenced_source_ids(payload)
        if source_id in source_map and source_map[source_id]["url"]
    }
    if len(referenced_urls) < 2:
        return "failed"
    return "completed"


def collect_referenced_source_ids(payload: InvestigateMembersPayload) -> list[str]:
    source_ids: list[str] = []
    if payload["ceo"]:
        source_ids.extend(payload["ceo"]["source_ids"])
    for member in payload["key_members"]:
        source_ids.extend(member["source_ids"])
    return dedupe_keep_order(source_ids)


def derive_strengths(
    ceo: InvestigateMemberProfile | None,
    key_members: list[InvestigateMemberProfile],
    role_coverage: InvestigateMembersRoleCoverage,
) -> list[str]:
    strengths: list[str] = []
    if ceo:
        strengths.append(
            f"CEO/대표 `{ceo['name']}`의 역할과 경력 근거를 공개 자료로 확인했습니다."
        )
    if key_members:
        member_names = ", ".join(member["name"] for member in key_members[:3])
        strengths.append(
            f"CEO 외 핵심팀 {len(key_members)}명({member_names})에 대한 공개 근거가 확인됩니다."
        )

    covered_roles = format_covered_roles(role_coverage)
    if covered_roles:
        strengths.append(f"공개 근거상 확인된 역할 축은 {covered_roles}입니다.")
    return strengths


def derive_evidence_gaps(
    ceo: InvestigateMemberProfile | None,
    key_members: list[InvestigateMemberProfile],
    signals: list[SearchSignal],
    role_coverage: InvestigateMembersRoleCoverage,
) -> list[str]:
    gaps: list[str] = []
    if ceo is None:
        gaps.append("공개 자료에서 대표/CEO를 확정할 수 있는 근거가 부족합니다.")
    if not key_members:
        gaps.append("CEO 외 핵심팀 1인 이상을 특정할 공개 근거가 부족합니다.")

    unique_urls = {signal["url"] for signal in signals if signal["url"]}
    if len(unique_urls) < 2:
        gaps.append("서로 다른 URL 2건 이상에서 팀 근거를 확보하지 못했습니다.")

    uncovered_roles = [
        ROLE_LABELS[key]
        for key in ROLE_TAXONOMY
        if not role_coverage[key]
    ]
    if uncovered_roles:
        gaps.append(
            f"현재 공개 자료만으로는 {', '.join(uncovered_roles[:4])} 역할 축 근거가 약합니다."
        )
    return gaps


def build_assessment_summary(
    ceo: InvestigateMemberProfile | None,
    key_members: list[InvestigateMemberProfile],
    role_coverage: InvestigateMembersRoleCoverage,
    signals: list[SearchSignal],
) -> str:
    if ceo is None and not key_members:
        return "공개 자료 기준으로 CEO 및 핵심팀 신뢰도를 판단할 인물 근거가 거의 확인되지 않습니다."

    covered_roles = format_covered_roles(role_coverage) or "역할 축 확인이 제한적"
    unique_urls = len({signal["url"] for signal in signals if signal["url"]})
    return (
        f"CEO 1인과 핵심팀 {len(key_members)}명에 대한 공개 근거를 바탕으로 "
        f"{covered_roles} 축을 확인했습니다. 현재 참조한 서로 다른 URL은 {unique_urls}건입니다."
    )


def build_evidence_quality(
    signals: list[SearchSignal],
    ceo: InvestigateMemberProfile | None,
    key_members: list[InvestigateMemberProfile],
) -> str:
    source_count = len(signals)
    unique_urls = len({signal["url"] for signal in signals if signal["url"]})
    if ceo is None and not key_members:
        return f"관련 공개 출처 {source_count}건을 검토했지만 팀 식별 근거가 충분하지 않습니다."
    return (
        f"관련 공개 출처 {source_count}건 중 서로 다른 URL {unique_urls}건을 활용해 "
        "리더십 및 핵심팀 근거를 교차 확인했습니다."
    )


def build_summary(
    company_name: str,
    status: str,
    payload: InvestigateMembersPayload,
    signals: list[SearchSignal],
) -> str:
    unique_urls = len({signal["url"] for signal in signals if signal["url"]})
    if status == "completed":
        return (
            f"{company_name}의 CEO와 핵심팀 {len(payload['key_members'])}명에 대한 공개 근거를 확보했고, "
            f"C.1 판단에 필요한 최소 기준(대표 확인, 핵심팀 확인, URL {unique_urls}건)을 충족했습니다."
        )
    return (
        f"{company_name}의 CEO 및 핵심팀 조사를 수행했지만, "
        f"C.1 최소 기준을 충족할 공개 근거가 부족해 failed로 정리했습니다. "
        f"{payload['assessment_summary']}"
    )


def build_findings(
    status: str,
    payload: InvestigateMembersPayload,
    signals: list[SearchSignal],
) -> list[str]:
    findings: list[str] = []
    ceo = payload["ceo"]
    if ceo is not None:
        findings.append(
            f"CEO/대표는 `{ceo['name']}`(`{ceo['current_role']}`)로 식별되며, 근거 source는 {', '.join(ceo['source_ids'])}입니다."
        )
    else:
        findings.append("CEO/대표를 확정할 공개 근거를 찾지 못했습니다.")

    if payload["key_members"]:
        member_summaries = ", ".join(
            f"{member['name']}({member['current_role']})"
            for member in payload["key_members"][:3]
        )
        findings.append(
            f"CEO 외 핵심팀 {len(payload['key_members'])}명을 확인했습니다: {member_summaries}."
        )
    else:
        findings.append("CEO 외 핵심팀 1인 이상을 특정할 공개 근거가 없습니다.")

    covered_roles = format_covered_roles(payload["role_coverage"])
    if covered_roles:
        findings.append(f"역할 커버리지는 {covered_roles} 축에서 공개 근거가 확인됩니다.")
    else:
        findings.append("로봇 HW/SW, 통합, 운영, 사업개발 축의 공개 근거가 충분하지 않습니다.")

    findings.append(payload["evidence_quality"])

    for gap in payload["evidence_gaps"]:
        findings.append(gap)
        if len(findings) >= 6:
            break

    if status == "completed" and len(findings) < 5:
        unique_urls = len({signal["url"] for signal in signals if signal["url"]})
        findings.append(f"Strict completion 기준을 만족한 서로 다른 공개 URL은 총 {unique_urls}건입니다.")

    return findings[:6]


def format_covered_roles(role_coverage: InvestigateMembersRoleCoverage) -> str:
    covered = [
        ROLE_LABELS[key]
        for key in ROLE_TAXONOMY
        if role_coverage[key]
    ]
    return ", ".join(covered)


def dedupe_query_plans(plans: list[SearchQueryPlan]) -> list[SearchQueryPlan]:
    deduped: list[SearchQueryPlan] = []
    seen_queries: set[str] = set()
    for plan in plans:
        query = clean_text(plan["query"])
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        deduped.append(
            {
                "query": query,
                "source_type": plan["source_type"],
                "topic": plan["topic"],
            }
        )
    return deduped


def parse_json_safely(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    trimmed = value[: max_chars - 3].rsplit(" ", 1)[0]
    return f"{trimmed or value[: max_chars - 3]}..."


def clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def clean_text_list(values: list[str]) -> list[str]:
    return [item for item in (clean_text(value) for value in values) if item]


def merge_text_lists(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def dedupe_keep_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def clamp_confidence(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 2)
