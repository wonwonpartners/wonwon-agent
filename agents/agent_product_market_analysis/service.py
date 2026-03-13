from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from langchain.agents import create_agent

from agents.agent_product_market_analysis.common import get_chat_model
from agents.agent_product_market_analysis.prompts import (
    get_research_system_prompt,
    get_writer_system_prompt,
    render_research_user_prompt,
)
from agents.agent_product_market_analysis.result import (
    AnalysisFieldResult,
    ProductMarketAnalysisResult,
)
from agents.agent_product_market_analysis.tools import (
    PRODUCT_MARKET_ANALYSIS_TOOLS,
)
from agents.workflow_common import ResearchAgentState, get_company_id, get_company_name


AGENT_NAME = "agent_product_market_analysis"
logger = logging.getLogger(__name__)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def run_product_market_analysis(
    selected_company: dict[str, Any] | None,
    previous_state: ResearchAgentState | None = None,
) -> ResearchAgentState:
    attempt_count = int((previous_state or {}).get("attempt_count", 0)) + 1
    company_id = get_company_id(selected_company)
    company_name = get_company_name(selected_company)

    if company_id is None:
        payload = {
            "agent_name": AGENT_NAME,
            "status": "skipped",
            "attempt_count": attempt_count,
            "input_company_id": None,
            "summary": "선정된 회사가 없어 제품/시장 분석을 진행하지 못했습니다.",
            "findings": [
                "선행 단계에서 `selected_company`가 비어 있어 조사를 건너뛰었습니다.",
            ],
            "sources": [],
            "structured_output": None,
        }
        logger.info(
            "[%s/final] status=%s attempt=%s company_id=-",
            AGENT_NAME,
            payload["status"],
            payload["attempt_count"],
        )
        return payload

    company_profile = build_company_profile(selected_company)
    try:
        research_notes, source_entries = run_product_market_research(company_profile)
        structured_result = write_product_market_result(
            company_profile,
            research_notes,
            source_entries,
        )
        structured_result = normalize_result_references(
            structured_result,
            source_entries,
        )
        payload = {
            "agent_name": AGENT_NAME,
            "status": "completed",
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": build_summary(company_name, structured_result),
            "findings": build_findings(structured_result),
            "sources": source_entries,
            "structured_output": build_structured_output_payload(
                structured_result,
            ),
        }
        logger.info(
            "[%s/final] status=%s attempt=%s company=%s(%s)",
            AGENT_NAME,
            payload["status"],
            payload["attempt_count"],
            company_name,
            company_id,
        )
        return payload
    except Exception as exc:
        logger.exception(
            "[%s/error] company=%s(%s) message=%s",
            AGENT_NAME,
            company_name,
            company_id,
            exc,
        )
        payload = {
            "agent_name": AGENT_NAME,
            "status": "failed",
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": (
                f"{company_name} ({company_id})의 제품/시장 분석을 실행하는 중 오류가 발생했습니다."
            ),
            "findings": [
                "제품/시장 조사 또는 구조화 단계에서 오류가 발생했습니다.",
                f"오류 메시지: {str(exc)}",
                "환경변수와 외부 도구 상태를 확인한 뒤 재시도해야 합니다.",
            ],
            "sources": [],
            "structured_output": {
                "target_kpi_logic": {
                    "text": "",
                    "references": [],
                    "evidence_gap": "실행 오류로 근거 연결을 검증하지 못했습니다.",
                },
                "technical_moat": {
                    "text": "",
                    "references": [],
                    "evidence_gap": "실행 오류로 근거 연결을 검증하지 못했습니다.",
                },
                "data_loop_structure": {
                    "text": "",
                    "references": [],
                    "evidence_gap": "실행 오류로 근거 연결을 검증하지 못했습니다.",
                },
                "product_summary": {
                    "text": f"실행 오류로 제품/시장 분석을 완료하지 못했습니다. 오류: {str(exc)}",
                    "references": [],
                    "evidence_gap": "실행 오류로 근거 연결을 검증하지 못했습니다.",
                },
            },
        }
        logger.info(
            "[%s/final] status=%s attempt=%s company=%s(%s)",
            AGENT_NAME,
            payload["status"],
            payload["attempt_count"],
            company_name,
            company_id,
        )
        return payload


def build_company_profile(selected_company: dict[str, Any] | None) -> dict[str, str]:
    return {
        "company_id": sanitize_text(get_company_id(selected_company) or ""),
        "company_name": sanitize_text(get_company_name(selected_company)),
        "product_name": sanitize_text(
            str((selected_company or {}).get("product_name") or "").strip()
        ),
        "description": sanitize_text(
            str((selected_company or {}).get("description") or "").strip()
        ),
    }


def run_product_market_research(
    company_profile: dict[str, str],
) -> tuple[str, list[dict[str, str]]]:
    agent = create_agent(
        model=get_chat_model(),
        tools=PRODUCT_MARKET_ANALYSIS_TOOLS,
        system_prompt=get_research_system_prompt(),
    )
    user_prompt = render_research_user_prompt(
        json.dumps(company_profile, ensure_ascii=False, indent=2)
    )
    result = agent.invoke({"messages": [("user", user_prompt)]})
    messages = result.get("messages", [])
    research_notes = collect_research_notes(messages)
    return research_notes, collect_sources(messages)


def write_product_market_result(
    company_profile: dict[str, str],
    research_notes: str,
    source_entries: list[dict[str, str]],
) -> ProductMarketAnalysisResult:
    writer = get_chat_model().with_structured_output(
        ProductMarketAnalysisResult,
        method="json_schema",
    )
    return writer.invoke(
        [
            ("system", get_writer_system_prompt()),
            (
                "user",
                "\n\n".join(
                    [
                        f"company_profile:\n{json.dumps(company_profile, ensure_ascii=False, indent=2)}",
                        f"research_notes:\n{research_notes}",
                        "source_balance_guidance:",
                        render_source_balance_guidance(source_entries),
                        "available_sources:",
                        render_available_sources(source_entries),
                    ]
                ),
            ),
        ]
    )


def render_available_sources(source_entries: list[dict[str, str]]) -> str:
    eligible_sources = filter_sources_with_url(source_entries)
    if not eligible_sources:
        return "- 없음"

    lines: list[str] = []
    ordered_sources = order_sources_for_writer(eligible_sources)
    for source in ordered_sources:
        lines.append(
            " | ".join(
                [
                    f"class={classify_source_for_writer(source)}",
                    f"citation={format_source_for_writer(source)}",
                    f"excerpt={sanitize_text(source.get('excerpt', '').strip())[:240] or '없음'}",
                ]
            )
        )
    return "\n".join(lines)


def render_source_balance_guidance(source_entries: list[dict[str, str]]) -> str:
    eligible_sources = filter_sources_with_url(source_entries)
    if not eligible_sources:
        return "- URL이 포함된 사용 가능한 출처가 없습니다."

    grouped = group_sources_for_writer(eligible_sources)
    lines = [
        (
            "external/domain sources: "
            f"{len(grouped['external'])}건"
        ),
        (
            "company-specific sources: "
            f"{len(grouped['company'])}건"
        ),
        (
            "other extracted/misc sources: "
            f"{len(grouped['other'])}건"
        ),
    ]
    if grouped["external"]:
        lines.append(
            "가능하면 각 판단 필드에서 external/domain 출처를 최소 1개 이상 사용하십시오."
        )
    else:
        lines.append(
            "external/domain 출처가 없으므로 외부 검증 부족을 명시하십시오."
        )
    return "\n".join(f"- {line}" for line in lines)


def order_sources_for_writer(
    source_entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    grouped = group_sources_for_writer(source_entries)
    return [
        *grouped["external"],
        *grouped["company"],
        *grouped["other"],
    ]


def group_sources_for_writer(
    source_entries: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {
        "external": [],
        "company": [],
        "other": [],
    }
    for source in source_entries:
        grouped[classify_source_for_writer(source)].append(source)
    return grouped


def filter_sources_with_url(
    source_entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        source
        for source in source_entries
        if is_reportable_source(source)
    ]


def is_reportable_source(source: dict[str, str]) -> bool:
    url = sanitize_text(source.get("url", "").strip())
    title = sanitize_text(source.get("title", "").strip())
    author = sanitize_text(source.get("author", "").strip())
    publisher = sanitize_text(source.get("publisher", "").strip())
    organization = sanitize_text(source.get("organization", "").strip())
    journal = sanitize_text(source.get("journal", "").strip())
    published_at = sanitize_text(source.get("published_at", "").strip())

    has_descriptor = bool(
        title or author or publisher or organization or journal or published_at
    )
    return bool(url and has_descriptor)


def classify_source_for_writer(source: dict[str, str]) -> str:
    tool_name = source.get("tool_name", "")
    if tool_name in {"domain_rag_search_tool", "web_benchmark_search_tool"}:
        return "external"
    if tool_name in {"company_rag_search_tool", "company_web_search_tool"}:
        return "company"
    return "other"


def format_source_for_writer(source: dict[str, str]) -> str:
    parts: list[str] = []

    title = sanitize_text(source.get("title", "").strip())
    author = sanitize_text(source.get("author", "").strip())
    publisher = (
        sanitize_text(source.get("publisher", "").strip())
        or sanitize_text(source.get("organization", "").strip())
        or sanitize_text(source.get("journal", "").strip())
    )
    published_at = sanitize_text(source.get("published_at", "").strip())
    url = sanitize_text(source.get("url", "").strip())
    source_path = sanitize_text(source.get("source_path", "").strip())
    source_type = sanitize_text(source.get("source_type", "").strip())

    if author:
        parts.append(author)
    if published_at:
        parts.append(f"({published_at})")
    if title:
        parts.append(title)
    if publisher:
        parts.append(publisher)
    if url:
        parts.append(url)
    elif source_path:
        parts.append(source_path)
    elif source_type:
        parts.append(f"type={source_type}")

    return ", ".join(part for part in parts if part) or "출처 정보 없음"


def collect_research_notes(messages: list[Any]) -> str:
    notes: list[str] = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = stringify_message_content(getattr(message, "content", ""))
        if content:
            notes.append(f"[{role}]\n{content}")
    return "\n\n".join(notes)


def collect_sources(messages: list[Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for index, message in enumerate(messages, start=1):
        role = str(getattr(message, "type", message.__class__.__name__))
        if role not in {"tool", "ToolMessage"}:
            continue
        tool_name = str(getattr(message, "name", "") or "tool")
        content = stringify_message_content(getattr(message, "content", ""))
        if not content.strip():
            continue
        parsed_items = parse_sources_from_tool_output(
            tool_name=tool_name,
            content=content,
            message_index=index,
        )
        for item in parsed_items:
            item["source_id"] = build_source_id(item)
            dedupe_key = build_source_dedupe_key(item)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            sources.append(item)
    return sources


def parse_sources_from_tool_output(
    *,
    tool_name: str,
    content: str,
    message_index: int,
) -> list[dict[str, str]]:
    if tool_name in {"company_rag_search_tool", "domain_rag_search_tool"}:
        return parse_rag_sources(
            content,
            tool_name=tool_name,
            message_index=message_index,
        )
    if tool_name in {"company_web_search_tool", "web_benchmark_search_tool"}:
        return parse_web_search_sources(
            content,
            tool_name=tool_name,
            message_index=message_index,
        )
    if tool_name == "web_page_extract_tool":
        return [
            parse_web_page_source(
                content,
                tool_name=tool_name,
                message_index=message_index,
            )
        ]
    return [
        {
            "source_type": "tool_output",
            "tool_name": tool_name,
            "message_index": str(message_index),
            "excerpt": content[:500],
        }
    ]


def parse_rag_sources(
    content: str,
    *,
    tool_name: str,
    message_index: int,
) -> list[dict[str, str]]:
    sections = [
        section.strip()
        for section in content.split("\n\n[document]")
        if section.strip()
    ]
    parsed: list[dict[str, str]] = []

    for raw_section in sections:
        section = raw_section
        if not section.startswith("[document]"):
            section = f"[document]\n{section}"
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        metadata: dict[str, str] = {
            "source_type": "rag_document",
            "tool_name": tool_name,
            "message_index": str(message_index),
        }
        excerpt_lines: list[str] = []
        in_excerpt = False

        for line in lines[1:]:
            if not in_excerpt and ":" in line:
                key, value = line.split(":", 1)
                normalized_key = key.strip().lower()
                normalized_value = value.strip()
                if normalized_key == "source":
                    metadata["source_path"] = normalized_value
                elif normalized_key in {
                    "title",
                    "author",
                    "organization",
                    "publisher",
                    "journal",
                    "published_at",
                    "url",
                }:
                    metadata[normalized_key] = normalized_value
                else:
                    in_excerpt = True
                    excerpt_lines.append(line)
            else:
                in_excerpt = True
                excerpt_lines.append(line)

        if excerpt_lines:
            metadata["excerpt"] = "\n".join(excerpt_lines)[:500]
        parsed.append(metadata)

    return parsed or [
        {
            "source_type": "rag_document",
            "tool_name": tool_name,
            "message_index": str(message_index),
            "excerpt": content[:500],
        }
    ]


def parse_web_search_sources(
    content: str,
    *,
    tool_name: str,
    message_index: int,
) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    sections = [section.strip() for section in content.split("\n\n") if section.strip()]

    for section in sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            continue
        item: dict[str, str] = {
            "source_type": "web_search",
            "tool_name": tool_name,
            "message_index": str(message_index),
        }

        title_line = lines[0]
        if "] " in title_line:
            item["title"] = title_line.split("] ", 1)[1].strip()
        else:
            item["title"] = title_line

        for line in lines[1:]:
            if line.startswith("URL:"):
                item["url"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                item["publisher"] = line.split(":", 1)[1].strip()
            elif line.startswith("Published:"):
                item["published_at"] = line.split(":", 1)[1].strip()
            elif line.startswith("Snippet:"):
                item["excerpt"] = line.split(":", 1)[1].strip()[:500]

        parsed.append(item)

    return parsed or [
        {
            "source_type": "web_search",
            "tool_name": tool_name,
            "message_index": str(message_index),
            "excerpt": content[:500],
        }
    ]


def parse_web_page_source(
    content: str,
    *,
    tool_name: str,
    message_index: int,
) -> dict[str, str]:
    item: dict[str, str] = {
        "source_type": "web_page",
        "tool_name": tool_name,
        "message_index": str(message_index),
    }
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("URL:"):
            item["url"] = line.split(":", 1)[1].strip()
        elif line.startswith("Title:"):
            item["title"] = line.split(":", 1)[1].strip()
            break
    item["excerpt"] = content[:500]
    return item


def build_source_dedupe_key(source: dict[str, str]) -> str:
    return (
        source.get("source_id")
        or source.get("url")
        or source.get("source_path")
        or " :: ".join(
            [
                source.get("title", ""),
                source.get("published_at", ""),
                source.get("tool_name", ""),
            ]
        )
    )


def stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return sanitize_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(sanitize_text(str(text)))
            else:
                parts.append(sanitize_text(str(item)))
        return sanitize_text("\n".join(parts))
    return sanitize_text(str(content))


def sanitize_text(value: str) -> str:
    cleaned = CONTROL_CHAR_RE.sub("", value)
    cleaned = cleaned.encode("utf-8", "replace").decode("utf-8")
    return cleaned.strip()


def build_summary(
    company_name: str,
    result: ProductMarketAnalysisResult,
) -> str:
    return f"{company_name}의 제품/시장 분석을 완료했습니다. {result.product_summary.text}"


def build_findings(
    result: ProductMarketAnalysisResult,
) -> list[str]:
    return [
        format_finding_with_references(
            "KPI/ROI 논리",
            result.target_kpi_logic,
        ),
        format_finding_with_references(
            "기술 해자",
            result.technical_moat,
        ),
        format_finding_with_references(
            "데이터 루프/폴백",
            result.data_loop_structure,
        ),
        format_finding_with_references(
            "종합 요약",
            result.product_summary,
        ),
    ]


def format_finding_with_references(
    label: str,
    field: AnalysisFieldResult,
) -> str:
    parts = [f"{label}: {field.text}"]
    if field.references:
        parts.append(f"출처: {', '.join(field.references)}")
    if field.evidence_gap:
        parts.append(f"근거한계: {field.evidence_gap}")

    return " | ".join(parts)


def build_source_id(source: dict[str, str]) -> str:
    base = " :: ".join(
        [
            source.get("url", "").strip(),
            source.get("source_path", "").strip(),
            source.get("title", "").strip(),
            source.get("published_at", "").strip(),
            source.get("tool_name", "").strip(),
        ]
    )
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"src_{digest}"


def normalize_result_references(
    result: ProductMarketAnalysisResult,
    source_entries: list[dict[str, str]],
) -> ProductMarketAnalysisResult:
    eligible_sources = filter_sources_with_url(source_entries)
    valid_references = {
        format_source_for_writer(source)
        for source in eligible_sources
        if format_source_for_writer(source).strip()
    }
    result.target_kpi_logic = normalize_analysis_field(result.target_kpi_logic, valid_references)
    result.technical_moat = normalize_analysis_field(result.technical_moat, valid_references)
    result.data_loop_structure = normalize_analysis_field(result.data_loop_structure, valid_references)
    result.product_summary = normalize_analysis_field(result.product_summary, valid_references)
    return result


def normalize_analysis_field(
    field: AnalysisFieldResult,
    valid_references: set[str],
) -> AnalysisFieldResult:
    normalized_references: list[str] = []
    seen_references: set[str] = set()

    for reference in field.references:
        normalized_reference = str(reference).strip()
        if (
            not normalized_reference
            or normalized_reference in seen_references
            or normalized_reference not in valid_references
        ):
            continue
        seen_references.add(normalized_reference)
        normalized_references.append(normalized_reference)

    field.references = normalized_references
    if field.text.strip() and not field.references and not field.evidence_gap.strip():
        field.evidence_gap = "근거 참고문헌 연결이 부족해 검증 강도가 제한됩니다."
    return field


def build_structured_output_payload(
    result: ProductMarketAnalysisResult,
) -> dict[str, Any]:
    return result.model_dump()
