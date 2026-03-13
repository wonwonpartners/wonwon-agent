from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, cast
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from agents.agent_report.common import (
    REPORTS_ROOT,
    get_chat_model,
    resolve_report_font_path,
)
from agents.agent_report.prompts import get_system_prompt, render_user_prompt
from agents.workflow_common import (
    EvalState,
    ReportState,
    ResearchAgentState,
    ReviewAggregateState,
    get_company_id,
    get_company_name,
)

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 850
REPORT_FONT_NAME = "WonwonReportFont"
REFERENCE_CATEGORY_ORDER = (
    "기관 보고서",
    "학술 논문",
    "웹페이지",
)
REPORT_SECTION_FIELDS = (
    (
        "1. 기업 개요",
        (
            ("회사 소개", "company_intro"),
            ("해결 문제와 솔루션", "problem_solution"),
            ("주요 제품/서비스", "products_services"),
        ),
    ),
    (
        "2. 시장 및 사업성 분석",
        (
            ("시장 규모와 성장성", "market_size_growth"),
            ("고객 수요", "customer_demand"),
            ("경쟁 환경", "competitive_landscape"),
        ),
    ),
    (
        "3. 제품·기술 경쟁력 분석",
        (
            ("제품 완성도", "product_maturity"),
            ("기술 차별성", "technical_differentiation"),
            ("AI/데이터 활용 강점", "ai_data_advantage"),
        ),
    ),
    (
        "4. 팀 역량 및 실행 현황",
        (
            ("창업자 및 핵심팀", "founders_team"),
            ("사업화 진행 상황", "commercialization_progress"),
            ("고객/파트너십/실적", "customers_partnerships_performance"),
        ),
    ),
    (
        "5. 주요 리스크 및 한계",
        (
            ("시장 리스크", "market_risk"),
            ("기술 리스크", "technical_risk"),
            ("규제/운영 리스크", "regulatory_operational_risk"),
        ),
    ),
    (
        "6. 종합 투자 의견 및 결론",
        (
            ("핵심 투자 포인트", "investment_points"),
            ("종합 평가", "overall_evaluation"),
            ("최종 투자 판단", "final_investment_judgment"),
        ),
    ),
)


class ReportFieldOutput(BaseModel):
    text: str = Field(default="")
    source_ids: list[str] = Field(default_factory=list)


class ReportDraftOutput(BaseModel):
    executive_summary: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    company_intro: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    problem_solution: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    products_services: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    market_size_growth: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    customer_demand: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    competitive_landscape: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    product_maturity: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    technical_differentiation: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    ai_data_advantage: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    founders_team: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    commercialization_progress: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    customers_partnerships_performance: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    market_risk: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    technical_risk: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    regulatory_operational_risk: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    investment_points: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    overall_evaluation: ReportFieldOutput = Field(default_factory=ReportFieldOutput)
    final_investment_judgment: ReportFieldOutput = Field(default_factory=ReportFieldOutput)


def build_report_state(
    *,
    selected_company: dict[str, Any] | None,
    force_report_generation: bool,
    company_search_summary: str,
    selected_company_reason: str,
    investigate_members_state: ResearchAgentState | None,
    agent_product_market_analysis_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
    agent_risk_search_state: ResearchAgentState | None,
    review_state: ReviewAggregateState | None,
    eval_state: EvalState | None,
) -> ReportState:
    company_id = get_company_id(selected_company)
    if (
        company_id is None
        or not eval_state
        or (
            not eval_state.get("ready_for_report")
            and not force_report_generation
        )
    ):
        payload = {
            "status": "skipped",
            "report_path": "",
            "pdf_path": "",
            "markdown": "",
        }
        logger.info(
            "[report/final] status=%s company_id=%s ready_for_report=%s forced=%s",
            payload["status"],
            company_id or "-",
            bool(eval_state and eval_state.get("ready_for_report")),
            force_report_generation,
        )
        return payload

    report_path = build_report_path(company_id)
    pdf_path = build_report_pdf_path(company_id)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    source_catalog = build_source_catalog(
        investigate_members_state=investigate_members_state,
        agent_product_market_analysis_state=agent_product_market_analysis_state,
        traction_state=traction_state,
        agent_risk_search_state=agent_risk_search_state,
    )
    report_input = build_report_input(
        selected_company=selected_company,
        force_report_generation=force_report_generation,
        company_search_summary=company_search_summary,
        selected_company_reason=selected_company_reason,
        investigate_members_state=investigate_members_state,
        agent_product_market_analysis_state=agent_product_market_analysis_state,
        traction_state=traction_state,
        agent_risk_search_state=agent_risk_search_state,
        review_state=review_state,
        eval_state=eval_state,
        source_catalog=source_catalog,
    )

    try:
        draft = run_report_draft(report_input, source_catalog)
    except Exception as exc:
        logger.exception("[report/error] llm_draft_failed message=%s", exc)
        draft = build_fallback_report_draft(report_input, source_catalog)

    fallback_draft = build_fallback_report_draft(report_input, source_catalog)
    draft = enrich_report_draft(
        draft=draft,
        fallback_draft=fallback_draft,
        source_catalog=source_catalog,
    )

    used_source_ids = collect_used_source_ids(draft, source_catalog)
    if not used_source_ids and source_catalog:
        logger.warning("[report/fallback] draft produced no source_ids; switching to fallback draft")
        draft = fallback_draft
        used_source_ids = collect_used_source_ids(draft, source_catalog)

    reference_sections = build_reference_sections(source_catalog, used_source_ids)
    company_name = get_company_name(selected_company)

    markdown = render_report_markdown(
        draft=draft,
        company_name=company_name,
        company_id=company_id,
        generated_at=generated_at,
        force_report_generation=force_report_generation,
        reference_sections=reference_sections,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    pdf_output_path = ""
    status = "completed"
    try:
        render_report_pdf(
            pdf_path=pdf_path,
            company_name=company_name,
            company_id=company_id,
            generated_at=generated_at,
            force_report_generation=force_report_generation,
            draft=draft,
            reference_sections=reference_sections,
        )
        pdf_output_path = str(pdf_path)
    except Exception as exc:
        logger.exception("[report/error] pdf_render_failed company_id=%s message=%s", company_id, exc)
        status = "failed"

    payload = {
        "status": status,
        "report_path": str(report_path),
        "pdf_path": pdf_output_path,
        "markdown": markdown,
    }
    logger.info(
        "[report/final] status=%s company_id=%s forced=%s markdown=%s pdf=%s",
        payload["status"],
        company_id,
        force_report_generation,
        payload["report_path"],
        payload["pdf_path"] or "-",
    )
    return payload


def build_report_path(company_id: str) -> Path:
    return (REPORTS_ROOT / f"{company_id}.md").resolve()


def build_report_pdf_path(company_id: str) -> Path:
    return (REPORTS_ROOT / f"{company_id}.pdf").resolve()


def run_report_draft(
    report_input: dict[str, Any],
    source_catalog: list[dict[str, Any]],
) -> ReportDraftOutput:
    writer = get_chat_model().with_structured_output(
        ReportDraftOutput,
        method="json_schema",
    )
    result = writer.invoke(
        [
            ("system", get_system_prompt()),
            (
                "user",
                render_user_prompt(
                    json.dumps(report_input, ensure_ascii=False, indent=2),
                ),
            ),
        ]
    )
    return normalize_report_draft(result, source_catalog)


def normalize_report_draft(
    result: ReportDraftOutput,
    source_catalog: list[dict[str, Any]],
) -> ReportDraftOutput:
    valid_source_ids = {str(item["id"]) for item in source_catalog}
    payload = result.model_dump()
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        field_payload = cast(dict[str, Any], value or {})
        if key == "executive_summary":
            text = trim_summary_text(field_payload.get("text", ""))
        else:
            text = normalize_text(field_payload.get("text", ""))
        source_ids = [
            source_id
            for source_id in dedupe_keep_order(field_payload.get("source_ids", []) or [])
            if source_id in valid_source_ids
        ]
        normalized[key] = {
            "text": text,
            "source_ids": source_ids,
        }
    return ReportDraftOutput.model_validate(normalized)


def enrich_report_draft(
    *,
    draft: ReportDraftOutput,
    fallback_draft: ReportDraftOutput,
    source_catalog: list[dict[str, Any]],
) -> ReportDraftOutput:
    valid_source_ids = {str(item["id"]) for item in source_catalog}
    normalized: dict[str, Any] = {}
    for field_name in ReportDraftOutput.model_fields:
        primary = cast(ReportFieldOutput, getattr(draft, field_name))
        fallback = cast(ReportFieldOutput, getattr(fallback_draft, field_name))
        if field_name == "executive_summary":
            text = normalize_summary_text(primary.text)
            fallback_text = normalize_summary_text(fallback.text)
        else:
            text = normalize_text(primary.text)
            fallback_text = normalize_text(fallback.text)

        if is_thin_report_text(text, field_name=field_name):
            text = merge_report_texts(
                primary_text=text,
                fallback_text=fallback_text,
                field_name=field_name,
            )

        source_ids = [
            source_id
            for source_id in dedupe_keep_order(
                list(primary.source_ids) + list(fallback.source_ids)
            )
            if source_id in valid_source_ids
        ]
        if field_name == "executive_summary":
            text = trim_summary_text(text)
        normalized[field_name] = {
            "text": text,
            "source_ids": source_ids,
        }
    return ReportDraftOutput.model_validate(normalized)


def build_fallback_report_draft(
    report_input: dict[str, Any],
    source_catalog: list[dict[str, Any]],
) -> ReportDraftOutput:
    selected_company = cast(dict[str, Any], report_input.get("selected_company") or {})
    company_name = str(selected_company.get("company_name") or "대상 기업")
    product_name = str(selected_company.get("product_name") or "").strip()
    description = str(selected_company.get("description") or "").strip()
    eval_state = cast(dict[str, Any], report_input.get("eval") or {})
    review_state = cast(dict[str, Any], report_input.get("review") or {})
    agents = cast(dict[str, Any], report_input.get("agents") or {})
    investigate = cast(dict[str, Any], agents.get("investigate_members") or {})
    product_market = cast(dict[str, Any], agents.get("agent_product_market_analysis") or {})
    traction = cast(dict[str, Any], agents.get("traction") or {})
    risk = cast(dict[str, Any], agents.get("agent_risk_search") or {})

    decision = str(eval_state.get("final_decision") or "watch")
    investigate_source_ids = choose_source_ids_from_investigate_payload(
        source_catalog,
        investigate,
        limit=2,
    )
    product_summary_source_ids = choose_source_ids_from_analysis_references(
        source_catalog,
        product_market,
        "product_summary",
        limit=2,
    )
    target_kpi_source_ids = choose_source_ids_from_analysis_references(
        source_catalog,
        product_market,
        "target_kpi_logic",
        limit=2,
    )
    technical_moat_source_ids = choose_source_ids_from_analysis_references(
        source_catalog,
        product_market,
        "technical_moat",
        limit=2,
    )
    data_loop_source_ids = choose_source_ids_from_analysis_references(
        source_catalog,
        product_market,
        "data_loop_structure",
        limit=2,
    )
    traction_source_ids = choose_source_ids(
        source_catalog,
        agent_names=("traction",),
        limit=3,
    )
    risk_source_ids = choose_source_ids(
        source_catalog,
        agent_names=("agent_risk_search",),
        limit=2,
    )
    summary_source_ids = merge_source_id_lists(
        investigate_source_ids,
        product_summary_source_ids,
        target_kpi_source_ids,
        technical_moat_source_ids,
        traction_source_ids,
        risk_source_ids,
        limit=4,
    )
    intro_text = build_company_intro_text(
        company_name=company_name,
        description=description,
        investigate=investigate,
        product_market=product_market,
    )
    problem_text = build_problem_solution_text(
        company_name=company_name,
        product_market=product_market,
    )
    product_text = build_products_services_text(
        company_name=company_name,
        product_name=product_name,
        product_market=product_market,
        traction=traction,
    )
    summary_text = trim_summary_text(
        build_executive_summary_text(
            company_name=company_name,
            eval_state=eval_state,
            investigate=investigate,
            product_market=product_market,
            traction=traction,
            risk=risk,
            review_state=review_state,
        )
    )

    base_output = {
        "executive_summary": {
            "text": summary_text,
            "source_ids": summary_source_ids,
        },
        "company_intro": {
            "text": intro_text,
            "source_ids": merge_source_id_lists(
                investigate_source_ids,
                product_summary_source_ids,
                limit=2,
            ),
        },
        "problem_solution": {
            "text": problem_text,
            "source_ids": target_kpi_source_ids,
        },
        "products_services": {
            "text": product_text,
            "source_ids": product_summary_source_ids,
        },
        "market_size_growth": {
            "text": build_market_size_growth_text(
                company_name=company_name,
                product_market=product_market,
                traction=traction,
            ),
            "source_ids": product_summary_source_ids,
        },
        "customer_demand": {
            "text": build_customer_demand_text(
                company_name=company_name,
                product_market=product_market,
                traction=traction,
            ),
            "source_ids": merge_source_id_lists(
                target_kpi_source_ids,
                traction_source_ids,
                limit=2,
            ),
        },
        "competitive_landscape": {
            "text": build_competitive_landscape_text(
                company_name=company_name,
                product_market=product_market,
                risk=risk,
            ),
            "source_ids": merge_source_id_lists(
                technical_moat_source_ids,
                risk_source_ids,
                limit=2,
            ),
        },
        "product_maturity": {
            "text": build_product_maturity_text(
                company_name=company_name,
                product_market=product_market,
                traction=traction,
            ),
            "source_ids": merge_source_id_lists(
                product_summary_source_ids,
                traction_source_ids,
                limit=2,
            ),
        },
        "technical_differentiation": {
            "text": build_technical_differentiation_text(
                company_name=company_name,
                product_market=product_market,
            ),
            "source_ids": technical_moat_source_ids,
        },
        "ai_data_advantage": {
            "text": build_ai_data_advantage_text(
                company_name=company_name,
                product_market=product_market,
            ),
            "source_ids": data_loop_source_ids,
        },
        "founders_team": {
            "text": build_founders_team_text(
                company_name=company_name,
                investigate=investigate,
            ),
            "source_ids": investigate_source_ids,
        },
        "commercialization_progress": {
            "text": build_commercialization_progress_text(
                company_name=company_name,
                traction=traction,
                product_market=product_market,
            ),
            "source_ids": traction_source_ids,
        },
        "customers_partnerships_performance": {
            "text": build_customers_partnerships_text(
                company_name=company_name,
                traction=traction,
            ),
            "source_ids": traction_source_ids,
        },
        "market_risk": {
            "text": build_market_risk_text(
                company_name=company_name,
                review_state=review_state,
                traction=traction,
            ),
            "source_ids": traction_source_ids,
        },
        "technical_risk": {
            "text": build_technical_risk_text(
                company_name=company_name,
                product_market=product_market,
                risk=risk,
                review_state=review_state,
            ),
            "source_ids": merge_source_id_lists(
                technical_moat_source_ids,
                risk_source_ids,
                limit=2,
            ),
        },
        "regulatory_operational_risk": {
            "text": build_regulatory_operational_risk_text(
                company_name=company_name,
                risk=risk,
            ),
            "source_ids": risk_source_ids,
        },
        "investment_points": {
            "text": build_investment_points_text(
                company_name=company_name,
                eval_state=eval_state,
                decision=decision,
            ),
            "source_ids": merge_source_id_lists(
                investigate_source_ids,
                product_summary_source_ids,
                traction_source_ids,
                limit=3,
            ),
        },
        "overall_evaluation": {
            "text": build_overall_evaluation_text(
                company_name=company_name,
                eval_state=eval_state,
                review_state=review_state,
                decision=decision,
            ),
            "source_ids": summary_source_ids,
        },
        "final_investment_judgment": {
            "text": build_final_judgment_text(
                decision=decision,
                company_name=company_name,
                key_risks=cast(list[str], eval_state.get("key_risks") or []),
            ),
            "source_ids": summary_source_ids,
        },
    }
    return normalize_report_draft(
        ReportDraftOutput.model_validate(base_output),
        source_catalog,
    )


def build_report_input(
    *,
    selected_company: dict[str, Any] | None,
    force_report_generation: bool,
    company_search_summary: str,
    selected_company_reason: str,
    investigate_members_state: ResearchAgentState | None,
    agent_product_market_analysis_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
    agent_risk_search_state: ResearchAgentState | None,
    review_state: ReviewAggregateState | None,
    eval_state: EvalState,
    source_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "selected_company": normalize_selected_company(selected_company),
        "report_policy": {
            "force_report_generation": force_report_generation,
            "eval_ready_for_report": bool(eval_state.get("ready_for_report")),
            "eval_status": str(eval_state.get("status", "")),
        },
        "selection_context": {
            "company_search_summary": normalize_text(company_search_summary),
            "selected_company_reason": normalize_text(selected_company_reason),
        },
        "grounding_notes": {
            "selected_company_context_only": True,
            "selection_context_not_external_evidence": True,
            "review_eval_are_internal_assessment_only": True,
            "product_market_unreferenced_fields": list_unreferenced_analysis_fields(
                agent_product_market_analysis_state
            ),
        },
        "eval": {
            "status": str(eval_state.get("status", "")),
            "final_decision": str(eval_state.get("final_decision", "watch")),
            "summary": normalize_text(eval_state.get("summary", "")),
            "criteria_scores": list(eval_state.get("criteria_scores", []) or []),
            "key_strengths": list(eval_state.get("key_strengths", []) or []),
            "key_risks": list(eval_state.get("key_risks", []) or []),
        },
        "review": {
            "summary": normalize_text((review_state or {}).get("summary", "")),
            "cautions": list((review_state or {}).get("cautions", []) or []),
            "contradictions": list((review_state or {}).get("contradictions", []) or []),
        },
        "agents": {
            "investigate_members": build_agent_payload(investigate_members_state),
            "agent_product_market_analysis": build_agent_payload(agent_product_market_analysis_state),
            "traction": build_agent_payload(traction_state),
            "agent_risk_search": build_agent_payload(agent_risk_search_state),
        },
        "source_catalog": [
            {
                "id": item["id"],
                "agent_name": item["agent_name"],
                "category": item["category"],
                "citation": item["citation"],
                "title": item.get("title", ""),
                "published_at": item.get("published_at", ""),
                "url": item.get("url", ""),
                "excerpt": item.get("excerpt", ""),
            }
            for item in source_catalog
        ],
    }


def build_executive_summary_text(
    *,
    company_name: str,
    eval_state: dict[str, Any],
    investigate: dict[str, Any],
    product_market: dict[str, Any],
    traction: dict[str, Any],
    risk: dict[str, Any],
    review_state: dict[str, Any],
) -> str:
    final_judgment_text = build_final_judgment_text(
        decision=str(eval_state.get("final_decision") or "watch"),
        company_name=company_name,
        key_risks=cast(list[str], eval_state.get("key_risks") or []),
    )
    return compose_report_paragraphs(
        [
            first_non_empty(
                str(eval_state.get("summary", "")),
                f"{company_name}는 Robotics AI 투자 검토 대상이며 팀, 시장, 기술, 리스크를 종합해 판단할 필요가 있다.",
            ),
        ],
        [
            first_non_empty(
                str(investigate.get("summary", "")),
                f"{company_name}는 창업자와 핵심팀의 공개 근거를 바탕으로 실행 역량을 점검할 필요가 있다.",
            ),
            first_non_empty(
                extract_grounded_analysis_text(product_market, "product_summary"),
                f"{company_name}의 제품과 시장성은 공개 자료 기준 일부 긍정 신호가 확인된다.",
            ),
        ],
        [
            first_non_empty(
                str(traction.get("summary", "")),
                join_list(review_state.get("cautions", []), limit=1),
                f"{company_name}의 상용화 진척과 시장 검증 수준은 추가 확인이 필요하다.",
            ),
            first_non_empty(
                str(risk.get("summary", "")),
                final_judgment_text,
            ),
            final_judgment_text,
        ],
    )


def build_company_intro_text(
    *,
    company_name: str,
    description: str,
    investigate: dict[str, Any],
    product_market: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        description or f"{company_name}는 Robotics AI 기반 자동화 솔루션을 제공하는 기업으로 파악된다.",
        first_non_empty(
            extract_grounded_analysis_text(product_market, "product_summary"),
            f"{company_name}는 공개 자료 기준 특정 산업 현장의 비효율을 줄이는 방향으로 제품을 전개한다.",
        ),
        first_non_empty(
            build_investigate_fallback_text(investigate),
            f"{company_name}의 경영진은 제품화와 사업 확장을 동시에 추진하는 구조로 보인다.",
        ),
    )


def build_problem_solution_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "target_kpi_logic"),
            f"{company_name}는 고객의 비용 절감과 운영 효율 개선 요구를 해결하려는 접근을 제시한다.",
        ),
        f"{company_name}의 솔루션은 단순 기능 제공보다 현장 도입의 경제성과 운영 개선 효과를 함께 설명하는 구조가 중요하다.",
    )


def build_products_services_text(
    *,
    company_name: str,
    product_name: str,
    product_market: dict[str, Any],
    traction: dict[str, Any],
) -> str:
    product_label = product_name or company_name
    return compose_report_paragraph(
        f"주요 제품은 {product_label}를 중심으로 구성되며, 공개 자료 기준 AI 소프트웨어와 운영 워크플로 개선 기능이 결합된 형태로 보인다.",
        first_non_empty(
            extract_grounded_analysis_text(product_market, "product_summary"),
            f"{company_name}의 제품 구조는 특정 고객군의 실제 업무 흐름에 맞춘 적용형 솔루션에 가깝다.",
        ),
        first_non_empty(
            str(traction.get("summary", "")),
            f"{company_name}는 제품을 단일 기능이 아니라 사업화 가능한 서비스 묶음으로 확장하려는 움직임이 보인다.",
        ),
    )


def build_market_size_growth_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
    traction: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "product_summary"),
            f"{company_name}가 겨냥한 시장은 성장 여지가 있으나 실제 침투 속도는 더 확인해야 한다.",
        ),
        first_non_empty(
            str(traction.get("summary", "")),
            f"{company_name}의 성장성은 상용화 신호와 고객 반응이 얼마나 반복적으로 확인되는지에 따라 달라진다.",
        ),
    )


def build_customer_demand_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
    traction: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "target_kpi_logic"),
            f"{company_name}는 고객의 시간 절감, 정확도 개선, 운영 효율화 요구를 겨냥한다.",
        ),
        first_non_empty(
            join_list(traction.get("findings", []), limit=2),
            f"{company_name}의 고객 수요는 공개된 도입 사례와 사업화 신호를 통해 추가 검증할 필요가 있다.",
        ),
    )


def build_competitive_landscape_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
    risk: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "technical_moat"),
            f"{company_name}의 경쟁력은 기술 해자와 현장 통합 역량에서 판가름 난다.",
        ),
        first_non_empty(
            str(risk.get("summary", "")),
            f"{company_name}는 경쟁 심화와 규제 환경 변화 속에서 차별성을 지속적으로 증명해야 한다.",
        ),
    )


def build_product_maturity_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
    traction: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "product_summary"),
            f"{company_name}의 제품 완성도는 일부 실증 신호가 있으나 추가 검증이 필요하다.",
        ),
        first_non_empty(
            str(traction.get("summary", "")),
            f"{company_name}의 제품 성숙도는 반복 도입과 운영 성과가 얼마나 확인되는지에 달려 있다.",
        ),
    )


def build_technical_differentiation_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "technical_moat"),
            f"{company_name}의 기술 차별성은 현장 적용 난이도와 시스템 통합 역량에서 나온다.",
        ),
        first_non_empty(
            extract_grounded_analysis_text(product_market, "product_summary"),
            f"{company_name}는 경쟁사 대비 실제 문제 해결 단위에서 차별화를 입증해야 한다.",
        ),
    )


def build_ai_data_advantage_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            extract_grounded_analysis_text(product_market, "data_loop_structure"),
            f"{company_name}의 AI 및 데이터 우위는 운영 과정에서 축적되는 데이터 루프가 핵심이다.",
        ),
        f"{company_name}는 단발성 모델 성능보다 데이터 축적, 개선 주기, 운영 fallback 구조를 함께 보여줄 필요가 있다.",
    )


def build_founders_team_text(
    *,
    company_name: str,
    investigate: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            str(investigate.get("summary", "")),
            build_investigate_fallback_text(investigate),
            f"{company_name}의 창업자 및 핵심팀 공개 근거는 일부 확보됐다.",
        ),
        first_non_empty(
            join_list(cast(dict[str, Any], investigate.get("structured_output") or {}).get("strengths", []), limit=2),
            f"{company_name}는 경영진의 도메인 전문성과 사업화 이력이 투자 판단의 중요한 축이다.",
        ),
        first_non_empty(
            join_list(cast(dict[str, Any], investigate.get("structured_output") or {}).get("evidence_gaps", []), limit=2),
            f"{company_name}는 비CEO 핵심 인력에 대한 추가 검증이 필요하다.",
        ),
    )


def build_commercialization_progress_text(
    *,
    company_name: str,
    traction: dict[str, Any],
    product_market: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            str(traction.get("summary", "")),
            f"{company_name}의 사업화 진행 상황은 공개된 파트너십, 고객, 투자 신호를 통해 추적해야 한다.",
        ),
        first_non_empty(
            join_list(traction.get("findings", []), limit=2),
            f"{company_name}는 제품 개발에서 실제 매출화와 반복 도입으로 넘어가는 구간의 확인이 중요하다.",
        ),
        first_non_empty(
            extract_grounded_analysis_text(product_market, "product_summary"),
            f"{company_name}는 기술 논리와 상용화 속도가 함께 맞물릴 때 투자 매력도가 높아진다.",
        ),
    )


def build_customers_partnerships_text(
    *,
    company_name: str,
    traction: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            join_list(traction.get("findings", []), limit=3),
            f"{company_name}의 고객 및 파트너십 신호는 일부 확인되지만 반복성은 추가 검증이 필요하다.",
        ),
        f"{company_name}의 실적 평가는 단일 기사나 단발성 협력보다 실제 고객 확산과 재계약 신호가 얼마나 이어지는지에 달려 있다.",
    )


def build_market_risk_text(
    *,
    company_name: str,
    review_state: dict[str, Any],
    traction: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            join_list(review_state.get("cautions", []), limit=2),
            f"{company_name}의 시장 리스크는 고객 확산 속도와 도입 전환율에 있다.",
        ),
        first_non_empty(
            str(traction.get("summary", "")),
            f"{company_name}는 공개된 traction 신호가 약할 경우 시장 검증 리스크가 더 크게 해석된다.",
        ),
    )


def build_technical_risk_text(
    *,
    company_name: str,
    product_market: dict[str, Any],
    risk: dict[str, Any],
    review_state: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            join_list(review_state.get("contradictions", []), limit=1, key="concern"),
            join_list(risk.get("findings", []), limit=1),
            f"{company_name}는 기술 차별성 주장 대비 실증 수준을 더 확인할 필요가 있다.",
        ),
        first_non_empty(
            extract_grounded_analysis_text(product_market, "technical_moat"),
            f"{company_name}의 기술 리스크는 완성도와 반복 성능이 실제 운영 환경에서도 유지되는지에 있다.",
        ),
    )


def build_regulatory_operational_risk_text(
    *,
    company_name: str,
    risk: dict[str, Any],
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            str(risk.get("summary", "")),
            f"{company_name}는 공개 자료 기준 중대한 규제 이슈는 제한적으로 보인다.",
        ),
        f"{company_name}는 인증, 컴플라이언스, 운영 안정성 이슈가 사업 확장 단계에서 어떻게 관리되는지 지속 점검이 필요하다.",
    )


def build_investment_points_text(
    *,
    company_name: str,
    eval_state: dict[str, Any],
    decision: str,
) -> str:
    return compose_report_paragraph(
        join_list(eval_state.get("key_strengths", []), limit=3)
        or decision_label_text(decision, positive=True),
        f"{company_name}의 투자 포인트는 팀, 제품, 시장 검증 신호가 서로 연결되는지에 있다.",
    )


def build_overall_evaluation_text(
    *,
    company_name: str,
    eval_state: dict[str, Any],
    review_state: dict[str, Any],
    decision: str,
) -> str:
    return compose_report_paragraph(
        first_non_empty(
            str(eval_state.get("summary", "")),
            decision_label_text(decision, positive=False),
        ),
        first_non_empty(
            join_list(review_state.get("cautions", []), limit=2),
            f"{company_name}는 강점과 리스크가 공존하는 단계로 해석된다.",
        ),
    )


def normalize_selected_company(selected_company: dict[str, Any] | None) -> dict[str, Any]:
    company = selected_company or {}
    return {
        "company_id": get_company_id(selected_company) or "",
        "company_name": get_company_name(selected_company),
        "product_name": normalize_text(company.get("product_name", "")),
        "description": normalize_text(company.get("description", "")),
        "invest_level": normalize_text(company.get("invest_level", "")),
        "categories": [str(item).strip() for item in company.get("categories", []) or [] if str(item).strip()],
    }


def build_agent_payload(agent_state: ResearchAgentState | None) -> dict[str, Any]:
    if not agent_state:
        return {
            "status": "missing",
            "summary": "",
            "findings": [],
            "structured_output": {},
        }
    return {
        "status": str(agent_state.get("status", "")),
        "summary": normalize_text(agent_state.get("summary", "")),
        "findings": [normalize_text(item) for item in list(agent_state.get("findings", []) or []) if normalize_text(item)],
        "structured_output": normalize_json_like(agent_state.get("structured_output")),
    }


def list_unreferenced_analysis_fields(
    agent_product_market_analysis_state: ResearchAgentState | None,
) -> list[str]:
    structured = cast(
        dict[str, Any],
        (agent_product_market_analysis_state or {}).get("structured_output") or {},
    )
    unreferenced: list[str] = []
    for field_name in (
        "target_kpi_logic",
        "technical_moat",
        "data_loop_structure",
        "product_summary",
    ):
        field = structured.get(field_name)
        if not isinstance(field, dict):
            continue
        references = [
            normalize_text(item)
            for item in field.get("references", []) or []
            if normalize_text(item)
        ]
        if references:
            continue
        unreferenced.append(field_name)
    return unreferenced


def build_source_catalog(
    *,
    investigate_members_state: ResearchAgentState | None,
    agent_product_market_analysis_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
    agent_risk_search_state: ResearchAgentState | None,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    agent_source_map = {
        "investigate_members": list((investigate_members_state or {}).get("sources", []) or []),
        "agent_product_market_analysis": list((agent_product_market_analysis_state or {}).get("sources", []) or []),
        "traction": list((traction_state or {}).get("sources", []) or []),
        "agent_risk_search": list((agent_risk_search_state or {}).get("sources", []) or []),
    }

    for agent_name, sources in agent_source_map.items():
        for source in sources:
            normalized = normalize_report_source(agent_name, cast(dict[str, Any], source))
            if not normalized:
                continue
            dedupe_key = (
                normalized["category"],
                normalized.get("title", ""),
                normalized.get("author", ""),
                normalized.get("published_at", ""),
                normalized.get("url", "") or normalized.get("source_path", ""),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            normalized["id"] = f"SRC{len(catalog) + 1:03d}"
            normalized["citation"] = format_reference_entry(normalized)
            catalog.append(normalized)
    return catalog


def normalize_report_source(
    agent_name: str,
    source: dict[str, Any],
) -> dict[str, Any] | None:
    source_type = normalize_text(source.get("source_type", ""))
    if source_type == "selected_company":
        return None

    title = normalize_text(source.get("title", ""))
    author = normalize_text(source.get("author", ""))
    publisher = normalize_text(source.get("publisher", ""))
    organization = normalize_text(source.get("organization", ""))
    journal = normalize_text(source.get("journal", ""))
    published_at = normalize_text(source.get("published_at", ""))
    url = normalize_text(source.get("url", "")) or normalize_text(source.get("source", ""))
    source_path = normalize_text(source.get("source_path", ""))
    excerpt = normalize_text(source.get("excerpt", "")) or normalize_text(source.get("snippet", ""))
    excerpt = excerpt[:400]
    raw_source_id = normalize_text(source.get("source_id", ""))

    if not title and url:
        title = derive_title_from_url(url)
    if not publisher and not organization:
        derived_publisher = derive_publisher(source, url)
        if derived_publisher:
            publisher = derived_publisher

    category = classify_report_source(
        source_type=source_type,
        title=title,
        author=author,
        publisher=publisher,
        organization=organization,
        journal=journal,
        url=url,
        source_path=source_path,
    )
    if category == "웹페이지" and not url:
        return None
    if not title and not url and not source_path:
        return None

    return {
        "agent_name": agent_name,
        "category": category,
        "title": title,
        "author": author,
        "publisher": publisher,
        "organization": organization,
        "journal": journal,
        "published_at": published_at,
        "url": url,
        "source_path": source_path,
        "excerpt": excerpt,
        "source_type": source_type,
        "raw_source_id": raw_source_id,
    }


def classify_report_source(
    *,
    source_type: str,
    title: str,
    author: str,
    publisher: str,
    organization: str,
    journal: str,
    url: str,
    source_path: str,
) -> str:
    normalized_title = title.lower()
    if journal or any(token in normalized_title for token in ("journal", "survey", "review", "conference", "학술", "논문")):
        return "학술 논문"
    if source_path.endswith(".pdf") or source_type in {"rag_document", "pdf"}:
        return "기관 보고서" if publisher or organization or title else "웹페이지"
    if any(token in normalized_title for token in ("report", "outlook", "insight", "brief", "리포트", "보고서", "브리프")):
        return "기관 보고서"
    if url:
        return "웹페이지"
    if publisher or organization:
        return "기관 보고서"
    return "웹페이지"


def format_reference_entry(source: dict[str, Any]) -> str:
    category = str(source.get("category", "웹페이지"))
    year = extract_year(str(source.get("published_at", "")))
    date = normalize_reference_date(str(source.get("published_at", "")))
    title = source.get("title", "") or "제목 미상"
    author = source.get("author", "")
    publisher = source.get("publisher", "") or source.get("organization", "")
    journal = source.get("journal", "")
    url = source.get("url", "")
    source_path = source.get("source_path", "")
    site_name = derive_site_name(url) or publisher or author

    if category == "기관 보고서":
        agency = publisher or author or site_name or "발행기관 미상"
        suffix = url or source_path
        return join_reference_parts(
            [
                f"{agency}({year})",
                f"*{title}*",
                suffix,
            ]
        )

    if category == "학술 논문":
        writer = author or publisher or site_name or "저자 미상"
        tail = f"*{journal}*" if journal else ""
        suffix = url or source_path
        return join_reference_parts(
            [
                f"{writer}({year})",
                title,
                tail,
                suffix,
            ]
        )

    writer = publisher or author or site_name or "작성자 미상"
    return join_reference_parts(
        [
            f"{writer}({date})",
            f"*{title}*",
            site_name,
            url or source_path,
        ]
    )


def build_reference_sections(
    source_catalog: list[dict[str, Any]],
    used_source_ids: list[str],
) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {
        category: []
        for category in REFERENCE_CATEGORY_ORDER
    }
    id_to_source = {str(item["id"]): item for item in source_catalog}
    for source_id in used_source_ids:
        source = id_to_source.get(source_id)
        if not source:
            continue
        category = str(source["category"])
        entries.setdefault(category, [])
        entries[category].append(str(source["citation"]))
    return entries


def collect_used_source_ids(
    draft: ReportDraftOutput,
    source_catalog: list[dict[str, Any]],
) -> list[str]:
    valid_source_ids = {str(item["id"]) for item in source_catalog}
    collected: list[str] = []
    for field_name in ReportDraftOutput.model_fields:
        field_value = cast(ReportFieldOutput, getattr(draft, field_name))
        collected.extend(
            source_id
            for source_id in field_value.source_ids
            if source_id in valid_source_ids
        )
    return dedupe_keep_order(collected)


def render_report_markdown(
    *,
    draft: ReportDraftOutput,
    company_name: str,
    company_id: str,
    generated_at: str,
    force_report_generation: bool,
    reference_sections: dict[str, list[str]],
) -> str:
    summary_text = draft.executive_summary.text or "요약을 생성하지 못했다."
    lines = [
        "# 투자 보고서",
        "",
        f"- 대상 기업: {company_name}",
        f"- 회사 ID: {company_id}",
        f"- 생성 시각: {generated_at}",
        (
            "- 보고서 모드: 강제 생성"
            if force_report_generation
            else "- 보고서 모드: 일반 생성"
        ),
        "",
        "## SUMMARY (Executive Summary)",
    ]
    append_markdown_paragraphs(lines, summary_text)
    lines.append("")
    for section_title, fields in REPORT_SECTION_FIELDS:
        lines.append(f"## {section_title}")
        for label, field_name in fields:
            field = cast(ReportFieldOutput, getattr(draft, field_name))
            lines.append(f"- {label}: {field.text or '근거 부족으로 요약을 생성하지 못했다.'}")
        lines.append("")

    lines.append("## REFERENCE")
    if not any(reference_sections.get(category) for category in REFERENCE_CATEGORY_ORDER):
        lines.append("- 활용 가능한 공개 참고자료를 정규화하지 못했다.")
    else:
        for category in REFERENCE_CATEGORY_ORDER:
            entries = reference_sections.get(category) or []
            if not entries:
                continue
            lines.append(f"### {category}")
            lines.extend(f"- {entry}" for entry in entries)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_report_pdf(
    *,
    pdf_path: Path,
    company_name: str,
    company_id: str,
    generated_at: str,
    force_report_generation: bool,
    draft: ReportDraftOutput,
    reference_sections: dict[str, list[str]],
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    font_name = ensure_report_font_registered()
    styles = build_pdf_styles(font_name)
    story: list[Any] = [
        Paragraph("투자 보고서", styles["title"]),
        Spacer(1, 4 * mm),
        Paragraph(f"대상 기업: {escape(company_name)}", styles["meta"]),
        Paragraph(f"회사 ID: {escape(company_id)}", styles["meta"]),
        Paragraph(f"생성 시각: {escape(generated_at)}", styles["meta"]),
        Paragraph(
            (
                "보고서 모드: 강제 생성"
                if force_report_generation
                else "보고서 모드: 일반 생성"
            ),
            styles["meta"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("SUMMARY (Executive Summary)", styles["section"]),
    ]
    append_pdf_paragraphs(
        story,
        draft.executive_summary.text or "요약을 생성하지 못했다.",
        styles["body"],
    )
    story.append(Spacer(1, 4 * mm))

    for section_title, fields in REPORT_SECTION_FIELDS:
        story.append(Paragraph(section_title, styles["section"]))
        for label, field_name in fields:
            field = cast(ReportFieldOutput, getattr(draft, field_name))
            text = field.text or "근거 부족으로 요약을 생성하지 못했다."
            story.append(
                Paragraph(
                    f"<b>{escape(label)}</b>: {escape(text)}",
                    styles["body"],
                )
            )
            story.append(Spacer(1, 1.5 * mm))
        story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("REFERENCE", styles["section"]))
    if not any(reference_sections.get(category) for category in REFERENCE_CATEGORY_ORDER):
        story.append(Paragraph("활용 가능한 공개 참고자료를 정규화하지 못했다.", styles["body"]))
    else:
        for category in REFERENCE_CATEGORY_ORDER:
            entries = reference_sections.get(category) or []
            if not entries:
                continue
            story.append(Paragraph(category, styles["subsection"]))
            for entry in entries:
                story.append(Paragraph(f"- {escape(entry)}", styles["body"]))
                story.append(Spacer(1, 1 * mm))
            story.append(Spacer(1, 1.5 * mm))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(story)


def build_pdf_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base_styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "WonwonTitle",
            parent=base_styles["Title"],
            fontName=font_name,
            fontSize=18,
            leading=24,
            textColor=colors.HexColor("#111827"),
        ),
        "meta": ParagraphStyle(
            "WonwonMeta",
            parent=base_styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#4B5563"),
        ),
        "section": ParagraphStyle(
            "WonwonSection",
            parent=base_styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=17,
            spaceBefore=4,
            spaceAfter=4,
            textColor=colors.HexColor("#0F172A"),
        ),
        "subsection": ParagraphStyle(
            "WonwonSubsection",
            parent=base_styles["Heading3"],
            fontName=font_name,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#1F2937"),
        ),
        "body": ParagraphStyle(
            "WonwonBody",
            parent=base_styles["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#111827"),
        ),
    }


def ensure_report_font_registered() -> str:
    if pdfmetrics.getRegisteredFontNames() and REPORT_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return REPORT_FONT_NAME

    font_path = resolve_report_font_path()
    if font_path.exists():
        pdfmetrics.registerFont(TTFont(REPORT_FONT_NAME, str(font_path)))
        return REPORT_FONT_NAME
    logger.warning("[report/pdf] font_path_missing path=%s", font_path)
    return "Helvetica"


def build_investigate_fallback_text(investigate: dict[str, Any]) -> str:
    structured = cast(dict[str, Any], investigate.get("structured_output") or {})
    ceo = cast(dict[str, Any], structured.get("ceo") or {})
    key_members = cast(list[dict[str, Any]], structured.get("key_members") or [])
    ceo_name = str(ceo.get("name", "")).strip()
    ceo_role = str(ceo.get("current_role", "")).strip()
    if ceo_name and key_members:
        return f"{ceo_name}({ceo_role or 'CEO'})를 포함한 핵심팀 공개 근거가 확인된다."
    if ceo_name:
        return f"{ceo_name}({ceo_role or 'CEO'}) 중심의 경영진 근거는 있으나 핵심팀 보강이 필요하다."
    return ""


def build_final_judgment_text(
    *,
    decision: str,
    company_name: str,
    key_risks: list[str],
) -> str:
    decision_text = {
        "invest": "투자 검토를 적극적으로 이어갈 만한 후보로 판단된다.",
        "watch": "추가 검증을 전제로 관찰 대상에 두는 판단이 적절하다.",
        "pass": "현시점에서는 보수적으로 패스 판단이 합리적이다.",
    }.get(decision, "추가 검증이 필요한 관찰 대상으로 보는 판단이 적절하다.")
    risk_suffix = f" 핵심 리스크는 {key_risks[0]}" if key_risks else ""
    return f"{company_name}는 {decision_text}{risk_suffix}".strip()


def decision_label_text(decision: str, *, positive: bool) -> str:
    labels = {
        "invest": (
            "핵심 강점은 팀과 시장 진입 신호가 함께 확인된다는 점이다.",
            "종합적으로 투자 검토를 이어갈 수 있는 후보로 본다.",
        ),
        "watch": (
            "긍정 신호는 존재하지만 추가 검증 포인트가 분명하다.",
            "종합적으로는 관찰 대상에 두고 후속 검증이 필요한 상태다.",
        ),
        "pass": (
            "긍정 신호보다 검증 부족과 리스크가 더 크게 남아 있다.",
            "종합적으로는 현시점 패스 판단이 보수적으로 타당하다.",
        ),
    }
    return labels.get(decision, labels["watch"])[0 if positive else 1]


def choose_source_ids(
    source_catalog: list[dict[str, Any]],
    *,
    agent_names: Iterable[str] | None = None,
    limit: int = 2,
) -> list[str]:
    allowed_agents = set(agent_names or [])
    collected: list[str] = []
    for item in source_catalog:
        if allowed_agents and str(item.get("agent_name", "")) not in allowed_agents:
            continue
        collected.append(str(item["id"]))
        if len(collected) >= limit:
            break
    return collected


def extract_analysis_text(payload: dict[str, Any], field_name: str) -> str:
    structured = cast(dict[str, Any], payload.get("structured_output") or {})
    field = structured.get(field_name)
    if isinstance(field, dict):
        return normalize_text(field.get("text", ""))
    return normalize_text(field)


def extract_grounded_analysis_text(payload: dict[str, Any], field_name: str) -> str:
    structured = cast(dict[str, Any], payload.get("structured_output") or {})
    field = structured.get(field_name)
    if not isinstance(field, dict):
        return ""
    references = [
        normalize_text(item)
        for item in field.get("references", []) or []
        if normalize_text(item)
    ]
    evidence_gap = normalize_text(field.get("evidence_gap", ""))
    if not references or evidence_gap:
        return ""
    return normalize_text(field.get("text", ""))


def extract_analysis_reference_titles(payload: dict[str, Any], field_name: str) -> list[str]:
    structured = cast(dict[str, Any], payload.get("structured_output") or {})
    field = structured.get(field_name)
    if not isinstance(field, dict):
        return []
    evidence_gap = normalize_text(field.get("evidence_gap", ""))
    if evidence_gap:
        return []
    return dedupe_keep_order(field.get("references", []) or [])


def choose_source_ids_from_analysis_references(
    source_catalog: list[dict[str, Any]],
    payload: dict[str, Any],
    field_name: str,
    *,
    limit: int = 2,
) -> list[str]:
    return choose_source_ids_from_reference_titles(
        source_catalog,
        extract_analysis_reference_titles(payload, field_name),
        agent_names=("agent_product_market_analysis",),
        limit=limit,
    )


def choose_source_ids_from_investigate_payload(
    source_catalog: list[dict[str, Any]],
    investigate: dict[str, Any],
    *,
    limit: int = 2,
) -> list[str]:
    structured = cast(dict[str, Any], investigate.get("structured_output") or {})
    raw_source_ids: list[str] = []
    ceo = cast(dict[str, Any], structured.get("ceo") or {})
    raw_source_ids.extend(ceo.get("source_ids", []) or [])
    for member in cast(list[dict[str, Any]], structured.get("key_members") or []):
        raw_source_ids.extend(member.get("source_ids", []) or [])

    normalized_requested = {
        normalize_text(value)
        for value in raw_source_ids
        if normalize_text(value)
    }
    collected: list[str] = []
    for item in source_catalog:
        if str(item.get("agent_name", "")) != "investigate_members":
            continue
        raw_source_id = normalize_text(item.get("raw_source_id", ""))
        if raw_source_id and raw_source_id in normalized_requested:
            collected.append(str(item["id"]))
        if len(collected) >= limit:
            break
    return dedupe_keep_order(collected)


def choose_source_ids_from_reference_titles(
    source_catalog: list[dict[str, Any]],
    reference_titles: Iterable[Any],
    *,
    agent_names: Iterable[str] | None = None,
    limit: int = 2,
) -> list[str]:
    allowed_agents = set(agent_names or [])
    normalized_references = {
        normalize_match_key(value)
        for value in reference_titles
        if normalize_match_key(value)
    }
    if not normalized_references or limit <= 0:
        return []

    collected: list[str] = []
    for item in source_catalog:
        if allowed_agents and str(item.get("agent_name", "")) not in allowed_agents:
            continue
        title_key = normalize_match_key(item.get("title", ""))
        citation_key = normalize_match_key(item.get("citation", ""))
        if title_key not in normalized_references and citation_key not in normalized_references:
            continue
        collected.append(str(item["id"]))
        if len(collected) >= limit:
            break
    return dedupe_keep_order(collected)


def trim_summary_text(text: str) -> str:
    normalized = normalize_summary_text(text)
    if len(normalized) <= SUMMARY_MAX_CHARS:
        return normalized
    trimmed = normalized[: SUMMARY_MAX_CHARS - 1].rstrip()
    sentence_end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"), trimmed.rfind("다."))
    if sentence_end > 200:
        return normalize_summary_text(trimmed[: sentence_end + 1].strip())
    return normalize_summary_text(f"{trimmed}…")


def is_thin_report_text(text: str, *, field_name: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    minimum_sentences = 3 if field_name == "executive_summary" else 2
    return count_report_sentences(normalized) < minimum_sentences


def merge_report_texts(
    *,
    primary_text: str,
    fallback_text: str,
    field_name: str,
) -> str:
    if not primary_text:
        merged = fallback_text
    elif not fallback_text or primary_text == fallback_text:
        merged = primary_text
    elif field_name == "executive_summary":
        merged = merge_executive_summary_texts(primary_text, fallback_text)
    else:
        merged = compose_report_paragraph(primary_text, fallback_text)
    if field_name == "executive_summary":
        return trim_summary_text(merged)
    return normalize_text(merged)


def compose_report_paragraph(*parts: Any) -> str:
    sentences: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for sentence in split_report_sentences(normalize_text(part)):
            if sentence in seen:
                continue
            seen.add(sentence)
            sentences.append(sentence)
    return " ".join(sentences).strip()


def compose_report_paragraphs(*paragraph_groups: Any) -> str:
    paragraphs: list[str] = []
    seen_sentences: set[str] = set()
    for group in paragraph_groups:
        group_parts = group if isinstance(group, (list, tuple)) else [group]
        paragraph_sentences: list[str] = []
        for part in group_parts:
            for sentence in split_report_sentences(normalize_text(part)):
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                paragraph_sentences.append(sentence)
        if paragraph_sentences:
            paragraphs.append(" ".join(paragraph_sentences).strip())
    return "\n\n".join(paragraphs).strip()


def merge_executive_summary_texts(primary_text: str, fallback_text: str) -> str:
    paragraphs = split_text_paragraphs(primary_text) + split_text_paragraphs(fallback_text)
    if not paragraphs:
        return ""
    return compose_report_paragraphs(*paragraphs)


def split_report_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", normalized)
    sentences: list[str] = []
    for piece in pieces:
        cleaned = normalize_text(piece)
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        sentences.append(cleaned)
    return sentences


def count_report_sentences(text: str) -> int:
    return len(split_report_sentences(text))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_summary_text(value: Any) -> str:
    paragraphs = split_text_paragraphs(value)
    if not paragraphs:
        return ""
    if len(paragraphs) > 4:
        overflow = compose_report_paragraph(*paragraphs[3:])
        paragraphs = paragraphs[:3] + ([overflow] if overflow else [])
    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs)
    sentences = split_report_sentences(paragraphs[0])
    if len(sentences) < 2:
        return paragraphs[0]
    target_paragraphs = 2 if len(sentences) <= 3 else 3 if len(sentences) <= 6 else 4
    target_paragraphs = min(target_paragraphs, len(sentences), 4)
    sentence_groups = split_sentences_evenly(sentences, target_paragraphs)
    return "\n\n".join(" ".join(group).strip() for group in sentence_groups if group).strip()


def split_text_paragraphs(value: Any) -> list[str]:
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    paragraphs = [
        normalize_text(part)
        for part in re.split(r"\n\s*\n+", raw)
        if normalize_text(part)
    ]
    return paragraphs


def split_sentences_evenly(sentences: list[str], paragraph_count: int) -> list[list[str]]:
    if paragraph_count <= 1 or len(sentences) <= 1:
        return [sentences]
    base_size, remainder = divmod(len(sentences), paragraph_count)
    groups: list[list[str]] = []
    index = 0
    for group_index in range(paragraph_count):
        size = base_size + (1 if group_index < remainder else 0)
        if size <= 0:
            continue
        groups.append(sentences[index : index + size])
        index += size
    return groups


def append_markdown_paragraphs(lines: list[str], text: str) -> None:
    paragraphs = split_text_paragraphs(text) or ["요약을 생성하지 못했다."]
    for index, paragraph in enumerate(paragraphs):
        if index > 0:
            lines.append("")
        lines.append(paragraph)


def append_pdf_paragraphs(story: list[Any], text: str, style: ParagraphStyle) -> None:
    paragraphs = split_text_paragraphs(text) or ["요약을 생성하지 못했다."]
    for paragraph in paragraphs:
        story.append(Paragraph(escape(paragraph), style))
        story.append(Spacer(1, 1.5 * mm))


def normalize_json_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): normalize_json_like(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, list):
        return [normalize_json_like(item) for item in value]
    return value


def derive_publisher(source: dict[str, Any], url: str) -> str:
    explicit = normalize_text(source.get("site_name", ""))
    if explicit:
        return explicit
    return derive_site_name(url)


def derive_site_name(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.removeprefix("www.")
    return host


def derive_title_from_url(url: str) -> str:
    if not url:
        return ""
    path = urlparse(url).path.rstrip("/")
    if not path:
        return derive_site_name(url)
    return path.split("/")[-1] or derive_site_name(url)


def extract_year(published_at: str) -> str:
    match = re.search(r"(19|20)\d{2}", published_at or "")
    return match.group(0) if match else "n.d."


def normalize_reference_date(published_at: str) -> str:
    match = re.search(r"(19|20)\d{2}(?:[-./]\d{2})?(?:[-./]\d{2})?", published_at or "")
    if not match:
        return "n.d."
    value = match.group(0).replace(".", "-").replace("/", "-")
    if len(value) == 7:
        return f"{value}-01"
    return value


def join_reference_parts(parts: list[str]) -> str:
    return ". ".join(part for part in parts if part).strip().rstrip(".") + "."


def merge_source_id_lists(*groups: Iterable[Any], limit: int = 4) -> list[str]:
    merged = dedupe_keep_order(
        value
        for group in groups
        for value in group
    )
    return merged[:limit]


def dedupe_keep_order(values: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def normalize_match_key(value: Any) -> str:
    return "".join(normalize_text(value).lower().split())


def join_list(
    values: Iterable[Any],
    *,
    limit: int = 3,
    key: str | None = None,
) -> str:
    items: list[str] = []
    for value in values:
        if key and isinstance(value, dict):
            text = normalize_text(value.get(key, ""))
        else:
            text = normalize_text(value)
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return " ".join(items)
