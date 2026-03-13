from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field

from agents.agent_eval.common import get_chat_model
from agents.agent_eval.prompts import (
    get_eval_system_prompt,
    render_eval_user_prompt,
)

from agents.workflow_common import (
    EvalCriterionScore,
    EvalState,
    ResearchAgentState,
    ReviewContradiction,
    ReviewAggregateState,
    get_company_id,
    get_company_name,
)


class EvalCriterionScoreOutput(BaseModel):
    criterion_id: str = Field(default="")
    criterion_name: str = Field(default="")
    score: int = Field(default=3)
    rationale: str = Field(default="")


class EvalDecisionOutput(BaseModel):
    final_decision: str = Field(default="watch")
    summary: str = Field(default="")
    criteria_scores: list[EvalCriterionScoreOutput] = Field(default_factory=list)
    key_strengths: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)


def build_eval_state(
    selected_company: dict[str, Any] | None,
    investigate_members_state: ResearchAgentState | None,
    agent_product_market_analysis_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
    agent_risk_search_state: ResearchAgentState | None,
    review_state: ReviewAggregateState | None,
) -> EvalState:
    company_name = get_company_name(selected_company)
    company_id = get_company_id(selected_company) or "알 수 없음"
    agent_states = {
        "investigate_members": investigate_members_state,
        "agent_product_market_analysis": agent_product_market_analysis_state,
        "traction": traction_state,
        "agent_risk_search": agent_risk_search_state,
    }
    agent_summaries = {
        agent_name: (agent_state.get("summary") if agent_state else "결과가 없습니다.")
        for agent_name, agent_state in agent_states.items()
    }
    ready_for_report = all(
        agent_state is not None and agent_state.get("status") == "completed"
        for agent_state in agent_states.values()
    )
    status = "completed" if ready_for_report else "blocked"
    review_cautions = list((review_state or {}).get("cautions", []) or [])
    review_summary = str((review_state or {}).get("summary", "") or "")
    review_contradictions = cast(
        list[ReviewContradiction],
        list((review_state or {}).get("contradictions", []) or []),
    )
    agent_structured_highlights = build_agent_structured_highlights(
        investigate_members_state=investigate_members_state,
        agent_product_market_analysis_state=agent_product_market_analysis_state,
        traction_state=traction_state,
        agent_risk_search_state=agent_risk_search_state,
    )
    eval_output = run_llm_eval(
        company_name=company_name,
        company_id=company_id,
        selected_company=selected_company,
        agent_states=agent_states,
        agent_summaries=agent_summaries,
        agent_structured_highlights=agent_structured_highlights,
        review_summary=review_summary,
        review_cautions=review_cautions,
        review_contradictions=review_contradictions,
        ready_for_report=ready_for_report,
        status=status,
    )
    return {
        "status": status,
        "ready_for_report": ready_for_report,
        "summary": eval_output["summary"],
        "agent_summaries": agent_summaries,
        "review_summary": review_summary,
        "review_cautions": review_cautions,
        "review_contradictions": review_contradictions,
        "agent_structured_highlights": agent_structured_highlights,
        "final_decision": eval_output["final_decision"],
        "criteria_scores": eval_output["criteria_scores"],
        "key_strengths": eval_output["key_strengths"],
        "key_risks": eval_output["key_risks"],
    }


def run_llm_eval(
    *,
    company_name: str,
    company_id: str,
    selected_company: dict[str, Any] | None,
    agent_states: dict[str, ResearchAgentState | None],
    agent_summaries: dict[str, str],
    agent_structured_highlights: dict[str, Any],
    review_summary: str,
    review_cautions: list[str],
    review_contradictions: list[ReviewContradiction],
    ready_for_report: bool,
    status: str,
) -> dict[str, Any]:
    payload = {
        "selected_company": selected_company,
        "graph_status": {
            "status": status,
            "ready_for_report": ready_for_report,
        },
        "agent_statuses": {
            agent_name: (
                agent_state.get("status", "missing") if agent_state else "missing"
            )
            for agent_name, agent_state in agent_states.items()
        },
        "agent_summaries": agent_summaries,
        "agent_structured_highlights": agent_structured_highlights,
        "review": {
            "summary": review_summary,
            "cautions": review_cautions,
            "contradictions": review_contradictions,
        },
    }
    try:
        reviewer = get_chat_model().with_structured_output(
            EvalDecisionOutput,
            method="json_schema",
        )
        result = reviewer.invoke(
            [
                ("system", get_eval_system_prompt()),
                ("user", render_eval_user_prompt(payload)),
            ]
        )
        criteria_scores = normalize_criteria_scores(result.criteria_scores)
        summary = result.summary.strip() or build_fallback_summary(
            company_name=company_name,
            company_id=company_id,
            status=status,
            ready_for_report=ready_for_report,
            review_cautions=review_cautions,
        )
        return {
            "final_decision": normalize_final_decision(result.final_decision),
            "summary": summary,
            "criteria_scores": criteria_scores,
            "key_strengths": [item.strip() for item in result.key_strengths if item.strip()],
            "key_risks": [item.strip() for item in result.key_risks if item.strip()],
        }
    except Exception:
        return build_fallback_eval_output(
            company_name=company_name,
            company_id=company_id,
            status=status,
            ready_for_report=ready_for_report,
            review_cautions=review_cautions,
            review_contradictions=review_contradictions,
            agent_structured_highlights=agent_structured_highlights,
        )


def build_fallback_eval_output(
    *,
    company_name: str,
    company_id: str,
    status: str,
    ready_for_report: bool,
    review_cautions: list[str],
    review_contradictions: list[ReviewContradiction],
    agent_structured_highlights: dict[str, Any],
) -> dict[str, Any]:
    risks = list(review_cautions)
    for contradiction in review_contradictions:
        concern = str(contradiction.get("concern", "")).strip()
        if concern and concern not in risks:
            risks.append(concern)
    strengths = build_fallback_strengths(agent_structured_highlights)
    return {
        "final_decision": "watch" if ready_for_report else "pass",
        "summary": build_fallback_summary(
            company_name=company_name,
            company_id=company_id,
            status=status,
            ready_for_report=ready_for_report,
            review_cautions=review_cautions,
        ),
        "criteria_scores": build_default_criteria_scores(),
        "key_strengths": strengths,
        "key_risks": risks or ["평가 근거가 제한적이어서 추가 검증이 필요합니다."],
    }


def build_fallback_summary(
    *,
    company_name: str,
    company_id: str,
    status: str,
    ready_for_report: bool,
    review_cautions: list[str],
) -> str:
    caution_suffix = (
        f" review-agent 유의사항 {len(review_cautions)}건을 함께 검토해야 합니다."
        if review_cautions
        else ""
    )
    return (
        f"{company_name} ({company_id})에 대한 4개 병렬 조사 결과를 묶어 "
        f"보고서 작성 가능 여부를 {status}로 정리했습니다."
        f"{caution_suffix}"
        if ready_for_report
        else (
            f"{company_name} ({company_id})는 일부 평가 근거가 부족하거나 실패한 조사 단계가 있어 "
            f"최종 보고서 작성 전 추가 검증이 필요합니다.{caution_suffix}"
        )
    )


def normalize_final_decision(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"invest", "watch", "pass"}:
        return normalized
    return "watch"


def normalize_criteria_scores(
    scores: list[EvalCriterionScoreOutput],
) -> list[EvalCriterionScore]:
    normalized: list[EvalCriterionScore] = []
    for item in scores:
        criterion_id = item.criterion_id.strip()
        criterion_name = item.criterion_name.strip()
        rationale = item.rationale.strip()
        score = max(1, min(int(item.score), 5))
        if not criterion_id or not criterion_name:
            continue
        normalized.append(
            {
                "criterion_id": criterion_id,
                "criterion_name": criterion_name,
                "score": score,
                "rationale": rationale,
            }
        )
    return normalized or build_default_criteria_scores()


def build_default_criteria_scores() -> list[EvalCriterionScore]:
    return [
        {
            "criterion_id": "C1",
            "criterion_name": "창업자 및 핵심팀 신뢰도",
            "score": 3,
            "rationale": "fallback 평가로 중립 점수를 부여했습니다.",
        },
        {
            "criterion_id": "C2",
            "criterion_name": "시장 문제 및 도입 논리 명확성",
            "score": 3,
            "rationale": "fallback 평가로 중립 점수를 부여했습니다.",
        },
        {
            "criterion_id": "C3",
            "criterion_name": "제품 완성도 및 시스템 차별성",
            "score": 3,
            "rationale": "fallback 평가로 중립 점수를 부여했습니다.",
        },
        {
            "criterion_id": "C4",
            "criterion_name": "상용화 진전도 및 시장 검증",
            "score": 3,
            "rationale": "fallback 평가로 중립 점수를 부여했습니다.",
        },
        {
            "criterion_id": "C5",
            "criterion_name": "AI 자율성 및 데이터 운영 우위",
            "score": 3,
            "rationale": "fallback 평가로 중립 점수를 부여했습니다.",
        },
        {
            "criterion_id": "C6",
            "criterion_name": "공개 리스크 및 안전·규제 대응",
            "score": 3,
            "rationale": "fallback 평가로 중립 점수를 부여했습니다.",
        },
    ]


def build_fallback_strengths(agent_structured_highlights: dict[str, Any]) -> list[str]:
    strengths: list[str] = []
    investigate = cast(dict[str, Any], agent_structured_highlights.get("investigate_members") or {})
    traction = cast(dict[str, Any], agent_structured_highlights.get("traction") or {})
    risk = cast(dict[str, Any], agent_structured_highlights.get("agent_risk_search") or {})
    if investigate.get("assessment_summary"):
        strengths.append(str(investigate["assessment_summary"]))
    if traction.get("traction_summary"):
        strengths.append(str(traction["traction_summary"]))
    if risk.get("risk_summary"):
        strengths.append(str(risk["risk_summary"]))
    return strengths[:3]


def build_agent_structured_highlights(
    *,
    investigate_members_state: ResearchAgentState | None,
    agent_product_market_analysis_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
    agent_risk_search_state: ResearchAgentState | None,
) -> dict[str, Any]:
    return {
        "investigate_members": build_investigate_members_highlights(
            investigate_members_state,
        ),
        "agent_product_market_analysis": build_product_market_highlights(
            agent_product_market_analysis_state,
        ),
        "traction": build_traction_highlights(traction_state),
        "agent_risk_search": build_risk_highlights(agent_risk_search_state),
    }


def build_investigate_members_highlights(
    agent_state: ResearchAgentState | None,
) -> dict[str, Any]:
    payload = cast(dict[str, Any], (agent_state or {}).get("structured_output") or {})
    ceo = cast(dict[str, Any], payload.get("ceo") or {})
    key_members = cast(list[dict[str, Any]], payload.get("key_members") or [])
    return {
        "status": (agent_state or {}).get("status", "missing"),
        "ceo": {
            "name": str(ceo.get("name", "")),
            "current_role": str(ceo.get("current_role", "")),
            "is_founder": bool(ceo.get("is_founder", False)),
            "experience_tags": list(ceo.get("experience_tags", []) or []),
        },
        "key_member_count": len(key_members),
        "key_members": [
            {
                "name": str(member.get("name", "")),
                "current_role": str(member.get("current_role", "")),
                "is_founder": bool(member.get("is_founder", False)),
                "experience_tags": list(member.get("experience_tags", []) or []),
            }
            for member in key_members[:5]
        ],
        "role_coverage": payload.get("role_coverage") or {},
        "strengths": list(payload.get("strengths", []) or []),
        "evidence_gaps": list(payload.get("evidence_gaps", []) or []),
        "assessment_summary": str(payload.get("assessment_summary", "")),
        "evidence_quality": str(payload.get("evidence_quality", "")),
    }


def build_product_market_highlights(
    agent_state: ResearchAgentState | None,
) -> dict[str, Any]:
    payload = cast(dict[str, Any], (agent_state or {}).get("structured_output") or {})
    return {
        "status": (agent_state or {}).get("status", "missing"),
        "target_kpi_logic": extract_analysis_field(payload.get("target_kpi_logic")),
        "technical_moat": extract_analysis_field(payload.get("technical_moat")),
        "data_loop_structure": extract_analysis_field(payload.get("data_loop_structure")),
        "product_summary": extract_analysis_field(payload.get("product_summary")),
    }


def build_traction_highlights(agent_state: ResearchAgentState | None) -> dict[str, Any]:
    payload = cast(dict[str, Any], (agent_state or {}).get("structured_output") or {})
    hiring = cast(dict[str, Any], payload.get("hiring_analysis") or {})
    return {
        "status": (agent_state or {}).get("status", "missing"),
        "partnerships": list(payload.get("partnerships", []) or []),
        "funding_velocity": list(payload.get("funding_velocity", []) or []),
        "hiring_analysis": {
            "field_engineer_ratio": hiring.get("field_engineer_ratio", 0),
            "field_engineer_count": hiring.get("field_engineer_count", 0),
            "hiring_trend_3m": hiring.get("hiring_trend_3m", 0),
        },
        "traction_summary": str(payload.get("traction_summary", "")),
    }


def build_risk_highlights(agent_state: ResearchAgentState | None) -> dict[str, Any]:
    payload = cast(dict[str, Any], (agent_state or {}).get("structured_output") or {})
    risk_state = cast(dict[str, Any], payload.get("risk_state") or {})
    return {
        "status": (agent_state or {}).get("status", "missing"),
        "legal_regulatory": str(risk_state.get("legal_regulatory", "")),
        "certification_status": list(risk_state.get("certification_status", []) or []),
        "red_flags": list(risk_state.get("red_flags", []) or []),
        "risk_summary": str(risk_state.get("risk_summary", "")),
    }


def extract_analysis_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "text": str(value.get("text", "") or value.get("target_kpi_logic", "") or ""),
            "references": list(value.get("references", []) or value.get("target_kpi_logic_sources", []) or []),
            "evidence_gap": str(value.get("evidence_gap", "") or ""),
        }
    if isinstance(value, str):
        return {
            "text": value,
            "references": [],
            "evidence_gap": "",
        }
    return {
        "text": "",
        "references": [],
        "evidence_gap": "",
    }
