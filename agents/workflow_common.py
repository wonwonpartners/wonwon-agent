from __future__ import annotations

import logging
from typing import Any, NotRequired, TypedDict

logger = logging.getLogger(__name__)


class ResearchAgentState(TypedDict):
    agent_name: str
    status: str
    attempt_count: int
    input_company_id: str | None
    summary: str
    findings: list[str]
    sources: list[dict[str, Any]]
    structured_output: NotRequired[dict[str, Any] | None]


class ReviewState(TypedDict):
    reviewed_agent: str
    decision: str
    reason: str
    review_count: int


class ReviewContradiction(TypedDict):
    topic: str
    concern: str
    severity: str
    related_agents: list[str]


class ReviewAggregateState(TypedDict):
    status: str
    summary: str
    agent_statuses: dict[str, str]
    cautions: list[str]
    contradictions: list[ReviewContradiction]


class EvalCriterionScore(TypedDict):
    criterion_id: str
    criterion_name: str
    score: int
    rationale: str


class EvalState(TypedDict):
    status: str
    ready_for_report: bool
    summary: str
    agent_summaries: dict[str, str]
    weighted_score: NotRequired[float]
    review_summary: NotRequired[str]
    review_cautions: NotRequired[list[str]]
    review_contradictions: NotRequired[list[ReviewContradiction]]
    agent_structured_highlights: NotRequired[dict[str, Any]]
    final_decision: NotRequired[str]
    criteria_scores: NotRequired[list[EvalCriterionScore]]
    key_strengths: NotRequired[list[str]]
    key_risks: NotRequired[list[str]]
    next_action: NotRequired[str]
    retry_reason: NotRequired[str]


class ReportState(TypedDict):
    status: str
    report_path: str
    pdf_path: str
    markdown: str


class GraphErrorState(TypedDict):
    stage: str
    agent_name: str
    message: str


def get_company_id(selected_company: dict[str, Any] | None) -> str | None:
    if not isinstance(selected_company, dict):
        return None
    company_id = selected_company.get("company_id")
    if company_id is None:
        return None
    return str(company_id)


def get_company_name(selected_company: dict[str, Any] | None) -> str:
    if not isinstance(selected_company, dict):
        return "알 수 없는 회사"
    company_name = selected_company.get("company_name")
    return str(company_name) if company_name else "알 수 없는 회사"


def build_skeleton_research_state(
    agent_name: str,
    selected_company: dict[str, Any] | None,
    previous_state: ResearchAgentState | None = None,
) -> ResearchAgentState:
    attempt_count = int((previous_state or {}).get("attempt_count", 0)) + 1
    company_id = get_company_id(selected_company)
    company_name = get_company_name(selected_company)

    if company_id is None:
        payload = {
            "agent_name": agent_name,
            "status": "skipped",
            "attempt_count": attempt_count,
            "input_company_id": None,
            "summary": "선정된 회사가 없어 스켈레톤 조사를 진행하지 못했습니다.",
            "findings": [
                "선행 단계에서 `selected_company`가 비어 있어 조사를 건너뛰었습니다.",
            ],
            "sources": [],
            "structured_output": None,
        }
        logger.info(
            "[%s/final] status=%s attempt=%s company_id=-",
            agent_name,
            payload["status"],
            payload["attempt_count"],
        )
        return payload

    payload = {
        "agent_name": agent_name,
        "status": "completed",
        "attempt_count": attempt_count,
        "input_company_id": company_id,
        "summary": (
            f"{company_name} ({company_id})에 대한 {agent_name} 스켈레톤 조사를 완료했습니다."
        ),
        "findings": [
            "이번 단계에서는 실제 외부 조사 대신 후속 구현을 위한 상태 구조만 생성했습니다.",
            "조사 결과는 selected_company 입력을 그대로 참조한 스켈레톤 출력입니다.",
            "review/eval/report 단계가 사용할 공통 필드를 채웠습니다.",
        ],
        "sources": [
            {
                "source_type": "selected_company",
                "company_id": company_id,
                "company_name": company_name,
            }
        ],
        "structured_output": None,
    }
    logger.info(
        "[%s/final] status=%s attempt=%s company=%s(%s)",
        agent_name,
        payload["status"],
        payload["attempt_count"],
        company_name,
        company_id,
    )
    return payload
