from __future__ import annotations

import json
import logging
from typing import Any, cast

from agents.agent_review.output import (
    AgentBReviewNodeOutput,
    AgentCReviewNodeOutput,
    InvestigateMembersReviewNodeOutput,
    ProductMarketAnalysisReviewNodeOutput,
    TractionReviewNodeOutput,
)
from agents.agent_review.service import review_research_state
from agents.workflow_common import ResearchAgentState, ReviewState

logger = logging.getLogger(__name__)


def _log_investigate_members_review_input(
    agent_state: ResearchAgentState | None,
) -> None:
    if not agent_state:
        logger.info("[review/investigate_members/input] missing agent_state")
        return

    def shorten(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    structured_output = cast(
        dict[str, Any] | None,
        agent_state.get("structured_output"),
    ) or {}
    key_members = structured_output.get("key_members") or []
    sources = cast(list[dict[str, Any]], agent_state.get("sources") or [])
    ceo = cast(dict[str, Any] | None, structured_output.get("ceo")) or {}

    source_preview: list[dict[str, str]] = []
    for source in sources[:5]:
        source_preview.append(
            {
                "source_id": str(source.get("source_id", "")),
                "domain": str(source.get("domain", "")),
                "title": shorten(source.get("title", ""), 100),
                "url": str(source.get("url", "")),
            }
        )

    key_member_preview: list[dict[str, Any]] = []
    for member in key_members[:3]:
        if not isinstance(member, dict):
            continue
        key_member_preview.append(
            {
                "name": str(member.get("name", "")),
                "current_role": str(member.get("current_role", "")),
                "is_founder": bool(member.get("is_founder", False)),
                "experience_tags": list(member.get("experience_tags", []) or []),
                "source_ids": list(member.get("source_ids", []) or []),
                "confidence": member.get("confidence", 0),
                "evidence_summary": shorten(member.get("evidence_summary", ""), 140),
            }
        )

    payload = {
        "agent_name": agent_state.get("agent_name", ""),
        "status": agent_state.get("status", ""),
        "attempt_count": agent_state.get("attempt_count", 0),
        "input_company_id": agent_state.get("input_company_id"),
        "summary": shorten(agent_state.get("summary", ""), 220),
        "findings": [
            shorten(item, 160)
            for item in list(agent_state.get("findings", []) or [])
        ],
        "sources_count": len(sources),
        "sources_preview": source_preview,
        "structured_output": {
            "ceo": (
                {
                    "name": str(ceo.get("name", "")),
                    "current_role": str(ceo.get("current_role", "")),
                    "is_founder": bool(ceo.get("is_founder", False)),
                    "experience_tags": list(ceo.get("experience_tags", []) or []),
                    "source_ids": list(ceo.get("source_ids", []) or []),
                    "confidence": ceo.get("confidence", 0),
                    "evidence_summary": shorten(ceo.get("evidence_summary", ""), 140),
                }
                if ceo
                else None
            ),
            "key_members_count": len(key_members),
            "key_members_preview": key_member_preview,
            "role_coverage": structured_output.get("role_coverage") or {},
            "strengths": list(structured_output.get("strengths", []) or []),
            "evidence_gaps": list(structured_output.get("evidence_gaps", []) or []),
            "assessment_summary": shorten(
                structured_output.get("assessment_summary", ""),
                220,
            ),
            "evidence_quality": shorten(
                structured_output.get("evidence_quality", ""),
                180,
            ),
            "search_queries": list(structured_output.get("search_queries", []) or []),
        },
    }
    logger.info(
        "[review/investigate_members/input]\n%s",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _review_node(
    state: dict[str, Any],
    *,
    agent_key: str,
    review_key: str,
    agent_name: str,
) -> dict[str, ReviewState]:
    agent_state = cast(ResearchAgentState | None, state.get(agent_key))
    previous_review = cast(ReviewState | None, state.get(review_key))
    logger.info(
        "[review/%s/start] agent_status=%s previous_review_count=%s",
        agent_name,
        (agent_state or {}).get("status", "missing"),
        int((previous_review or {}).get("review_count", 0)),
    )
    if agent_name == "investigate_members":
        _log_investigate_members_review_input(agent_state)
    return {
        review_key: review_research_state(
            agent_name,
            agent_state,
            previous_review,
        )
    }


def review_investigate_members_node(
    state: dict[str, Any],
) -> InvestigateMembersReviewNodeOutput:
    return cast(
        InvestigateMembersReviewNodeOutput,
        _review_node(
            state,
            agent_key="investigate_members_state",
            review_key="investigate_members_review",
            agent_name="investigate_members",
        ),
    )


def review_traction_node(state: dict[str, Any]) -> TractionReviewNodeOutput:
    return cast(
        TractionReviewNodeOutput,
        _review_node(
            state,
            agent_key="traction_state",
            review_key="traction_review",
            agent_name="traction",
        ),
    )


def review_agent_product_market_analysis_node(state: dict[str, Any]) -> ProductMarketAnalysisReviewNodeOutput:
    return cast(
        ProductMarketAnalysisReviewNodeOutput,
        _review_node(
            state,
            agent_key="agent_product_market_analysis_state",
            review_key="agent_product_market_analysis_review",
            agent_name="agent_product_market_analysis",
        ),
    )


def review_agent_b_node(state: dict[str, Any]) -> AgentBReviewNodeOutput:
    return cast(
        AgentBReviewNodeOutput,
        _review_node(
            state,
            agent_key="agent_b_state",
            review_key="agent_b_review",
            agent_name="agent_b",
        ),
    )


def review_agent_c_node(state: dict[str, Any]) -> AgentCReviewNodeOutput:
    return cast(
        AgentCReviewNodeOutput,
        _review_node(
            state,
            agent_key="agent_c_state",
            review_key="agent_c_review",
            agent_name="agent_c",
        ),
    )
