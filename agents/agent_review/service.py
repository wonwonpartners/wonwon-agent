from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from agents.agent_review.common import get_chat_model, get_fallback_chat_model
from agents.agent_review.prompts import (
    get_parallel_review_system_prompt,
    render_parallel_review_user_prompt,
)
from agents.workflow_common import (
    ResearchAgentState,
    ReviewAggregateState,
    ReviewState,
)
from utils.openai_fallback import invoke_with_rate_limit_fallback

logger = logging.getLogger(__name__)


class ReviewContradictionOutput(BaseModel):
    topic: str = Field(default="")
    concern: str = Field(default="")
    severity: str = Field(default="medium")
    related_agents: list[str] = Field(default_factory=list)


class ParallelReviewOutput(BaseModel):
    summary: str = Field(default="")
    cautions: list[str] = Field(default_factory=list)
    contradictions: list[ReviewContradictionOutput] = Field(default_factory=list)


def review_research_state(
    reviewed_agent: str,
    agent_state: ResearchAgentState | None,
    previous_review: ReviewState | None = None,
) -> ReviewState:
    review_count = int((previous_review or {}).get("review_count", 0)) + 1

    if not agent_state:
        payload = {
            "reviewed_agent": reviewed_agent,
            "decision": "rejected",
            "reason": "검토할 조사 결과가 없어 reject 처리했습니다.",
            "review_count": review_count,
        }
        logger.info(
            "[review/%s] decision=%s review_count=%s reason=%s",
            reviewed_agent,
            payload["decision"],
            payload["review_count"],
            payload["reason"],
        )
        return payload

    if agent_state.get("status") != "completed":
        payload = {
            "reviewed_agent": reviewed_agent,
            "decision": "rejected",
            "reason": "조사 상태가 completed가 아니어서 reject 처리했습니다.",
            "review_count": review_count,
        }
        logger.info(
            "[review/%s] decision=%s review_count=%s reason=%s",
            reviewed_agent,
            payload["decision"],
            payload["review_count"],
            payload["reason"],
        )
        return payload

    payload = {
        "reviewed_agent": reviewed_agent,
        "decision": "approved",
        "reason": "스켈레톤 review agent가 기본 정책에 따라 통과시켰습니다.",
        "review_count": review_count,
    }
    logger.info(
        "[review/%s] decision=%s review_count=%s reason=%s",
        reviewed_agent,
        payload["decision"],
        payload["review_count"],
        payload["reason"],
    )
    return payload


def review_parallel_research(
    *,
    investigate_members_state: ResearchAgentState | None,
    agent_product_market_analysis_state: ResearchAgentState | None,
    agent_risk_search_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
) -> ReviewAggregateState:
    agent_states = {
        "investigate_members": investigate_members_state,
        "agent_product_market_analysis": agent_product_market_analysis_state,
        "agent_risk_search": agent_risk_search_state,
        "traction": traction_state,
    }
    agent_statuses = {
        agent_name: (
            agent_state.get("status", "missing") if agent_state else "missing"
        )
        for agent_name, agent_state in agent_states.items()
    }
    serialized_input = json.dumps(
        {
            "agents": {
                agent_name: build_agent_review_payload(agent_state)
                for agent_name, agent_state in agent_states.items()
            }
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        payload = [
            ("system", get_parallel_review_system_prompt()),
            ("user", render_parallel_review_user_prompt(serialized_input)),
        ]
        result = invoke_with_rate_limit_fallback(
            payload=payload,
            primary_factory=lambda: get_chat_model().with_structured_output(
                ParallelReviewOutput,
                method="json_schema",
            ),
            fallback_factory=lambda: get_fallback_chat_model().with_structured_output(
                ParallelReviewOutput,
                method="json_schema",
            ),
            logger=logger,
            operation_name="agent_review.review_parallel_agents",
        )
        contradictions = [
            {
                "topic": item.topic.strip(),
                "concern": item.concern.strip(),
                "severity": item.severity.strip() or "medium",
                "related_agents": [
                    agent_name.strip()
                    for agent_name in item.related_agents
                    if agent_name.strip()
                ],
            }
            for item in result.contradictions
            if item.concern.strip()
        ]
        cautions = [item.strip() for item in result.cautions if item.strip()]
        for contradiction in contradictions:
            if contradiction["concern"] not in cautions:
                cautions.append(str(contradiction["concern"]))

        payload = {
            "status": "completed",
            "summary": (
                result.summary.strip()
                or "review-agent가 병렬 조사 결과를 검토했지만 추가 유의사항을 생성하지 않았습니다."
            ),
            "agent_statuses": agent_statuses,
            "cautions": cautions,
            "contradictions": contradictions,
        }
        logger.info(
            "[review/parallel] status=%s cautions=%s contradictions=%s",
            payload["status"],
            len(payload["cautions"]),
            len(payload["contradictions"]),
        )
        return payload
    except Exception as exc:
        logger.exception("[review/parallel/error] message=%s", exc)
        return {
            "status": "failed",
            "summary": (
                "review-agent가 병렬 조사 결과의 상충 여부를 판단하는 중 오류가 발생했습니다."
            ),
            "agent_statuses": agent_statuses,
            "cautions": [
                "review-agent 실행 오류로 상충 여부를 자동 검토하지 못했습니다."
            ],
            "contradictions": [],
        }


def build_agent_review_payload(
    agent_state: ResearchAgentState | None,
) -> dict[str, Any] | None:
    if not agent_state:
        return None
    return {
        "status": str(agent_state.get("status", "")),
        "summary": str(agent_state.get("summary", "")),
        "findings": [str(item) for item in list(agent_state.get("findings", []) or [])],
        "structured_output": normalize_structured_output(
            agent_state.get("structured_output"),
        ),
    }


def normalize_structured_output(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): normalize_structured_output(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, list):
        return [normalize_structured_output(item) for item in value]
    return value
