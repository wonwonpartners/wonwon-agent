from __future__ import annotations

import logging

from agents.workflow_common import ResearchAgentState, ReviewState

logger = logging.getLogger(__name__)


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
