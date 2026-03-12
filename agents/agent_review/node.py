from __future__ import annotations

import logging
from typing import Any, cast

from agents.agent_review.output import (
    AgentAReviewNodeOutput,
    AgentBReviewNodeOutput,
    AgentCReviewNodeOutput,
    InvestigateMembersReviewNodeOutput,
)
from agents.agent_review.service import review_research_state
from agents.workflow_common import ResearchAgentState, ReviewState

logger = logging.getLogger(__name__)


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


def review_agent_a_node(state: dict[str, Any]) -> AgentAReviewNodeOutput:
    return cast(
        AgentAReviewNodeOutput,
        _review_node(
            state,
            agent_key="agent_a_state",
            review_key="agent_a_review",
            agent_name="agent_a",
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
