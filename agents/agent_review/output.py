from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ReviewState


class InvestigateMembersReviewNodeOutput(TypedDict):
    investigate_members_review: ReviewState


class AgentAReviewNodeOutput(TypedDict):
    agent_a_review: ReviewState


class AgentBReviewNodeOutput(TypedDict):
    agent_b_review: ReviewState


class AgentCReviewNodeOutput(TypedDict):
    agent_c_review: ReviewState
