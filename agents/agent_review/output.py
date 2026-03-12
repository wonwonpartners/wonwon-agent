from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ReviewState


class InvestigateMembersReviewNodeOutput(TypedDict):
    investigate_members_review: ReviewState


class ProductMarketAnalysisReviewNodeOutput(TypedDict):
    agent_product_market_analysis_review: ReviewState
    
class TractionReviewNodeOutput(TypedDict):
    traction_review: ReviewState


class AgentBReviewNodeOutput(TypedDict):
    agent_b_review: ReviewState


class AgentCReviewNodeOutput(TypedDict):
    agent_c_review: ReviewState
