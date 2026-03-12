from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class ProductMarketAnalysisNodeOutput(TypedDict):
    agent_product_market_analysis_state: ResearchAgentState
