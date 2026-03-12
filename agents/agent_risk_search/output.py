from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class AgentRiskSearchNodeOutput(TypedDict):
    agent_risk_search_state: ResearchAgentState
