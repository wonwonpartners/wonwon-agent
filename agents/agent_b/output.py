from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class AgentBNodeOutput(TypedDict):
    agent_b_state: ResearchAgentState
