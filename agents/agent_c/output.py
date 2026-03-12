from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class AgentCNodeOutput(TypedDict):
    agent_c_state: ResearchAgentState
