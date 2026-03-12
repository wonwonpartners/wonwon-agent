from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class AgentANodeOutput(TypedDict):
    agent_a_state: ResearchAgentState
