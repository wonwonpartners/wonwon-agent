from __future__ import annotations

from typing import Any

from agents.workflow_common import ResearchAgentState, build_skeleton_research_state


AGENT_NAME = "investigate_members"


def run_investigate_members(
    selected_company: dict[str, Any] | None,
    previous_state: ResearchAgentState | None = None,
) -> ResearchAgentState:
    return build_skeleton_research_state(
        AGENT_NAME,
        selected_company,
        previous_state,
    )
