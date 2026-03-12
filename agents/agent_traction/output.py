from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class TractionNodeOutput(TypedDict):
    traction_state: ResearchAgentState
