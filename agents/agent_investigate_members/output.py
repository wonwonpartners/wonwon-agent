from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class InvestigateMembersNodeOutput(TypedDict):
    investigate_members_state: ResearchAgentState
