from __future__ import annotations

import logging
from typing import Any, cast

from agents.agent_investigate_members.output import InvestigateMembersNodeOutput
from agents.agent_investigate_members.service import AGENT_NAME, run_investigate_members
from agents.workflow_common import ResearchAgentState

logger = logging.getLogger(__name__)


def investigate_members_node(state: dict[str, Any]) -> InvestigateMembersNodeOutput:
    previous_state = cast(
        ResearchAgentState | None,
        state.get("investigate_members_state"),
    )
    selected_company = cast(dict[str, Any] | None, state.get("selected_company"))
    company_id = selected_company.get("company_id") if selected_company else None
    logger.info(
        "[%s/start] company_id=%s previous_attempt=%s",
        AGENT_NAME,
        company_id or "-",
        int((previous_state or {}).get("attempt_count", 0)),
    )
    payload = run_investigate_members(
        selected_company,
        previous_state,
    )
    return {
        "investigate_members_state": payload,
        "leadership_research": payload.get("structured_output"),
    }
