from __future__ import annotations

import logging
from typing import Any, cast

from agents.agent_c.output import AgentCNodeOutput
from agents.agent_c.service import AGENT_NAME, run_agent_c
from agents.workflow_common import ResearchAgentState

logger = logging.getLogger(__name__)


def agent_c_node(state: dict[str, Any]) -> AgentCNodeOutput:
    previous_state = cast(ResearchAgentState | None, state.get("agent_c_state"))
    selected_company = cast(dict[str, Any] | None, state.get("selected_company"))
    company_id = selected_company.get("company_id") if selected_company else None
    logger.info(
        "[%s/start] company_id=%s previous_attempt=%s",
        AGENT_NAME,
        company_id or "-",
        int((previous_state or {}).get("attempt_count", 0)),
    )
    payload = run_agent_c(
        selected_company,
        previous_state,
    )
    return {"agent_c_state": payload}
