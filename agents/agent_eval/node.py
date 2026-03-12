from __future__ import annotations

import logging
from typing import Any, cast

from agents.agent_eval.output import EvalNodeOutput
from agents.agent_eval.service import build_eval_state
from agents.workflow_common import ResearchAgentState

logger = logging.getLogger(__name__)


def eval_node(state: dict[str, Any]) -> EvalNodeOutput:
    selected_company = cast(dict[str, Any] | None, state.get("selected_company"))
    company_id = selected_company.get("company_id") if selected_company else None
    logger.info("[eval/start] company_id=%s", company_id or "-")
    payload = build_eval_state(
        selected_company,
        cast(ResearchAgentState | None, state.get("investigate_members_state")),
        cast(ResearchAgentState | None, state.get("traction_state")),
        cast(ResearchAgentState | None, state.get("agent_a_state")),
        cast(ResearchAgentState | None, state.get("agent_b_state")),
        cast(ResearchAgentState | None, state.get("agent_c_state")),
    )
    logger.info(
        "[eval/final] status=%s ready_for_report=%s",
        payload["status"],
        payload["ready_for_report"],
    )
    return {"eval_state": payload}
