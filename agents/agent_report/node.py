from __future__ import annotations

import logging
from typing import Any, cast

from agents.agent_report.output import ReportNodeOutput
from agents.agent_report.service import build_report_state
from agents.workflow_common import EvalState, ResearchAgentState

logger = logging.getLogger(__name__)


def report_node(state: dict[str, Any]) -> ReportNodeOutput:
    selected_company = cast(dict[str, Any] | None, state.get("selected_company"))
    company_id = selected_company.get("company_id") if selected_company else None
    logger.info("[report/start] company_id=%s", company_id or "-")
    payload = build_report_state(
        selected_company=selected_company,
        company_search_summary=str(state.get("company_search_summary", "")),
        selected_company_reason=str(state.get("selected_company_reason", "")),
        investigate_members_state=cast(
            ResearchAgentState | None,
            state.get("investigate_members_state"),
        ),
        agent_product_market_analysis_state=cast(
            ResearchAgentState | None,
            state.get("agent_product_market_analysis_state"),
        ),
        traction_state=cast(ResearchAgentState | None, state.get("traction_state")),
        agent_risk_search_state=cast(
            ResearchAgentState | None,
            state.get("agent_risk_search_state"),
        ),
        eval_state=cast(EvalState | None, state.get("eval_state")),
    )
    return {"report_state": payload}
