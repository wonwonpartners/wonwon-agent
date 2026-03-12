from __future__ import annotations

import logging
from typing import Any, cast

from agents.agent_product_market_analysis.output import (
    ProductMarketAnalysisNodeOutput,
)
from agents.agent_product_market_analysis.service import (
    AGENT_NAME,
    run_product_market_analysis,
)
from agents.workflow_common import ResearchAgentState

logger = logging.getLogger(__name__)


def product_market_analysis_node(
    state: dict[str, Any],
) -> ProductMarketAnalysisNodeOutput:
    previous_state = cast(
        ResearchAgentState | None,
        state.get("agent_product_market_analysis_state"),
    )
    selected_company = cast(dict[str, Any] | None, state.get("selected_company"))
    company_id = selected_company.get("company_id") if selected_company else None
    logger.info(
        "[%s/start] company_id=%s previous_attempt=%s",
        AGENT_NAME,
        company_id or "-",
        int((previous_state or {}).get("attempt_count", 0)),
    )
    payload = run_product_market_analysis(
        selected_company,
        previous_state,
    )
    return {"agent_product_market_analysis_state": payload}
