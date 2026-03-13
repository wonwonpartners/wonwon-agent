from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.agent_risk_search import agent_risk_search_node
from agents.agent_eval import eval_node
from agents.agent_find_company import find_company_node
from agents.agent_investigate_members import investigate_members_node
from agents.agent_product_market_analysis import product_market_analysis_node
from agents.agent_report import report_node
from agents.agent_traction import traction_node
from agents.agent_review import review_node
from agents.workflow_common import EvalState, GraphErrorState, ReportState, ResearchAgentState, ReviewAggregateState

logger = logging.getLogger(__name__)

PARALLEL_RESEARCH_NODES = (
    "investigate_members",
    "product_market_analysis",
    "agent_risk_search",
    "traction",
)


class CompanyResearchState(TypedDict, total=False):
    user_query: str
    company_search_summary: str
    company_search_filters: dict[str, Any] | None
    selected_company: dict[str, Any] | None
    selected_company_reason: str
    leadership_research: dict[str, Any] | None
    investigate_members_state: ResearchAgentState
    agent_product_market_analysis_state: ResearchAgentState
    agent_risk_search_state: ResearchAgentState
    traction_state: ResearchAgentState
    review_state: ReviewAggregateState
    eval_state: EvalState
    report_state: ReportState
    graph_error: GraphErrorState


def route_after_find_company(state: CompanyResearchState) -> str | list[str]:
    if not state.get("selected_company"):
        logger.info(
            "[graph/route] find_company produced no selected_company; ending graph"
        )
        return END
    logger.info(
        "[graph/route] selected_company found; launching parallel_agents=%s",
        ", ".join(PARALLEL_RESEARCH_NODES),
    )
    return list(PARALLEL_RESEARCH_NODES)


def build_company_research_graph():
    graph = StateGraph(CompanyResearchState)
    graph.add_node("find_company", find_company_node)
    graph.add_node("investigate_members", investigate_members_node)
    graph.add_node("product_market_analysis", product_market_analysis_node)
    graph.add_node("agent_risk_search", agent_risk_search_node)
    graph.add_node("traction", traction_node)
    graph.add_node("review_agent", review_node)
    graph.add_node("eval_agent", eval_node)
    graph.add_node("report_agent", report_node)

    graph.add_edge(START, "find_company")
    graph.add_conditional_edges("find_company", route_after_find_company)
    graph.add_edge(
        list(PARALLEL_RESEARCH_NODES),
        "review_agent",
    )
    graph.add_edge("review_agent", "eval_agent")
    graph.add_edge("eval_agent", "report_agent")
    graph.add_edge("report_agent", END)
    return graph.compile()


def run_company_research(user_query: str) -> CompanyResearchState:
    logger.info("[graph/start] user_query=%s", user_query)
    graph = build_company_research_graph()
    result = graph.invoke({"user_query": user_query})
    logger.info(
        "[graph/final] selected_company=%s report_generated=%s graph_error=%s",
        bool(result.get("selected_company")),
        bool((result.get("report_state") or {}).get("report_path")),
        bool(result.get("graph_error")),
    )
    return result
