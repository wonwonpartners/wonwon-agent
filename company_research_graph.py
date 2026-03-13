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
    evaluated_company_ids: list[str]
    candidate_eval_history: list[dict[str, Any]]
    company_retry_count: int
    max_company_retries: int
    retry_find_company_reason: str
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


def route_after_eval(state: CompanyResearchState) -> str:
    eval_state = state.get("eval_state") or {}
    next_action = str(eval_state.get("next_action", "stop"))
    retry_count = int(state.get("company_retry_count", 0))
    max_retries = int(state.get("max_company_retries", 2))

    if next_action == "report":
        logger.info("[graph/route_after_eval] next_action=report")
        return "report_agent"
    if next_action == "retry_find_company" and retry_count < max_retries:
        logger.info(
            "[graph/route_after_eval] next_action=retry_find_company retry=%s/%s",
            retry_count + 1,
            max_retries,
        )
        return "prepare_next_company"
    logger.info(
        "[graph/route_after_eval] next_action=%s retry=%s/%s ending graph",
        next_action,
        retry_count,
        max_retries,
    )
    return END


def prepare_next_company_node(state: CompanyResearchState) -> dict[str, Any]:
    selected_company = state.get("selected_company") or {}
    company_id = str(selected_company.get("company_id", "")).strip()
    company_name = str(selected_company.get("company_name", "알 수 없는 회사")).strip()
    evaluated_company_ids = list(state.get("evaluated_company_ids", []) or [])
    if company_id and company_id not in evaluated_company_ids:
        evaluated_company_ids.append(company_id)

    eval_state = state.get("eval_state") or {}
    history = list(state.get("candidate_eval_history", []) or [])
    history.append(
        {
            "company_id": company_id,
            "company_name": company_name,
            "weighted_score": eval_state.get("weighted_score"),
            "final_decision": eval_state.get("final_decision"),
            "next_action": eval_state.get("next_action"),
            "retry_reason": eval_state.get("retry_reason", ""),
        }
    )
    retry_reason = str(eval_state.get("retry_reason", "") or "")
    logger.info(
        "[graph/prepare_next_company] excluded_company=%s(%s) retry_reason=%s",
        company_name,
        company_id or "-",
        retry_reason or "-",
    )
    return {
        "evaluated_company_ids": evaluated_company_ids,
        "candidate_eval_history": history,
        "company_retry_count": int(state.get("company_retry_count", 0)) + 1,
        "retry_find_company_reason": retry_reason,
        "company_search_summary": "",
        "company_search_filters": None,
        "selected_company": None,
        "selected_company_reason": "",
        "leadership_research": None,
        "investigate_members_state": None,
        "agent_product_market_analysis_state": None,
        "agent_risk_search_state": None,
        "traction_state": None,
        "review_state": None,
        "eval_state": None,
        "report_state": None,
    }


def build_company_research_graph():
    graph = StateGraph(CompanyResearchState)
    graph.add_node("find_company", find_company_node)
    graph.add_node("investigate_members", investigate_members_node)
    graph.add_node("product_market_analysis", product_market_analysis_node)
    graph.add_node("agent_risk_search", agent_risk_search_node)
    graph.add_node("traction", traction_node)
    graph.add_node("review_agent", review_node)
    graph.add_node("eval_agent", eval_node)
    graph.add_node("prepare_next_company", prepare_next_company_node)
    graph.add_node("report_agent", report_node)

    graph.add_edge(START, "find_company")
    graph.add_conditional_edges("find_company", route_after_find_company)
    graph.add_edge(
        list(PARALLEL_RESEARCH_NODES),
        "review_agent",
    )
    graph.add_edge("review_agent", "eval_agent")
    graph.add_conditional_edges("eval_agent", route_after_eval)
    graph.add_edge("prepare_next_company", "find_company")
    graph.add_edge("report_agent", END)
    return graph.compile()


def run_company_research(user_query: str) -> CompanyResearchState:
    logger.info("[graph/start] user_query=%s", user_query)
    graph = build_company_research_graph()
    result = graph.invoke(
        {
            "user_query": user_query,
            "evaluated_company_ids": [],
            "candidate_eval_history": [],
            "company_retry_count": 0,
            "max_company_retries": 2,
            "retry_find_company_reason": "",
        }
    )
    logger.info(
        "[graph/final] selected_company=%s report_generated=%s graph_error=%s",
        bool(result.get("selected_company")),
        bool((result.get("report_state") or {}).get("report_path")),
        bool(result.get("graph_error")),
    )
    return result
