from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.agent_risk_search import agent_risk_search_node
from agents.agent_c import agent_c_node
from agents.agent_eval import eval_node
from agents.agent_find_company import find_company_node
from agents.agent_investigate_members import investigate_members_node
from agents.agent_product_market_analysis import product_market_analysis_node
from agents.agent_report import report_node
from agents.agent_traction import traction_node
from agents.agent_review import (
    review_agent_c_node,
    review_agent_risk_search_node,
    review_agent_product_market_analysis_node,
    review_investigate_members_node,
    review_traction_node,
)
from agents.workflow_common import EvalState, GraphErrorState, ReportState, ResearchAgentState, ReviewState

logger = logging.getLogger(__name__)


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
    agent_c_state: ResearchAgentState
    investigate_members_review: ReviewState
    agent_product_market_analysis_review: ReviewState
    traction_state: ResearchAgentState
    traction_review: ReviewState
    agent_risk_search_review: ReviewState
    agent_c_review: ReviewState
    eval_state: EvalState
    report_state: ReportState
    graph_error: GraphErrorState


@dataclass(frozen=True)
class ReviewBranchConfig:
    agent_name: str
    agent_node_name: str
    review_node_name: str
    approved_node_name: str
    error_node_name: str
    review_state_key: str


REVIEW_BRANCHES = (
    ReviewBranchConfig(
        agent_name="investigate_members",
        agent_node_name="investigate_members",
        review_node_name="review_investigate_members",
        approved_node_name="investigate_members_approved",
        error_node_name="investigate_members_error",
        review_state_key="investigate_members_review",
    ),
    ReviewBranchConfig(
        agent_name="agent_product_market_analysis",
        agent_node_name="agent_product_market_analysis",
        review_node_name="review_agent_product_market_analysis",
        approved_node_name="agent_product_market_analysis_approved",
        error_node_name="agent_product_market_analysis_error",
        review_state_key="agent_product_market_analysis_review",
    ),
    ReviewBranchConfig(
        agent_name="traction",
        agent_node_name="traction",
        review_node_name="review_traction",
        approved_node_name="traction_approved",
        error_node_name="traction_error",
        review_state_key="traction_review",
    ),
    ReviewBranchConfig(
        agent_name="agent_risk_search",
        agent_node_name="agent_risk_search",
        review_node_name="review_agent_risk_search",
        approved_node_name="agent_risk_search_approved",
        error_node_name="agent_risk_search_error",
        review_state_key="agent_risk_search_review",
    ),
    ReviewBranchConfig(
        agent_name="agent_c",
        agent_node_name="agent_c",
        review_node_name="review_agent_c",
        approved_node_name="agent_c_approved",
        error_node_name="agent_c_error",
        review_state_key="agent_c_review",
    ),
)


def route_after_find_company(state: CompanyResearchState) -> str | list[str]:
    if not state.get("selected_company"):
        logger.info(
            "[graph/route] find_company produced no selected_company; ending graph"
        )
        return END
    logger.info(
        "[graph/route] selected_company found; launching parallel_agents=%s",
        ", ".join(branch.agent_node_name for branch in REVIEW_BRANCHES),
    )
    return [branch.agent_node_name for branch in REVIEW_BRANCHES]


def make_review_route(branch: ReviewBranchConfig) -> Callable[[CompanyResearchState], str]:
    def route(state: CompanyResearchState) -> str:
        review_state = state.get(branch.review_state_key)
        if not isinstance(review_state, dict):
            logger.warning(
                "[graph/review_route] agent=%s review_state_missing next=%s",
                branch.agent_name,
                branch.error_node_name,
            )
            return branch.error_node_name

        decision = review_state.get("decision")
        review_count = int(review_state.get("review_count", 0))
        if decision == "approved":
            logger.info(
                "[graph/review_route] agent=%s decision=%s review_count=%s next=%s",
                branch.agent_name,
                decision,
                review_count,
                branch.approved_node_name,
            )
            return branch.approved_node_name
        if decision == "rejected" and review_count <= 1:
            logger.info(
                "[graph/review_route] agent=%s decision=%s review_count=%s next=%s",
                branch.agent_name,
                decision,
                review_count,
                branch.agent_node_name,
            )
            return branch.agent_node_name
        logger.warning(
            "[graph/review_route] agent=%s decision=%s review_count=%s next=%s",
            branch.agent_name,
            decision,
            review_count,
            branch.error_node_name,
        )
        return branch.error_node_name

    return route


def make_approval_node(agent_name: str) -> Callable[[CompanyResearchState], dict[str, Any]]:
    def approval_node(_: CompanyResearchState) -> dict[str, Any]:
        logger.info(
            "[graph/approved] agent=%s review approved; waiting for remaining branches",
            agent_name,
        )
        return {}

    approval_node.__name__ = f"{agent_name}_approved_node"
    return approval_node


def make_error_node(
    agent_name: str,
    review_state_key: str,
) -> Callable[[CompanyResearchState], Command[str]]:
    def error_node(state: CompanyResearchState) -> Command[str]:
        existing_error = state.get("graph_error")
        if existing_error:
            logger.warning(
                "[graph/error] agent=%s existing_error_present; ending graph",
                agent_name,
            )
            return Command(goto=END)

        review_state = state.get(review_state_key) or {}
        reason = str(
            review_state.get(
                "reason",
                "review agent가 2회 연속 reject 하여 그래프를 종료했습니다.",
            )
        )
        logger.warning(
            "[graph/error] agent=%s stage=review message=%s",
            agent_name,
            reason,
        )
        return Command(
            update={
                "graph_error": {
                    "stage": "review",
                    "agent_name": agent_name,
                    "message": reason,
                }
            },
            goto=END,
        )

    error_node.__name__ = f"{agent_name}_error_node"
    return error_node


def build_company_research_graph():
    graph = StateGraph(CompanyResearchState)
    graph.add_node("find_company", find_company_node)
    graph.add_node("investigate_members", investigate_members_node)
    graph.add_node("agent_product_market_analysis", product_market_analysis_node)
    graph.add_node("agent_risk_search", agent_risk_search_node)
    graph.add_node("agent_c", agent_c_node)
    graph.add_node("review_investigate_members", review_investigate_members_node)
    graph.add_node(
        "review_agent_product_market_analysis",
        review_agent_product_market_analysis_node,
    )
    graph.add_node("traction", traction_node)
    graph.add_node("review_traction", review_traction_node)
    graph.add_node("review_agent_risk_search", review_agent_risk_search_node)
    graph.add_node("review_agent_c", review_agent_c_node)
    graph.add_node("eval", eval_node)
    graph.add_node("report", report_node)

    for branch in REVIEW_BRANCHES:
        graph.add_node(
            branch.approved_node_name,
            make_approval_node(branch.agent_name),
        )
        graph.add_node(
            branch.error_node_name,
            make_error_node(branch.agent_name, branch.review_state_key),
        )
        graph.add_edge(branch.agent_node_name, branch.review_node_name)
        graph.add_conditional_edges(
            branch.review_node_name,
            make_review_route(branch),
        )

    graph.add_edge(START, "find_company")
    graph.add_conditional_edges("find_company", route_after_find_company)
    graph.add_edge(
        [branch.approved_node_name for branch in REVIEW_BRANCHES],
        "eval",
    )
    graph.add_edge("eval", "report")
    graph.add_edge("report", END)
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
