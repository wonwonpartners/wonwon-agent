from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.agent_find_company import find_company_node


class CompanyResearchState(TypedDict, total=False):
    user_query: str
    company_search_summary: str
    company_search_filters: dict[str, Any] | None
    selected_company: dict[str, Any] | None
    selected_company_reason: str
    leadership_research: dict[str, Any] | None


def build_company_research_graph():
    graph = StateGraph(CompanyResearchState)
    graph.add_node("find_company", find_company_node)
    graph.add_edge(START, "find_company")
    graph.add_edge("find_company", END)
    return graph.compile()


def run_company_research(user_query: str) -> CompanyResearchState:
    graph = build_company_research_graph()
    return graph.invoke({"user_query": user_query})
