from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Annotated, Dict, List, Optional, TypedDict

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from product_market_tools import PRODUCT_MARKET_TOOLS

if TYPE_CHECKING:
    from langgraph.graph.state import StateGraph


class TeamAnalysisState(TypedDict):
    founder_profile: str
    core_expertise: List[Dict[str, str]]
    team_summary: str


class ProductMarketState(TypedDict):
    target_kpi_logic: str
    technical_moat: str
    data_loop_structure: str
    product_summary: str


class TractionMomentumState(TypedDict):
    partnerships: List[str]
    hiring_analysis: Dict[str, float]
    funding_velocity: str
    traction_summary: str


class RiskDetectionState(TypedDict):
    legal_regulatory: str
    certification_status: List[str]
    red_flags: List[str]
    risk_summary: str


class InvestmentGraphState(TypedDict):
    startup_name: Annotated[str, operator.setitem]
    team: Annotated[Optional[TeamAnalysisState], operator.setitem]
    product_market: Annotated[Optional[ProductMarketState], operator.setitem]
    traction: Annotated[Optional[TractionMomentumState], operator.setitem]
    risk: Annotated[Optional[RiskDetectionState], operator.setitem]
    evaluation_results: Annotated[Optional[Dict[str, Any]], operator.setitem]
    final_report: Annotated[Optional[str], operator.setitem]


PRODUCT_MARKET_ANALYSIS_NODE_NAME = "product_market_analysis"


PRODUCT_MARKET_RESEARCH_PROMPT = """
You are the product and market analysis agent for startup investment research.

Your goal is to gather evidence and assess the startup's product-market logic.
Use the available tools aggressively before making conclusions.

Tool usage rules:
- Use rag_search_tool with corpus='company' for official homepage, product page, whitepaper, and PR materials.
- Use rag_search_tool with corpus='domain' for papers, reports, market data, and benchmark references.
- Use web_benchmark_search_tool for competitors, comparable services, and recent external comparisons.

Research goals:
1. Identify the customer pain point and KPI / ROI logic.
2. Assess the technical moat versus competitors or alternatives.
3. Assess the AI autonomy, data loop, and fallback structure.

Important rules:
- Do not treat company-authored claims as verified facts.
- Compare company claims against external sources when possible.
- If evidence is weak or missing, say so clearly.
- Keep final findings concise and evidence-oriented.
""".strip()


PRODUCT_MARKET_SYNTHESIS_PROMPT = """
You are writing ProductMarketState for an investment evaluation graph.

Write the final answer in Korean and return structured output that matches ProductMarketState exactly.

Field requirements:
- target_kpi_logic: Explain the customer pain point, target KPI, and ROI logic. Distinguish company claims from externally supported conclusions. End the field with a `출처:` section listing only the references used for this field.
- technical_moat: Judge whether there is a durable technical moat, what type of moat it is, and how it compares with similar services. End the field with a `출처:` section listing only the references used for this field.
- data_loop_structure: Explain where AI is used, whether operational data creates a learning loop, and whether fallback is credible. End the field with a `출처:` section listing only the references used for this field.
- product_summary: Give a concise investment-style synthesis of the above three fields. End the field with a `출처:` section listing only the references used for this field.

Important rules:
- Be judgment-oriented, not source-summary oriented.
- If evidence is insufficient, make the uncertainty explicit.
- Do not invent facts that are not grounded in the research notes.
- In each field, format the citation block like:
  출처:
  - 한국은행(2024). 금융안정보고서. https://...
  - 김철수(2024). 인공지능 산업 전망. 투자연구, 10(2), 50-60.
  - IEA(2024-04-15). Global EV Outlook 2024. IEA. https://...
- If exact author, year, publisher, journal, or date is unavailable, keep the citation conservative and use only fields grounded in the research notes.
- Do not list sources that were not actually used for that specific field.
""".strip()


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _collect_research_notes(messages: list[Any]) -> str:
    notes: list[str] = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = _stringify_message_content(getattr(message, "content", ""))
        if content:
            notes.append(f"[{role}]\n{content}")
    return "\n\n".join(notes)


def _build_research_agent():
    research_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    return create_agent(
        model=research_llm,
        tools=PRODUCT_MARKET_TOOLS,
        system_prompt=PRODUCT_MARKET_RESEARCH_PROMPT,
    )


def _build_writer_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(
        ProductMarketState
    )


async def product_market_analysis_agent(
    state: InvestmentGraphState,
) -> dict[str, ProductMarketState]:
    startup_name = state["startup_name"]
    research_agent = _build_research_agent()
    writer_llm = _build_writer_llm()

    research_request = f"""
Analyze the startup named {startup_name}.

You must gather evidence to assess:
1. target KPI logic and ROI
2. technical moat
3. AI autonomy, data loop, and fallback strategy

Use both company and domain RAG sources when available.
Use web benchmark search when competitor or comparable-service context is needed.
At the end, provide concise research notes with clear evidence and uncertainty.
""".strip()

    research_result = await research_agent.ainvoke(
        {"messages": [("user", research_request)]}
    )
    research_notes = _collect_research_notes(research_result["messages"])

    product_market = await writer_llm.ainvoke(
        [
            ("system", PRODUCT_MARKET_SYNTHESIS_PROMPT),
            (
                "user",
                f"startup_name: {startup_name}\n\nresearch_notes:\n{research_notes}",
            ),
        ]
    )

    return {"product_market": product_market}


def register_product_market_analysis_node(builder: "StateGraph") -> "StateGraph":
    builder.add_node(PRODUCT_MARKET_ANALYSIS_NODE_NAME, product_market_analysis_agent)
    return builder
