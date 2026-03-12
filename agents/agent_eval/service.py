from __future__ import annotations

from typing import Any

from agents.workflow_common import EvalState, ResearchAgentState, get_company_id, get_company_name


def build_eval_state(
    selected_company: dict[str, Any] | None,
    investigate_members_state: ResearchAgentState | None,
    traction_state: ResearchAgentState | None,
    agent_a_state: ResearchAgentState | None,
    agent_b_state: ResearchAgentState | None,
    agent_c_state: ResearchAgentState | None,
) -> EvalState:
    company_name = get_company_name(selected_company)
    company_id = get_company_id(selected_company) or "알 수 없음"
    agent_states = {
        "investigate_members": investigate_members_state,
        "traction": traction_state,
        "agent_a": agent_a_state,
        "agent_b": agent_b_state,
        "agent_c": agent_c_state,
    }
    agent_summaries = {
        agent_name: (agent_state.get("summary") if agent_state else "결과가 없습니다.")
        for agent_name, agent_state in agent_states.items()
    }
    ready_for_report = all(
        agent_state is not None and agent_state.get("status") == "completed"
        for agent_state in agent_states.values()
    )
    status = "completed" if ready_for_report else "blocked"
    return {
        "status": status,
        "ready_for_report": ready_for_report,
        "summary": (
            f"{company_name} ({company_id})에 대한 5개 병렬 조사 결과를 묶어 "
            f"보고서 작성 가능 여부를 {status}로 정리했습니다."
        ),
        "agent_summaries": agent_summaries,
    }
