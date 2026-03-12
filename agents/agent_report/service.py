from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.agent_report.common import REPORTS_ROOT
from agents.workflow_common import EvalState, ReportState, ResearchAgentState, get_company_id, get_company_name

logger = logging.getLogger(__name__)


def build_report_state(
    *,
    selected_company: dict[str, Any] | None,
    company_search_summary: str,
    selected_company_reason: str,
    investigate_members_state: ResearchAgentState | None,
    agent_a_state: ResearchAgentState | None,
    agent_b_state: ResearchAgentState | None,
    agent_c_state: ResearchAgentState | None,
    eval_state: EvalState | None,
) -> ReportState:
    company_id = get_company_id(selected_company)
    if company_id is None or not eval_state or not eval_state.get("ready_for_report"):
        payload = {
            "status": "skipped",
            "report_path": "",
            "markdown": "",
        }
        logger.info(
            "[report/final] status=%s company_id=%s ready_for_report=%s",
            payload["status"],
            company_id or "-",
            bool(eval_state and eval_state.get("ready_for_report")),
        )
        return payload

    report_path = build_report_path(company_id)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    markdown = render_report_markdown(
        selected_company=selected_company,
        company_search_summary=company_search_summary,
        selected_company_reason=selected_company_reason,
        investigate_members_state=investigate_members_state,
        agent_a_state=agent_a_state,
        agent_b_state=agent_b_state,
        agent_c_state=agent_c_state,
        eval_state=eval_state,
        generated_at=generated_at,
        report_path=report_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    payload = {
        "status": "completed",
        "report_path": str(report_path),
        "markdown": markdown,
    }
    logger.info(
        "[report/final] status=%s company_id=%s path=%s",
        payload["status"],
        company_id,
        payload["report_path"],
    )
    return payload


def build_report_path(company_id: str) -> Path:
    return (REPORTS_ROOT / f"{company_id}.md").resolve()


def render_report_markdown(
    *,
    selected_company: dict[str, Any] | None,
    company_search_summary: str,
    selected_company_reason: str,
    investigate_members_state: ResearchAgentState | None,
    agent_a_state: ResearchAgentState | None,
    agent_b_state: ResearchAgentState | None,
    agent_c_state: ResearchAgentState | None,
    eval_state: EvalState,
    generated_at: str,
    report_path: Path,
) -> str:
    company_name = get_company_name(selected_company)
    company_id = get_company_id(selected_company) or "알 수 없음"
    sections = [
        f"# {company_name} 투자 조사 보고서",
        "",
        "## 회사 선정 결과",
        f"- 회사명: {company_name}",
        f"- 회사 ID: {company_id}",
        f"- 검색 요약: {company_search_summary or '없음'}",
        f"- 선정 사유: {selected_company_reason or '없음'}",
        "",
        render_agent_section("investigate_members", investigate_members_state),
        "",
        render_agent_section("agent_a", agent_a_state),
        "",
        render_agent_section("agent_b", agent_b_state),
        "",
        render_agent_section("agent_c", agent_c_state),
        "",
        "## eval 요약",
        f"- 상태: {eval_state.get('status', 'unknown')}",
        f"- 보고서 준비 여부: {eval_state.get('ready_for_report', False)}",
        f"- 요약: {eval_state.get('summary', '')}",
        "",
        "## 생성 메타정보",
        f"- 생성 시각: {generated_at}",
        f"- 저장 경로: {report_path}",
    ]
    return "\n".join(sections).strip() + "\n"


def render_agent_section(
    title: str,
    agent_state: ResearchAgentState | None,
) -> str:
    if not agent_state:
        return "\n".join(
            [
                f"## {title}",
                "- 상태: missing",
                "- 요약: 결과가 없습니다.",
            ]
        )

    findings = agent_state.get("findings") or []
    sources = agent_state.get("sources") or []
    lines = [
        f"## {title}",
        f"- 상태: {agent_state.get('status', 'unknown')}",
        f"- 시도 횟수: {agent_state.get('attempt_count', 0)}",
        f"- 요약: {agent_state.get('summary', '')}",
        "- findings:",
    ]
    if findings:
        lines.extend(f"  - {finding}" for finding in findings)
    else:
        lines.append("  - 없음")

    lines.append("- sources:")
    if sources:
        lines.extend(f"  - {json.dumps(source, ensure_ascii=False)}" for source in sources)
    else:
        lines.append("  - 없음")

    return "\n".join(lines)
