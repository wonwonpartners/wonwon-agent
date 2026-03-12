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
    traction_state: ResearchAgentState | None,
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
        traction_state=traction_state,
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
    traction_state: ResearchAgentState | None,
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
        render_agent_section("traction", traction_state),
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

    if title == "investigate_members" and agent_state.get("structured_output"):
        return render_investigate_members_section(agent_state)
    if title == "traction" and agent_state.get("structured_output"):
        return render_traction_section(agent_state)

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


def render_traction_section(agent_state: ResearchAgentState) -> str:
    payload = agent_state.get("structured_output") or {}
    partnerships = payload.get("partnerships") or []
    hiring = payload.get("hiring_analysis") or {}
    funding_velocity = payload.get("funding_velocity") or []
    sources = agent_state.get("sources") or []

    lines = [
        "## traction",
        f"- 상태: {agent_state.get('status', 'unknown')}",
        f"- 시도 횟수: {agent_state.get('attempt_count', 0)}",
        f"- 요약: {agent_state.get('summary', '')}",
        "",
        "### 파트너십",
    ]

    if partnerships:
        lines.extend(f"- {item}" for item in partnerships)
    else:
        lines.append("- 없음")

    lines.extend(
        [
            "",
            "### 채용 분석",
            f"- field_engineer_ratio: {hiring.get('field_engineer_ratio', 0)}",
            f"- field_engineer_count: {hiring.get('field_engineer_count', 0)}",
            f"- hiring_trend_3m: {hiring.get('hiring_trend_3m', 0)}",
            "",
            "### 투자/성장 신호",
        ]
    )

    if funding_velocity:
        lines.extend(f"- {item}" for item in funding_velocity)
    else:
        lines.append("- 없음")

    lines.extend(["", "### sources"])
    if sources:
        lines.extend(f"- {json.dumps(source, ensure_ascii=False)}" for source in sources)
    else:
        lines.append("- 없음")

    return "\n".join(lines)


def render_investigate_members_section(agent_state: ResearchAgentState) -> str:
    payload = agent_state.get("structured_output") or {}
    ceo = payload.get("ceo")
    key_members = payload.get("key_members") or []
    role_coverage = payload.get("role_coverage") or {}
    strengths = payload.get("strengths") or []
    evidence_gaps = payload.get("evidence_gaps") or []
    search_queries = payload.get("search_queries") or []
    sources = agent_state.get("sources") or []

    lines = [
        "## investigate_members",
        f"- 상태: {agent_state.get('status', 'unknown')}",
        f"- 시도 횟수: {agent_state.get('attempt_count', 0)}",
        f"- 요약: {agent_state.get('summary', '')}",
        f"- 평가 요약: {payload.get('assessment_summary', '')}",
        f"- 근거 품질: {payload.get('evidence_quality', '')}",
        "",
        "### CEO",
    ]

    if ceo:
        lines.extend(
            [
                f"- 이름: {ceo.get('name', '')}",
                f"- 역할: {ceo.get('current_role', '')}",
                f"- 창업자 여부: {ceo.get('is_founder', False)}",
                f"- 경험 태그: {', '.join(ceo.get('experience_tags', [])) or '없음'}",
                f"- 근거 요약: {ceo.get('evidence_summary', '')}",
                f"- source_ids: {', '.join(ceo.get('source_ids', [])) or '없음'}",
                f"- confidence: {ceo.get('confidence', 0)}",
            ]
        )
    else:
        lines.append("- 확인된 CEO/대표 근거가 없습니다.")

    lines.extend(["", "### 핵심팀"])
    if key_members:
        for member in key_members:
            lines.extend(
                [
                    f"- {member.get('name', '')} | {member.get('current_role', '')}",
                    f"  - 창업자 여부: {member.get('is_founder', False)}",
                    f"  - 경험 태그: {', '.join(member.get('experience_tags', [])) or '없음'}",
                    f"  - 근거 요약: {member.get('evidence_summary', '')}",
                    f"  - source_ids: {', '.join(member.get('source_ids', [])) or '없음'}",
                    f"  - confidence: {member.get('confidence', 0)}",
                ]
            )
    else:
        lines.append("- 확인된 핵심팀이 없습니다.")

    lines.extend(["", "### 역할 커버리지"])
    for key, value in role_coverage.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "### 강점"])
    if strengths:
        lines.extend(f"- {item}" for item in strengths)
    else:
        lines.append("- 없음")

    lines.extend(["", "### 근거 부족"])
    if evidence_gaps:
        lines.extend(f"- {item}" for item in evidence_gaps)
    else:
        lines.append("- 없음")

    lines.extend(["", "### 검색 쿼리"])
    if search_queries:
        lines.extend(f"- {query}" for query in search_queries)
    else:
        lines.append("- 없음")

    lines.extend(["", "### sources"])
    if sources:
        lines.extend(f"- {json.dumps(source, ensure_ascii=False)}" for source in sources)
    else:
        lines.append("- 없음")

    return "\n".join(lines)
