from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from agents.agent_investigate_members.common import get_chat_model
from agents.agent_traction.output import TractionNodeOutput
from agents.agent_traction.service import TractionAgent
from agents.workflow_common import ResearchAgentState, get_company_id, get_company_name

logger = logging.getLogger(__name__)

AGENT_NAME = "traction"


def traction_node(state: dict[str, Any]) -> TractionNodeOutput:
    previous_state = cast(ResearchAgentState | None, state.get("traction_state"))
    selected_company = cast(dict[str, Any] | None, state.get("selected_company"))
    company_id = get_company_id(selected_company)
    company_name = get_company_name(selected_company)
    attempt_count = int((previous_state or {}).get("attempt_count", 0)) + 1

    logger.info(
        "[%s/start] company_id=%s previous_attempt=%s",
        AGENT_NAME,
        company_id or "-",
        int((previous_state or {}).get("attempt_count", 0)),
    )

    if company_id is None:
        return {
            "traction_state": {
                "agent_name": AGENT_NAME,
                "status": "skipped",
                "attempt_count": attempt_count,
                "input_company_id": None,
                "summary": "선정된 회사가 없어 traction 조사를 진행하지 못했습니다.",
                "findings": [
                    "선행 단계에서 `selected_company`가 비어 있어 traction 조사를 건너뛰었습니다.",
                ],
                "sources": [],
                "structured_output": None,
            }
        }

    try:
        agent = TractionAgent(llm=get_chat_model())
        result = asyncio.run(agent({"startup_name": company_name}))
    except Exception as exc:
        logger.exception(
            "[%s/error] company=%s(%s) message=%s",
            AGENT_NAME,
            company_name,
            company_id,
            exc,
        )
        return {
            "traction_state": {
                "agent_name": AGENT_NAME,
                "status": "failed",
                "attempt_count": attempt_count,
                "input_company_id": company_id,
                "summary": (
                    f"{company_name} ({company_id})의 traction 조사를 실행하는 중 오류가 발생했습니다."
                ),
                "findings": [
                    "traction 검색 또는 LLM structured extraction 단계에서 오류가 발생했습니다.",
                    f"오류 메시지: {str(exc)}",
                ],
                "sources": [
                    {
                        "source_type": "selected_company",
                        "company_id": company_id,
                        "company_name": company_name,
                    }
                ],
                "structured_output": None,
            }
        }

    traction_payload = result.get("traction")
    if not isinstance(traction_payload, dict):
        return {
            "traction_state": {
                "agent_name": AGENT_NAME,
                "status": "failed",
                "attempt_count": attempt_count,
                "input_company_id": company_id,
                "summary": f"{company_name} ({company_id})에 대한 traction 결과를 생성하지 못했습니다.",
                "findings": [
                    "TractionAgent가 `traction` payload를 반환하지 않았습니다.",
                    "검색 품질 부족 또는 LLM 응답 정규화 실패 가능성이 있습니다.",
                ],
                "sources": [
                    {
                        "source_type": "selected_company",
                        "company_id": company_id,
                        "company_name": company_name,
                    }
                ],
                "structured_output": None,
            }
        }

    partnerships = traction_payload.get("partnerships") or []
    hiring_analysis = traction_payload.get("hiring_analysis") or {}
    funding_velocity = traction_payload.get("funding_velocity") or []

    ratio = hiring_analysis.get("field_engineer_ratio", 0.0)
    field_count = hiring_analysis.get("field_engineer_count", 0)
    hiring_trend = hiring_analysis.get("hiring_trend_3m", 0)

    findings = [
        (
            "파트너십 신호: "
            + (", ".join(str(item).strip() for item in partnerships[:3] if str(item).strip()) or "확인 불가")
        ),
        (
            f"채용 신호: Field Engineer 비중 {float(ratio):.1%}, "
            f"공고 수 {int(field_count)}건, 최근 3개월 트렌드 {int(hiring_trend)}"
        ),
        (
            "투자/성장 신호: "
            + (", ".join(str(item).strip() for item in funding_velocity[:3] if str(item).strip()) or "확인 불가")
        ),
    ]

    return {
        "traction_state": {
            "agent_name": AGENT_NAME,
            "status": "completed",
            "attempt_count": attempt_count,
            "input_company_id": company_id,
            "summary": str(traction_payload.get("traction_summary", "")).strip()
            or f"{company_name} ({company_id})의 traction 조사를 완료했습니다.",
            "findings": findings,
            "sources": [
                {
                    "source_type": "selected_company",
                    "company_id": company_id,
                    "company_name": company_name,
                }
            ],
            "structured_output": traction_payload,
        }
    }
