from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from agents.agent_eval.service import (
    EvalCriterionScoreOutput,
    EvalDecisionOutput,
    build_eval_state,
)


def build_agent_state(
    *,
    agent_name: str,
    summary: str,
    structured_output: dict[str, object],
    status: str = "completed",
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "status": status,
        "attempt_count": 1,
        "input_company_id": "CP_EVAL",
        "summary": summary,
        "findings": [],
        "sources": [],
        "structured_output": structured_output,
    }


class EvalServiceTests(unittest.TestCase):
    def test_build_eval_state_uses_llm_and_preserves_review_contradictions(self) -> None:
        structured_llm = Mock(
            return_value=EvalDecisionOutput(
                final_decision="watch",
                summary="팀과 traction은 강하지만 제품/시장 검증과 규제 대응 추가 확인이 필요합니다.",
                criteria_scores=[
                    EvalCriterionScoreOutput(
                        criterion_id="C1",
                        criterion_name="창업자 및 핵심팀 신뢰도",
                        score=4,
                        rationale="핵심팀 기본 축이 확인됩니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C2",
                        criterion_name="시장 문제 및 도입 논리 명확성",
                        score=3,
                        rationale="도입 논리는 있으나 추가 근거가 필요합니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C3",
                        criterion_name="제품 완성도 및 시스템 차별성",
                        score=3,
                        rationale="실증 근거는 있으나 차별성 판단은 보수적입니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C4",
                        criterion_name="상용화 진전도 및 시장 검증",
                        score=4,
                        rationale="파트너십과 투자 흐름이 관측됩니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C5",
                        criterion_name="AI 자율성 및 데이터 운영 우위",
                        score=3,
                        rationale="데이터 루프 설명은 제한적입니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C6",
                        criterion_name="공개 리스크 및 안전·규제 대응",
                        score=4,
                        rationale="중대한 부정 신호는 제한적입니다.",
                    ),
                ],
                key_strengths=["팀과 상용화 신호가 좋습니다."],
                key_risks=["제품/시장 근거 보강이 필요합니다."],
            )
        )
        model = Mock()
        model.with_structured_output.return_value.invoke = structured_llm

        review_contradictions = [
            {
                "topic": "실행 리스크",
                "concern": "traction은 강하지만 핵심팀 증거는 제한적입니다.",
                "severity": "medium",
                "related_agents": ["investigate_members", "traction"],
            }
        ]

        with patch("agents.agent_eval.service.get_chat_model", return_value=model):
            result = build_eval_state(
                selected_company={
                    "company_id": "CP_EVAL",
                    "company_name": "테스트컴퍼니",
                },
                investigate_members_state=build_agent_state(
                    agent_name="investigate_members",
                    summary="핵심팀이 확인됩니다.",
                    structured_output={
                        "ceo": {"name": "홍대표", "current_role": "CEO", "experience_tags": ["robot_sw_ai"]},
                        "key_members": [{"name": "김총괄", "current_role": "COO", "experience_tags": ["business_development"]}],
                        "role_coverage": {"robot_sw_ai": True},
                        "assessment_summary": "핵심팀 공개 근거 확보",
                        "evidence_quality": "공개 URL 2건 이상",
                    },
                ),
                agent_product_market_analysis_state=build_agent_state(
                    agent_name="agent_product_market_analysis",
                    summary="제품/시장 논리가 있습니다.",
                    structured_output={
                        "target_kpi_logic": {"text": "ROI 개선 논리", "references": ["[1]"]},
                        "technical_moat": {"text": "통합 난이도 해자", "references": ["[1]"]},
                        "data_loop_structure": {"text": "현장 데이터 루프", "references": ["[1]"]},
                        "product_summary": {"text": "추가 검증이 필요", "references": ["[1]"]},
                    },
                ),
                traction_state=build_agent_state(
                    agent_name="traction",
                    summary="상용화 신호가 있습니다.",
                    structured_output={
                        "partnerships": ["파트너십 1건"],
                        "hiring_analysis": {"field_engineer_ratio": 0.2, "field_engineer_count": 2, "hiring_trend_3m": 1},
                        "funding_velocity": ["시리즈 A 투자"],
                        "traction_summary": "상용화 진전 확인",
                    },
                ),
                agent_risk_search_state=build_agent_state(
                    agent_name="agent_risk_search",
                    summary="중대한 리스크는 제한적입니다.",
                    structured_output={
                        "risk_state": {
                            "legal_regulatory": "중대한 법적 이슈 미확인",
                            "certification_status": ["기본 인증 언급"],
                            "red_flags": [],
                            "risk_summary": "리스크는 제한적입니다.",
                        }
                    },
                ),
                review_state={
                    "status": "completed",
                    "summary": "일부 해석상 긴장이 있습니다.",
                    "agent_statuses": {},
                    "cautions": ["팀 근거 보강 필요"],
                    "contradictions": review_contradictions,
                },
            )

        self.assertEqual(result["final_decision"], "watch")
        self.assertEqual(result["weighted_score"], 3.5)
        self.assertEqual(result["next_action"], "report")
        self.assertEqual(result["review_contradictions"], review_contradictions)
        self.assertIn("agent_product_market_analysis", result["agent_structured_highlights"])
        self.assertEqual(len(result["criteria_scores"]), 6)
        self.assertEqual(
            result["agent_structured_highlights"]["investigate_members"]["ceo"]["name"],
            "홍대표",
        )

    def test_build_eval_state_routes_to_retry_when_weighted_score_is_too_low(self) -> None:
        structured_llm = Mock(
            return_value=EvalDecisionOutput(
                final_decision="pass",
                summary="근거가 전반적으로 부족합니다.",
                criteria_scores=[
                    EvalCriterionScoreOutput(
                        criterion_id="C1",
                        criterion_name="창업자 및 핵심팀 신뢰도",
                        score=2,
                        rationale="팀 증거가 약합니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C2",
                        criterion_name="시장 문제 및 도입 논리 명확성",
                        score=2,
                        rationale="시장 논리가 약합니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C3",
                        criterion_name="제품 완성도 및 시스템 차별성",
                        score=2,
                        rationale="제품 차별화 근거가 제한적입니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C4",
                        criterion_name="상용화 진전도 및 시장 검증",
                        score=2,
                        rationale="상용화 신호가 약합니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C5",
                        criterion_name="AI 자율성 및 데이터 운영 우위",
                        score=2,
                        rationale="데이터 운영 근거가 약합니다.",
                    ),
                    EvalCriterionScoreOutput(
                        criterion_id="C6",
                        criterion_name="공개 리스크 및 안전·규제 대응",
                        score=3,
                        rationale="리스크는 중립 수준입니다.",
                    ),
                ],
                key_strengths=[],
                key_risks=["추가 검증 필요"],
            )
        )
        model = Mock()
        model.with_structured_output.return_value.invoke = structured_llm

        with patch("agents.agent_eval.service.get_chat_model", return_value=model):
            result = build_eval_state(
                selected_company={
                    "company_id": "CP_RETRY",
                    "company_name": "재탐색회사",
                },
                investigate_members_state=build_agent_state(
                    agent_name="investigate_members",
                    summary="팀 근거 부족",
                    structured_output={},
                ),
                agent_product_market_analysis_state=build_agent_state(
                    agent_name="agent_product_market_analysis",
                    summary="시장 근거 부족",
                    structured_output={},
                ),
                traction_state=build_agent_state(
                    agent_name="traction",
                    summary="traction 약함",
                    structured_output={"traction_summary": "약한 신호"},
                ),
                agent_risk_search_state=build_agent_state(
                    agent_name="agent_risk_search",
                    summary="리스크 중립",
                    structured_output={"risk_state": {"risk_summary": "중립"}},
                ),
                review_state={
                    "status": "completed",
                    "summary": "전반적으로 약함",
                    "agent_statuses": {},
                    "cautions": [],
                    "contradictions": [],
                },
            )

        self.assertEqual(result["next_action"], "retry_find_company")
        self.assertLess(result["weighted_score"], 2.35)
        self.assertIn("가중 점수", result["retry_reason"])

    def test_build_eval_state_stops_on_system_failure(self) -> None:
        result = build_eval_state(
            selected_company={
                "company_id": "CP_STOP",
                "company_name": "중단회사",
            },
            investigate_members_state=build_agent_state(
                agent_name="investigate_members",
                summary="핵심팀 확인",
                structured_output={},
            ),
            agent_product_market_analysis_state=build_agent_state(
                agent_name="agent_product_market_analysis",
                summary="Error code: 429 - rate limit exceeded",
                structured_output={},
                status="failed",
            ),
            traction_state=build_agent_state(
                agent_name="traction",
                summary="traction 좋음",
                structured_output={"traction_summary": "좋음"},
            ),
            agent_risk_search_state=build_agent_state(
                agent_name="agent_risk_search",
                summary="리스크 제한적",
                structured_output={"risk_state": {"risk_summary": "제한적"}},
            ),
            review_state={
                "status": "completed",
                "summary": "시스템 오류 포함",
                "agent_statuses": {},
                "cautions": [],
                "contradictions": [],
            },
        )

        self.assertEqual(result["next_action"], "stop")
        self.assertIn("시스템 오류", result["retry_reason"])

    def test_build_eval_state_uses_fallback_model_on_rate_limit(self) -> None:
        primary_runnable = Mock()
        primary_runnable.invoke.side_effect = RuntimeError("429 rate limit exceeded")
        primary_model = Mock()
        primary_model.with_structured_output.return_value = primary_runnable

        fallback_runnable = Mock(
            return_value=EvalDecisionOutput(
                final_decision="watch",
                summary="fallback 모델이 평가를 완료했습니다.",
                criteria_scores=[
                    EvalCriterionScoreOutput(
                        criterion_id="C1",
                        criterion_name="창업자 및 핵심팀 신뢰도",
                        score=3,
                        rationale="fallback",
                    )
                ],
                key_strengths=["fallback 강점"],
                key_risks=[],
            )
        )
        fallback_model = Mock()
        fallback_model.with_structured_output.return_value.invoke = fallback_runnable

        with (
            patch("agents.agent_eval.service.get_chat_model", return_value=primary_model),
            patch(
                "agents.agent_eval.service.get_fallback_chat_model",
                return_value=fallback_model,
            ),
        ):
            result = build_eval_state(
                selected_company={
                    "company_id": "CP_FALLBACK",
                    "company_name": "폴백회사",
                },
                investigate_members_state=build_agent_state(
                    agent_name="investigate_members",
                    summary="팀 확인",
                    structured_output={},
                ),
                agent_product_market_analysis_state=build_agent_state(
                    agent_name="agent_product_market_analysis",
                    summary="제품 확인",
                    structured_output={},
                ),
                traction_state=build_agent_state(
                    agent_name="traction",
                    summary="traction 확인",
                    structured_output={},
                ),
                agent_risk_search_state=build_agent_state(
                    agent_name="agent_risk_search",
                    summary="리스크 확인",
                    structured_output={},
                ),
                review_state={
                    "status": "completed",
                    "summary": "리뷰 완료",
                    "agent_statuses": {},
                    "cautions": [],
                    "contradictions": [],
                },
            )

        self.assertEqual(result["summary"], "fallback 모델이 평가를 완료했습니다.")
        self.assertEqual(result["next_action"], "report")
        fallback_runnable.assert_called_once()


if __name__ == "__main__":
    unittest.main()
