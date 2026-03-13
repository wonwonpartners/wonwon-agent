from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from agents.agent_review.node import review_node
from agents.agent_review.service import ParallelReviewOutput, ReviewContradictionOutput


def build_agent_state(
    *,
    agent_name: str,
    summary: str,
    findings: list[str],
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "status": "completed",
        "attempt_count": 1,
        "input_company_id": "CP_REVIEW",
        "summary": summary,
        "findings": findings,
        "sources": [],
        "structured_output": {
            "summary": summary,
            "findings": findings,
        },
    }


class ReviewNodeTests(unittest.TestCase):
    def test_review_node_adds_llm_cautions_to_state(self) -> None:
        structured_llm = Mock(
            return_value=ParallelReviewOutput(
                summary="제품/시장 관점은 낙관적이지만 팀/리스크 관점의 보수적 해석을 함께 봐야 합니다.",
                cautions=[
                    "상용화 신호는 있으나 핵심 실행 인력 근거가 충분하지 않아 traction 해석에 유의가 필요합니다."
                ],
                contradictions=[
                    ReviewContradictionOutput(
                        topic="상용화 해석",
                        concern="traction은 상용화 신호를 강조하지만 investigate_members는 실행 인력 근거 부족을 보여줍니다.",
                        severity="medium",
                        related_agents=["traction", "investigate_members"],
                    )
                ],
            )
        )
        model = Mock()
        model.with_structured_output.return_value.invoke = structured_llm

        state = {
            "investigate_members_state": build_agent_state(
                agent_name="investigate_members",
                summary="대표는 확인되지만 핵심팀 공개 근거는 제한적입니다.",
                findings=["핵심팀 공개 자료 부족"],
            ),
            "agent_product_market_analysis_state": build_agent_state(
                agent_name="agent_product_market_analysis",
                summary="제품/시장 적합성은 긍정적으로 보입니다.",
                findings=["해자와 데이터 루프 가능성 존재"],
            ),
            "agent_risk_search_state": build_agent_state(
                agent_name="agent_risk_search",
                summary="중대한 리스크는 제한적입니다.",
                findings=["즉시 중단 수준 리스크 미확인"],
            ),
            "traction_state": build_agent_state(
                agent_name="traction",
                summary="상용화와 파트너십 신호가 있습니다.",
                findings=["파트너십과 투자 신호 확인"],
            ),
        }

        with patch("agents.agent_review.service.get_chat_model", return_value=model):
            result = review_node(state)

        self.assertEqual(result["review_state"]["status"], "completed")
        self.assertIn("보수적 해석", result["review_state"]["summary"])
        self.assertEqual(len(result["review_state"]["cautions"]), 2)
        self.assertEqual(len(result["review_state"]["contradictions"]), 1)
        self.assertEqual(
            result["review_state"]["contradictions"][0]["related_agents"],
            ["traction", "investigate_members"],
        )


if __name__ == "__main__":
    unittest.main()
