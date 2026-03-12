from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.agent_investigate_members.result import (
    InvestigateMemberExtraction,
    InvestigateMembersExtractionResult,
)
from company_research_graph import run_company_research


def build_selected_company(
    *,
    company_id: str = "CP00000001",
    company_name: str = "테스트컴퍼니",
) -> dict[str, object]:
    return {
        "company_id": company_id,
        "company_name": company_name,
        "product_name": "테스트 제품",
        "description": "테스트 설명",
        "employees": 10,
        "revenue": 1000,
        "invest_count": 1,
        "invest_level": "seed",
        "hiring": True,
        "categories": ["AI/딥테크/블록체인"],
        "keywords": ["AI"],
    }


def build_find_company_output(
    *,
    selected_company: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "company_search_summary": "테스트 검색 요약",
        "company_search_filters": {"query": "테스트"},
        "selected_company": selected_company,
        "selected_company_reason": "테스트 선정 사유",
    }


def build_signal(
    *,
    source_id: str,
    url: str,
    title: str,
    snippet: str,
    query: str = '"테스트컴퍼니" CEO 대표 창업자',
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "snippet": snippet,
        "published_at": "2026-03-01",
        "query": query,
        "source_kind": "web",
        "domain": url.split("/")[2],
    }


def build_signals(*, distinct_urls: int = 2) -> list[dict[str, str]]:
    signals = [
        build_signal(
            source_id="S1",
            url="https://news.example.com/leadership",
            title="테스트컴퍼니 홍대표 인터뷰",
            snippet="테스트컴퍼니 홍대표가 로봇 SW/AI와 제품화 전략을 설명했다.",
        ),
        build_signal(
            source_id="S2",
            url="https://company.example.com/team",
            title="테스트컴퍼니 핵심팀 소개",
            snippet="테스트컴퍼니 김총괄 COO와 이CTO가 현장 배치와 시스템 통합을 이끌고 있다.",
            query='"테스트컴퍼니" CTO COO CPO Head 총괄',
        ),
    ]
    if distinct_urls <= 1:
        signals[1]["url"] = signals[0]["url"]
        signals[1]["domain"] = signals[0]["domain"]
    return signals


def build_completed_extraction() -> InvestigateMembersExtractionResult:
    return InvestigateMembersExtractionResult(
        ceo=InvestigateMemberExtraction(
            name="홍대표",
            current_role="CEO",
            is_founder=True,
            experience_tags=["robot_sw_ai", "productization_deployment"],
            evidence_summary="로봇 AI 제품 상용화와 배치 경험이 언급된다.",
            source_ids=["S1"],
            confidence=0.91,
        ),
        key_members=[
            InvestigateMemberExtraction(
                name="김총괄",
                current_role="COO",
                is_founder=False,
                experience_tags=["system_integration", "business_development"],
                evidence_summary="사업 운영과 시스템 통합 총괄 역할이 확인된다.",
                source_ids=["S2"],
                confidence=0.82,
            )
        ],
        strengths=["CEO와 COO 모두 로봇 제품화/운영 경험 근거가 확인됩니다."],
        evidence_gaps=["제조/운영 리더십의 추가 검증은 더 필요합니다."],
        assessment_summary="대표와 핵심 운영 리더에 대한 공개 근거가 확보됐습니다.",
        evidence_quality="서로 다른 공개 URL 2건에서 CEO와 핵심팀을 교차 확인했습니다.",
    )


def build_ceo_only_extraction() -> InvestigateMembersExtractionResult:
    return InvestigateMembersExtractionResult(
        ceo=InvestigateMemberExtraction(
            name="홍대표",
            current_role="CEO",
            is_founder=True,
            experience_tags=["robot_sw_ai"],
            evidence_summary="대표의 로봇 AI 경력이 공개 인터뷰에 언급됩니다.",
            source_ids=["S1"],
            confidence=0.88,
        ),
        key_members=[],
        strengths=[],
        evidence_gaps=["핵심팀 공개 자료가 부족합니다."],
        assessment_summary="대표는 보이지만 비CEO 핵심팀 근거가 없습니다.",
        evidence_quality="리더십 공개 근거가 제한적입니다.",
    )


def build_completed_traction_state(
    *,
    company_id: str = "CP00000001",
    company_name: str = "테스트컴퍼니",
) -> dict[str, object]:
    return {
        "agent_name": "traction",
        "status": "completed",
        "attempt_count": 1,
        "input_company_id": company_id,
        "summary": "파트너십과 투자 이력이 확인되어 traction 신호가 존재합니다.",
        "findings": [
            "파트너십 신호: 테스트 파트너십 1건",
            "채용 신호: Field Engineer 비중 0.0%, 공고 수 0건, 최근 3개월 트렌드 0",
            "투자/성장 신호: 시리즈 A 투자 유치",
        ],
        "sources": [
            {
                "source_type": "selected_company",
                "company_id": company_id,
                "company_name": company_name,
            }
        ],
        "structured_output": {
            "partnerships": ["테스트 파트너십 1건"],
            "hiring_analysis": {
                "field_engineer_ratio": 0.0,
                "field_engineer_count": 0,
                "hiring_trend_3m": 0,
            },
            "funding_velocity": ["시리즈 A 투자 유치"],
            "traction_summary": "파트너십과 투자 이력이 확인되어 traction 신호가 존재합니다.",
        },
    }


class CompanyResearchGraphTests(unittest.TestCase):
    def test_ends_early_when_no_company_is_selected(self) -> None:
        investigate_members_mock = Mock(return_value={})
        traction_mock = Mock(return_value={})
        agent_a_mock = Mock(return_value={})
        review_mock = Mock(return_value={})
        eval_mock = Mock(return_value={})
        report_mock = Mock(return_value={})

        with (
            patch(
                "company_research_graph.find_company_node",
                return_value=build_find_company_output(selected_company=None),
            ),
            patch(
                "company_research_graph.investigate_members_node",
                investigate_members_mock,
            ),
            patch("company_research_graph.traction_node", traction_mock),
            patch("company_research_graph.agent_a_node", agent_a_mock),
            patch("company_research_graph.review_investigate_members_node", review_mock),
            patch("company_research_graph.eval_node", eval_mock),
            patch("company_research_graph.report_node", report_mock),
        ):
            result = run_company_research("로봇 회사")

        self.assertIsNone(result["selected_company"])
        self.assertNotIn("graph_error", result)
        self.assertNotIn("report_state", result)
        investigate_members_mock.assert_not_called()
        traction_mock.assert_not_called()
        agent_a_mock.assert_not_called()
        review_mock.assert_not_called()
        eval_mock.assert_not_called()
        report_mock.assert_not_called()

    def test_success_path_runs_eval_and_generates_report(self) -> None:
        selected_company = build_selected_company(company_id="CP_SUCCESS")
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch(
                    "company_research_graph.find_company_node",
                    return_value=build_find_company_output(
                        selected_company=selected_company
                    ),
                ),
                patch(
                    "agents.agent_investigate_members.service.collect_investigate_member_signals",
                    return_value=build_signals(),
                ),
                patch(
                    "agents.agent_investigate_members.service.extract_investigate_members",
                    return_value=build_completed_extraction(),
                ),
                patch(
                    "company_research_graph.traction_node",
                    return_value={
                        "traction_state": build_completed_traction_state(
                            company_id="CP_SUCCESS",
                            company_name="테스트컴퍼니",
                        )
                    },
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")
                report_path = Path(result["report_state"]["report_path"])
                self.assertTrue(report_path.exists())

        self.assertEqual(result["eval_state"]["status"], "completed")
        self.assertEqual(result["report_state"]["status"], "completed")
        self.assertEqual(result["investigate_members_state"]["status"], "completed")
        self.assertEqual(result["leadership_research"]["ceo"]["name"], "홍대표")
        self.assertTrue(result["report_state"]["report_path"].endswith("CP_SUCCESS.md"))
        self.assertIn("### CEO", result["report_state"]["markdown"])
        self.assertIn("### 핵심팀", result["report_state"]["markdown"])
        self.assertIn("## traction", result["report_state"]["markdown"])
        self.assertIn("## eval 요약", result["report_state"]["markdown"])

    def test_retry_once_after_failed_investigation_reaches_success(self) -> None:
        selected_company = build_selected_company(company_id="CP_RETRY")
        collect_mock = Mock(side_effect=[build_signals(), build_signals()])
        extract_mock = Mock(
            side_effect=[
                build_ceo_only_extraction(),
                build_completed_extraction(),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch(
                    "company_research_graph.find_company_node",
                    return_value=build_find_company_output(
                        selected_company=selected_company
                    ),
                ),
                patch(
                    "agents.agent_investigate_members.service.collect_investigate_member_signals",
                    collect_mock,
                ),
                patch(
                    "agents.agent_investigate_members.service.extract_investigate_members",
                    extract_mock,
                ),
                patch(
                    "company_research_graph.traction_node",
                    return_value={
                        "traction_state": build_completed_traction_state(
                            company_id="CP_RETRY",
                            company_name="테스트컴퍼니",
                        )
                    },
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(collect_mock.call_count, 2)
        self.assertEqual(extract_mock.call_count, 2)
        self.assertEqual(result["investigate_members_state"]["attempt_count"], 2)
        self.assertEqual(result["investigate_members_state"]["status"], "completed")
        self.assertEqual(result["report_state"]["status"], "completed")
        self.assertNotIn("graph_error", result)

    def test_second_failed_investigation_terminates_graph_with_error_state(self) -> None:
        selected_company = build_selected_company(company_id="CP_FAIL")
        collect_mock = Mock(side_effect=[build_signals(), build_signals()])
        extract_mock = Mock(
            side_effect=[
                build_ceo_only_extraction(),
                build_ceo_only_extraction(),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch(
                    "company_research_graph.find_company_node",
                    return_value=build_find_company_output(
                        selected_company=selected_company
                    ),
                ),
                patch(
                    "agents.agent_investigate_members.service.collect_investigate_member_signals",
                    collect_mock,
                ),
                patch(
                    "agents.agent_investigate_members.service.extract_investigate_members",
                    extract_mock,
                ),
                patch(
                    "company_research_graph.traction_node",
                    return_value={
                        "traction_state": build_completed_traction_state(
                            company_id="CP_FAIL",
                            company_name="테스트컴퍼니",
                        )
                    },
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(result["graph_error"]["stage"], "review")
        self.assertEqual(result["graph_error"]["agent_name"], "investigate_members")
        self.assertIn("completed가 아니어서", result["graph_error"]["message"])
        self.assertNotIn("eval_state", result)
        self.assertNotIn("report_state", result)

    def test_report_path_is_stable_and_overwrites_existing_file(self) -> None:
        selected_company = build_selected_company(company_id="CP_STABLE")
        eval_v1 = {
            "eval_state": {
                "status": "completed",
                "ready_for_report": True,
                "summary": "첫 번째 평가 요약",
                "agent_summaries": {
                    "investigate_members": "v1",
                    "agent_a": "v1",
                    "agent_b": "v1",
                    "agent_c": "v1",
                },
            }
        }
        eval_v2 = {
            "eval_state": {
                "status": "completed",
                "ready_for_report": True,
                "summary": "두 번째 평가 요약",
                "agent_summaries": {
                    "investigate_members": "v2",
                    "agent_a": "v2",
                    "agent_b": "v2",
                    "agent_c": "v2",
                },
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch(
                    "company_research_graph.find_company_node",
                    return_value=build_find_company_output(
                        selected_company=selected_company
                    ),
                ),
                patch(
                    "agents.agent_investigate_members.service.collect_investigate_member_signals",
                    return_value=build_signals(),
                ),
                patch(
                    "agents.agent_investigate_members.service.extract_investigate_members",
                    return_value=build_completed_extraction(),
                ),
                patch(
                    "company_research_graph.traction_node",
                    return_value={
                        "traction_state": build_completed_traction_state(
                            company_id="CP_STABLE",
                            company_name="테스트컴퍼니",
                        )
                    },
                ),
                patch("company_research_graph.eval_node", return_value=eval_v1),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                first_result = run_company_research("로봇 회사")

            with (
                patch(
                    "company_research_graph.find_company_node",
                    return_value=build_find_company_output(
                        selected_company=selected_company
                    ),
                ),
                patch(
                    "agents.agent_investigate_members.service.collect_investigate_member_signals",
                    return_value=build_signals(),
                ),
                patch(
                    "agents.agent_investigate_members.service.extract_investigate_members",
                    return_value=build_completed_extraction(),
                ),
                patch(
                    "company_research_graph.traction_node",
                    return_value={
                        "traction_state": build_completed_traction_state(
                            company_id="CP_STABLE",
                            company_name="테스트컴퍼니",
                        )
                    },
                ),
                patch("company_research_graph.eval_node", return_value=eval_v2),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                second_result = run_company_research("로봇 회사")

            report_path = Path(second_result["report_state"]["report_path"])
            content = report_path.read_text(encoding="utf-8")

        self.assertEqual(
            first_result["report_state"]["report_path"],
            second_result["report_state"]["report_path"],
        )
        self.assertIn("두 번째 평가 요약", content)
        self.assertNotIn("첫 번째 평가 요약", content)


if __name__ == "__main__":
    unittest.main()
