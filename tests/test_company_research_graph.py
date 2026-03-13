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


def build_product_market_analysis_output(
    *,
    status: str = "completed",
    attempt_count: int = 1,
) -> dict[str, object]:
    return {
        "agent_product_market_analysis_state": {
            "agent_name": "agent_product_market_analysis",
            "status": status,
            "attempt_count": attempt_count,
            "input_company_id": "CP_AGENT_PM",
            "summary": "제품/시장 분석 요약",
            "findings": [
                "KPI/ROI 논리: 반복 매출 전환 가능성이 보인다.",
                "기술 해자: 현장 통합 난이도가 장벽이다.",
                "데이터 루프/폴백: 운영 데이터 축적이 가능하다.",
                "종합 요약: 추가 검증은 필요하나 초기 신호는 있다.",
            ],
            "sources": [
                {
                    "source_type": "rag_document",
                    "tool_name": "domain_rag_search_tool",
                    "title": "테스트 산업 보고서",
                    "publisher": "테스트 발행처",
                    "published_at": "2026-03-12",
                    "url": "https://example.com/report",
                    "excerpt": "preview",
                }
            ],
            "structured_output": {
                "target_kpi_logic": "반복 매출 논리",
                "target_kpi_logic_sources": ["[1] 테스트 산업 보고서"],
                "technical_moat": "통합 해자",
                "technical_moat_sources": ["[1] 테스트 산업 보고서"],
                "data_loop_structure": "데이터 루프",
                "data_loop_structure_sources": ["[1] 테스트 산업 보고서"],
                "product_summary": "종합 요약",
                "product_summary_sources": ["[1] 테스트 산업 보고서"],
            },
        }
    }


def build_agent_risk_search_output(
    *,
    status: str = "completed",
    attempt_count: int = 1,
) -> dict[str, object]:
    return {
        "agent_risk_search_state": {
            "agent_name": "agent_risk_search",
            "status": status,
            "attempt_count": attempt_count,
            "input_company_id": "CP_AGENT_RISK",
            "summary": "리스크 탐지 요약",
            "findings": [
                "법적/규제 리스크: 현재 공개된 중대한 이슈는 제한적입니다.",
                "인증/특허: 기본 인증 및 지식재산 신호가 일부 보입니다.",
                "레드플래그: 즉시 중단 수준의 강한 신호는 아직 없습니다.",
            ],
            "sources": [
                {
                    "source_type": "web",
                    "title": "테스트컴퍼니 인증 현황",
                    "url": "https://example.com/risk",
                    "snippet": "인증 및 특허 관련 공개 정보",
                    "published_at": "2026-03-12",
                    "query": "\"테스트컴퍼니\" 인증 특허",
                }
            ],
            "structured_output": {
                "risk_state": {
                    "legal_regulatory": "중대한 리스크 미확인",
                    "certification_status": ["기본 인증 언급"],
                    "red_flags": [],
                    "risk_summary": "현재 공개 자료 기준 중대한 리스크는 제한적입니다.",
                }
            },
        }
    }


def build_review_output(
    *,
    cautions: list[str] | None = None,
    contradictions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "review_state": {
            "status": "completed",
            "summary": "리뷰 요약",
            "agent_statuses": {
                "investigate_members": "completed",
                "agent_product_market_analysis": "completed",
                "agent_risk_search": "completed",
                "traction": "completed",
            },
            "cautions": cautions or [],
            "contradictions": contradictions or [],
        }
    }


def build_eval_output(
    *,
    summary: str = "평가 요약",
    ready_for_report: bool = True,
    status: str = "completed",
    weighted_score: float = 3.2,
    next_action: str | None = None,
    retry_reason: str = "",
    review_cautions: list[str] | None = None,
    review_contradictions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "eval_state": {
            "status": status,
            "ready_for_report": ready_for_report,
            "summary": summary,
            "weighted_score": weighted_score,
            "agent_summaries": {
                "investigate_members": "v",
                "agent_product_market_analysis": "v",
                "agent_risk_search": "v",
                "traction": "v",
            },
            "review_summary": "리뷰 요약",
            "review_cautions": review_cautions or [],
            "review_contradictions": review_contradictions or [],
            "agent_structured_highlights": {},
            "final_decision": "watch",
            "criteria_scores": [],
            "key_strengths": [],
            "key_risks": [],
            "next_action": next_action or ("report" if ready_for_report else "stop"),
            "retry_reason": retry_reason,
        }
    }


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
        product_market_analysis_mock = Mock(return_value={})
        agent_risk_search_mock = Mock(return_value={})
        traction_mock = Mock(return_value={})
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
            patch(
                "company_research_graph.product_market_analysis_node",
                product_market_analysis_mock,
            ),
            patch(
                "company_research_graph.agent_risk_search_node",
                agent_risk_search_mock,
            ),
            patch("company_research_graph.traction_node", traction_mock),
            patch("company_research_graph.review_node", review_mock),
            patch("company_research_graph.eval_node", eval_mock),
            patch("company_research_graph.report_node", report_mock),
        ):
            result = run_company_research("로봇 회사")

        self.assertIsNone(result["selected_company"])
        self.assertNotIn("graph_error", result)
        self.assertNotIn("report_state", result)
        investigate_members_mock.assert_not_called()
        product_market_analysis_mock.assert_not_called()
        agent_risk_search_mock.assert_not_called()
        traction_mock.assert_not_called()
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
                    "company_research_graph.product_market_analysis_node",
                    return_value=build_product_market_analysis_output(),
                ),
                patch(
                    "company_research_graph.agent_risk_search_node",
                    return_value=build_agent_risk_search_output(),
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
                patch(
                    "company_research_graph.review_node",
                    return_value=build_review_output(),
                ),
                patch(
                    "company_research_graph.eval_node",
                    return_value=build_eval_output(
                        summary="첫 평가 요약",
                        ready_for_report=True,
                        status="completed",
                    ),
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

    def test_failed_investigation_without_review_retry_blocks_eval(self) -> None:
        selected_company = build_selected_company(company_id="CP_RETRY")
        collect_mock = Mock(return_value=build_signals())
        extract_mock = Mock(
            return_value=build_ceo_only_extraction()
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
                    "company_research_graph.product_market_analysis_node",
                    return_value=build_product_market_analysis_output(),
                ),
                patch(
                    "company_research_graph.agent_risk_search_node",
                    return_value=build_agent_risk_search_output(),
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
                patch(
                    "company_research_graph.review_node",
                    return_value=build_review_output(
                        cautions=["핵심팀 근거 부족으로 실행 리스크 해석에 유의 필요"],
                    ),
                ),
                patch(
                    "company_research_graph.eval_node",
                    return_value=build_eval_output(
                        summary="보완 필요",
                        ready_for_report=False,
                        status="blocked",
                        next_action="stop",
                        review_cautions=["핵심팀 근거 부족으로 실행 리스크 해석에 유의 필요"],
                    ),
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(collect_mock.call_count, 1)
        self.assertEqual(extract_mock.call_count, 1)
        self.assertEqual(result["investigate_members_state"]["attempt_count"], 1)
        self.assertEqual(result["investigate_members_state"]["status"], "failed")
        self.assertEqual(result["eval_state"]["status"], "blocked")
        self.assertFalse(result["eval_state"]["ready_for_report"])
        self.assertEqual(
            result["eval_state"]["review_cautions"],
            ["핵심팀 근거 부족으로 실행 리스크 해석에 유의 필요"],
        )
        self.assertNotIn("report_state", result)
        self.assertNotIn("graph_error", result)

    def test_failed_investigation_keeps_graph_alive_but_blocks_report(self) -> None:
        selected_company = build_selected_company(company_id="CP_FAIL")
        collect_mock = Mock(return_value=build_signals())
        extract_mock = Mock(return_value=build_ceo_only_extraction())

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
                    "company_research_graph.product_market_analysis_node",
                    return_value=build_product_market_analysis_output(),
                ),
                patch(
                    "company_research_graph.agent_risk_search_node",
                    return_value=build_agent_risk_search_output(),
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
                patch(
                    "company_research_graph.review_node",
                    return_value=build_review_output(
                        cautions=["조사 결과 간 해석 차이가 있어 보수적 판단 필요"],
                    ),
                ),
                patch(
                    "company_research_graph.eval_node",
                    return_value=build_eval_output(
                        summary="추가 검증 필요",
                        ready_for_report=False,
                        status="blocked",
                        next_action="stop",
                        review_cautions=["조사 결과 간 해석 차이가 있어 보수적 판단 필요"],
                    ),
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(collect_mock.call_count, 1)
        self.assertEqual(extract_mock.call_count, 1)
        self.assertEqual(result["eval_state"]["status"], "blocked")
        self.assertFalse(result["eval_state"]["ready_for_report"])
        self.assertNotIn("report_state", result)
        self.assertNotIn("graph_error", result)

    def test_retry_find_company_routes_back_to_search_with_exclusion_history(self) -> None:
        first_company = build_selected_company(company_id="CP_FIRST", company_name="첫회사")
        second_company = build_selected_company(company_id="CP_SECOND", company_name="둘회사")

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch(
                    "company_research_graph.find_company_node",
                    side_effect=[
                        build_find_company_output(selected_company=first_company),
                        build_find_company_output(selected_company=second_company),
                    ],
                ) as find_company_mock,
                patch(
                    "company_research_graph.investigate_members_node",
                    return_value={"investigate_members_state": {"agent_name": "investigate_members", "status": "completed", "attempt_count": 1, "input_company_id": "CP_FIRST", "summary": "팀 요약", "findings": [], "sources": [], "structured_output": {}}},
                ),
                patch(
                    "company_research_graph.product_market_analysis_node",
                    return_value=build_product_market_analysis_output(),
                ),
                patch(
                    "company_research_graph.agent_risk_search_node",
                    return_value=build_agent_risk_search_output(),
                ),
                patch(
                    "company_research_graph.traction_node",
                    return_value={"traction_state": build_completed_traction_state(company_id="CP_FIRST", company_name="첫회사")},
                ),
                patch(
                    "company_research_graph.review_node",
                    return_value=build_review_output(),
                ),
                patch(
                    "company_research_graph.eval_node",
                    side_effect=[
                        build_eval_output(
                            summary="첫 회사 재탐색",
                            ready_for_report=False,
                            status="blocked",
                            weighted_score=2.1,
                            next_action="retry_find_company",
                            retry_reason="가중 점수가 낮아 다른 회사를 찾습니다.",
                        ),
                        build_eval_output(
                            summary="둘 회사 통과",
                            ready_for_report=True,
                            status="completed",
                            weighted_score=3.4,
                            next_action="report",
                        ),
                    ],
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(find_company_mock.call_count, 2)
        self.assertEqual(result["selected_company"]["company_id"], "CP_SECOND")
        self.assertEqual(result["company_retry_count"], 1)
        self.assertEqual(result["evaluated_company_ids"], ["CP_FIRST"])
        self.assertEqual(result["candidate_eval_history"][0]["company_id"], "CP_FIRST")
        self.assertEqual(result["report_state"]["status"], "completed")

    def test_report_path_is_stable_and_overwrites_existing_file(self) -> None:
        selected_company = build_selected_company(company_id="CP_STABLE")
        eval_v1 = {
            "eval_state": {
                **build_eval_output(
                    summary="첫 번째 평가 요약",
                    ready_for_report=True,
                    status="completed",
                )["eval_state"],
                "agent_summaries": {
                    "investigate_members": "v1",
                    "agent_product_market_analysis": "v1",
                    "agent_risk_search": "v1",
                    "traction": "v1",
                },
            }
        }
        eval_v2 = {
            "eval_state": {
                **build_eval_output(
                    summary="두 번째 평가 요약",
                    ready_for_report=True,
                    status="completed",
                )["eval_state"],
                "agent_summaries": {
                    "investigate_members": "v2",
                    "agent_product_market_analysis": "v2",
                    "agent_risk_search": "v2",
                    "traction": "v2",
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
                    "company_research_graph.product_market_analysis_node",
                    return_value=build_product_market_analysis_output(),
                ),
                patch(
                    "company_research_graph.agent_risk_search_node",
                    return_value=build_agent_risk_search_output(),
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
                patch(
                    "company_research_graph.review_node",
                    return_value=build_review_output(),
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
                    "company_research_graph.product_market_analysis_node",
                    return_value=build_product_market_analysis_output(),
                ),
                patch(
                    "company_research_graph.agent_risk_search_node",
                    return_value=build_agent_risk_search_output(),
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
                patch(
                    "company_research_graph.review_node",
                    return_value=build_review_output(),
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
