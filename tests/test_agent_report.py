from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import agents.agent_report.common as report_common
from agents.agent_report.service import (
    ReportDraftOutput,
    ReportFieldOutput,
    SUMMARY_MAX_CHARS,
    build_report_state,
)


def build_selected_company() -> dict[str, object]:
    return {
        "company_id": "CP_REPORT",
        "company_name": "테스트로보틱스",
        "product_name": "지능형 물류 로봇",
        "description": "물류 자동화용 Robotics AI 솔루션을 제공한다.",
        "invest_level": "series_a",
        "categories": ["로보틱스"],
    }


def build_investigate_members_state() -> dict[str, object]:
    return {
        "agent_name": "investigate_members",
        "status": "completed",
        "attempt_count": 1,
        "input_company_id": "CP_REPORT",
        "summary": "대표와 핵심팀 공개 근거가 확보됐다.",
        "findings": ["대표 인터뷰와 팀 소개 페이지가 확인된다."],
        "sources": [
            {
                "source_id": "S1",
                "title": "테스트로보틱스 대표 인터뷰",
                "url": "https://news.example.com/ceo",
                "published_at": "2026-03-05",
                "snippet": "대표가 제품화 전략을 설명했다.",
            },
            {
                "source_id": "S2",
                "title": "테스트로보틱스 팀 소개",
                "url": "https://company.example.com/team",
                "published_at": "2026-03-06",
                "snippet": "핵심팀 소개 페이지",
            },
        ],
        "structured_output": {
            "ceo": {"name": "홍대표", "current_role": "CEO"},
            "key_members": [{"name": "김총괄", "current_role": "COO"}],
            "assessment_summary": "대표와 핵심 운영 리더가 확인된다.",
            "evidence_quality": "서로 다른 URL 2건 이상",
        },
    }


def build_product_market_state() -> dict[str, object]:
    return {
        "agent_name": "agent_product_market_analysis",
        "status": "completed",
        "attempt_count": 1,
        "input_company_id": "CP_REPORT",
        "summary": "도입 논리와 기술 차별성 신호가 있다.",
        "findings": ["현장 통합 난이도와 데이터 루프 가능성이 보인다."],
        "sources": [
            {
                "source_type": "rag_document",
                "tool_name": "domain_rag_search_tool",
                "title": "글로벌 로보틱스 시장 전망",
                "publisher": "테스트연구소",
                "published_at": "2026-03-01",
                "url": "https://example.com/report",
                "excerpt": "산업 보고서 본문",
            },
            {
                "source_type": "rag_document",
                "tool_name": "domain_rag_search_tool",
                "title": "Robotics Benchmarking",
                "author": "홍길동",
                "journal": "Journal of Robotics",
                "published_at": "2025",
                "excerpt": "논문 초록",
            },
        ],
        "structured_output": {
            "target_kpi_logic": {
                "text": "ROI 개선 논리가 있다.",
                "references": ["글로벌 로보틱스 시장 전망"],
                "evidence_gap": "",
            },
            "technical_moat": {
                "text": "현장 통합 난이도가 진입장벽이다.",
                "references": ["글로벌 로보틱스 시장 전망"],
                "evidence_gap": "",
            },
            "data_loop_structure": {
                "text": "운영 데이터 축적 구조가 일부 확인된다.",
                "references": ["Robotics Benchmarking"],
                "evidence_gap": "",
            },
            "product_summary": {
                "text": "초기 PMF 신호가 있으나 추가 검증이 필요하다.",
                "references": ["글로벌 로보틱스 시장 전망"],
                "evidence_gap": "",
            },
        },
    }


def build_traction_state() -> dict[str, object]:
    evidence_sources = [
        {
            "source_type": "web",
            "signal_type": "partnership",
            "query": '"테스트로보틱스" 파트너십',
            "title": "파트너십 기사",
            "publisher": "traction.example.com",
            "published_at": "2026-03-09",
            "url": "https://traction.example.com/partnership",
            "source": "https://traction.example.com/partnership",
            "score": 0.92,
        }
    ]
    return {
        "agent_name": "traction",
        "status": "completed",
        "attempt_count": 1,
        "input_company_id": "CP_REPORT",
        "summary": "파트너십과 상용화 신호가 확인된다.",
        "findings": ["파트너십 1건", "시리즈 A 투자 유치"],
        "sources": evidence_sources,
        "structured_output": {
            "partnerships": ["물류사와 파트너십 1건"],
            "hiring_analysis": {
                "field_engineer_ratio": 0.2,
                "field_engineer_count": 2,
                "hiring_trend_3m": 1,
            },
            "funding_velocity": ["시리즈 A 투자 유치"],
            "traction_summary": "상용화 신호가 관측된다.",
            "evidence_sources": evidence_sources,
        },
    }


def build_risk_state() -> dict[str, object]:
    return {
        "agent_name": "agent_risk_search",
        "status": "completed",
        "attempt_count": 1,
        "input_company_id": "CP_REPORT",
        "summary": "중대한 공개 리스크는 제한적이다.",
        "findings": ["중대한 법적 이슈 미확인"],
        "sources": [
            {
                "source_type": "web",
                "title": "인증 현황",
                "url": "https://risk.example.com/cert",
                "published_at": "2026-03-08",
                "snippet": "인증 관련 공개 정보",
            }
        ],
        "structured_output": {
            "risk_state": {
                "legal_regulatory": "중대한 법적 이슈 미확인",
                "certification_status": ["기본 인증 언급"],
                "red_flags": [],
                "risk_summary": "공개 자료 기준 중대한 리스크는 제한적이다.",
            }
        },
    }


def build_review_state() -> dict[str, object]:
    return {
        "status": "completed",
        "summary": "제품 검증은 긍정적이지만 고객 확산 속도는 추가 확인이 필요하다.",
        "agent_statuses": {},
        "cautions": ["고객 확산 속도는 추가 확인이 필요하다."],
        "contradictions": [],
    }


def build_eval_state(*, ready_for_report: bool = True) -> dict[str, object]:
    return {
        "status": "completed" if ready_for_report else "blocked",
        "ready_for_report": ready_for_report,
        "summary": "팀과 시장 신호는 긍정적이나 상용화 속도는 추가 검증이 필요하다.",
        "agent_summaries": {
            "investigate_members": "대표와 핵심팀 확인",
            "agent_product_market_analysis": "시장성과 제품 경쟁력 존재",
            "agent_risk_search": "리스크 제한적",
            "traction": "상용화 신호 존재",
        },
        "review_summary": "리뷰 요약",
        "review_cautions": ["고객 확산 속도는 추가 확인이 필요하다."],
        "review_contradictions": [],
        "agent_structured_highlights": {},
        "final_decision": "watch",
        "criteria_scores": [
            {
                "criterion_id": "C1",
                "criterion_name": "창업자 및 핵심팀 신뢰도",
                "score": 4,
                "rationale": "대표와 핵심팀이 공개 자료에서 확인된다.",
            }
        ],
        "key_strengths": ["핵심팀과 상용화 초기 신호가 동시에 보인다."],
        "key_risks": ["고객 확산 속도는 추가 검증이 필요하다."],
    }


def build_llm_draft() -> ReportDraftOutput:
    return ReportDraftOutput(
        executive_summary=ReportFieldOutput(
            text=(
                "테스트로보틱스는 물류 자동화 시장에서 AI 기반 운영 효율 개선 솔루션을 전개하는 기업이다. "
                "핵심팀 공개 근거와 초기 상용화 신호는 확인되지만 고객 확산 속도는 추가 검증이 필요하다. "
                "종합 판단은 watch이며 후속 검증을 전제로 투자 검토를 이어갈 수 있다."
            ),
            source_ids=["SRC003", "SRC004", "SRC005"],
        ),
        company_intro=ReportFieldOutput(
            text="테스트로보틱스는 물류 운영 현장의 자동화를 겨냥한 Robotics AI 기업이다. 공개 자료 기준 제품화와 현장 적용을 함께 추진한다.",
            source_ids=["SRC003"],
        ),
        problem_solution=ReportFieldOutput(
            text="이 기업은 물류 현장의 비효율과 운영 비용 부담을 줄이는 문제를 겨냥한다. 솔루션은 도입 ROI와 작업 정확도 개선 논리를 함께 제시한다.",
            source_ids=["SRC003"],
        ),
        products_services=ReportFieldOutput(
            text="주요 제품은 지능형 물류 로봇과 운영 소프트웨어를 결합한 형태다. 단일 기능보다 현장 워크플로 전체를 개선하는 방향으로 설계한다.",
            source_ids=["SRC003"],
        ),
        market_size_growth=ReportFieldOutput(
            text="물류 자동화 시장은 비용 절감과 인력 효율화 요구를 배경으로 성장 여지가 있다. 다만 실제 시장 침투 속도는 추가 검증이 필요하다.",
            source_ids=["SRC003"],
        ),
        customer_demand=ReportFieldOutput(
            text="고객은 운영 효율, 정확도, 반복 가능한 자동화 도입 효과를 요구한다. 공개 신호 기준 해당 수요와 맞닿은 문제 정의는 비교적 명확하다.",
            source_ids=["SRC003"],
        ),
        competitive_landscape=ReportFieldOutput(
            text="경쟁 환경에서는 현장 통합 난이도와 도입 전환율이 중요한 변수다. 테스트로보틱스는 기술 차별성과 적용 속도를 함께 증명해야 한다.",
            source_ids=["SRC004"],
        ),
        product_maturity=ReportFieldOutput(
            text="제품 완성도는 초기 실증과 상용화 신호를 통해 일부 확인된다. 반복 도입 근거가 더 쌓이면 신뢰도가 높아진다.",
            source_ids=["SRC003"],
        ),
        technical_differentiation=ReportFieldOutput(
            text="기술 차별성은 현장 통합 난이도와 데이터 기반 운영 구조에서 나온다. 경쟁사 대비 실제 문제 해결 단위에서 우위를 더 증명해야 한다.",
            source_ids=["SRC004"],
        ),
        ai_data_advantage=ReportFieldOutput(
            text="AI 데이터 강점은 운영 과정에서 축적되는 피드백 루프에 있다. 모델 성능뿐 아니라 개선 주기와 fallback 구조를 함께 보여줄 필요가 있다.",
            source_ids=["SRC004"],
        ),
        founders_team=ReportFieldOutput(
            text="대표와 핵심 운영 인력의 공개 근거가 확인된다. 다만 비CEO 핵심 인력의 폭은 추가 검증이 필요하다.",
            source_ids=["SRC001"],
        ),
        commercialization_progress=ReportFieldOutput(
            text="사업화는 파트너십과 투자 유치 신호를 통해 일부 진척이 보인다. 매출화와 반복 도입 근거가 더 확인되면 평가가 강화된다.",
            source_ids=["SRC005"],
        ),
        customers_partnerships_performance=ReportFieldOutput(
            text="고객 및 파트너십 성과는 초기 신호가 존재한다. 단발성 협력을 넘어 반복성과 확산 속도를 더 확인해야 한다.",
            source_ids=["SRC005"],
        ),
        market_risk=ReportFieldOutput(
            text="시장 리스크는 고객 확산 속도와 실제 도입 전환율에 있다. traction 신호가 약해지면 시장 검증 리스크가 크게 해석된다.",
            source_ids=["SRC003"],
        ),
        technical_risk=ReportFieldOutput(
            text="기술 리스크는 차별성 주장 대비 실증 수준이 충분한지에 있다. 반복 성능과 운영 안정성을 추가 확인해야 한다.",
            source_ids=["SRC004"],
        ),
        regulatory_operational_risk=ReportFieldOutput(
            text="규제 및 운영 리스크는 인증과 운영 안정성 관리 체계에 있다. 확장 단계에서의 컴플라이언스 대응을 계속 점검해야 한다.",
            source_ids=["SRC006"],
        ),
        investment_points=ReportFieldOutput(
            text="투자 포인트는 팀 근거와 초기 상용화 신호가 동시에 존재한다는 점이다. 제품과 시장 검증 신호가 연결되면 투자 매력도가 높아진다.",
            source_ids=["SRC003", "SRC005"],
        ),
        overall_evaluation=ReportFieldOutput(
            text="종합 평가는 긍정 신호와 추가 검증 포인트가 공존하는 단계다. 현시점에서는 관찰 대상으로 두고 후속 검증을 이어가는 접근이 적절하다.",
            source_ids=["SRC003", "SRC005"],
        ),
        final_investment_judgment=ReportFieldOutput(
            text="최종 판단은 watch다. 후속 검증을 전제로 투자 검토를 이어갈 수 있다.",
            source_ids=["SRC003", "SRC005"],
        ),
    )


class ReportCommonTests(unittest.TestCase):
    def test_get_chat_model_uses_gpt4o_default(self) -> None:
        report_common.get_chat_model.cache_clear()
        try:
            with (
                patch("agents.agent_report.common.require_env", return_value="test-key"),
                patch("langchain_openai.ChatOpenAI") as chat_model_cls,
            ):
                report_common.get_chat_model()
        finally:
            report_common.get_chat_model.cache_clear()

        chat_model_cls.assert_called_once()
        self.assertEqual(chat_model_cls.call_args.kwargs["model"], "gpt-4o")


class ReportServiceTests(unittest.TestCase):
    def test_build_report_state_generates_markdown_pdf_and_used_references_only(self) -> None:
        model = Mock()
        writer = Mock()
        writer.invoke.return_value = build_llm_draft()
        model.with_structured_output.return_value = writer

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
                patch("agents.agent_report.service.get_chat_model", return_value=model),
            ):
                result = build_report_state(
                    selected_company=build_selected_company(),
                    force_report_generation=False,
                    company_search_summary="검색 요약",
                    selected_company_reason="선정 이유",
                    investigate_members_state=build_investigate_members_state(),
                    agent_product_market_analysis_state=build_product_market_state(),
                    traction_state=build_traction_state(),
                    agent_risk_search_state=build_risk_state(),
                    review_state=build_review_state(),
                    eval_state=build_eval_state(),
                )
                self.assertEqual(result["status"], "completed")
                self.assertTrue(Path(result["report_path"]).exists())
                self.assertTrue(Path(result["pdf_path"]).exists())
                self.assertIn("## SUMMARY (Executive Summary)", result["markdown"])
                self.assertIn("## REFERENCE", result["markdown"])
                self.assertIn("### 기관 보고서", result["markdown"])
                self.assertIn("### 학술 논문", result["markdown"])
                self.assertIn("### 웹페이지", result["markdown"])
                self.assertIn("글로벌 로보틱스 시장 전망", result["markdown"])
                self.assertIn("Robotics Benchmarking", result["markdown"])
                self.assertIn("파트너십 기사", result["markdown"])
                self.assertNotIn("입니다.", result["markdown"])
                self.assertNotIn("합니다.", result["markdown"])

                summary_block = result["markdown"].split("## SUMMARY (Executive Summary)\n", 1)[1]
                summary_text = summary_block.split("\n## 1. 기업 개요", 1)[0].strip()
                summary_paragraphs = [paragraph for paragraph in summary_text.split("\n\n") if paragraph.strip()]
                self.assertLessEqual(len(summary_text), SUMMARY_MAX_CHARS)
                self.assertGreaterEqual(len(summary_text), 120)
                self.assertGreaterEqual(len(summary_paragraphs), 2)
                self.assertLessEqual(len(summary_paragraphs), 4)

    def test_build_report_state_falls_back_when_llm_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
                patch("agents.agent_report.service.get_chat_model", side_effect=RuntimeError("boom")),
            ):
                result = build_report_state(
                    selected_company=build_selected_company(),
                    force_report_generation=False,
                    company_search_summary="검색 요약",
                    selected_company_reason="선정 이유",
                    investigate_members_state=build_investigate_members_state(),
                    agent_product_market_analysis_state=build_product_market_state(),
                    traction_state=build_traction_state(),
                    agent_risk_search_state=build_risk_state(),
                    review_state=build_review_state(),
                    eval_state=build_eval_state(),
                )
                self.assertEqual(result["status"], "completed")
                self.assertTrue(Path(result["report_path"]).exists())
                self.assertTrue(Path(result["pdf_path"]).exists())
                self.assertIn("## 6. 종합 투자 의견 및 결론", result["markdown"])
                self.assertIn("## REFERENCE", result["markdown"])
                self.assertNotIn("입니다.", result["markdown"])
                self.assertNotIn("합니다.", result["markdown"])
                summary_block = result["markdown"].split("## SUMMARY (Executive Summary)\n", 1)[1]
                summary_text = summary_block.split("\n## 1. 기업 개요", 1)[0].strip()
                summary_paragraphs = [paragraph for paragraph in summary_text.split("\n\n") if paragraph.strip()]
                self.assertGreaterEqual(len(summary_paragraphs), 2)
                self.assertLessEqual(len(summary_paragraphs), 4)

    def test_fallback_report_does_not_promote_unreferenced_product_market_claims(self) -> None:
        product_market_state = build_product_market_state()
        product_market_state["summary"] = "검증된 유료 고객 50곳을 확보했다."
        product_market_state["structured_output"] = {
            "target_kpi_logic": {
                "text": "유료 고객 50곳이 이미 도입해 ROI가 확정됐다.",
                "references": [],
                "evidence_gap": "공개 근거가 부족하다.",
            },
            "technical_moat": {
                "text": "업계 최고 수준의 기술 우위를 확보했다.",
                "references": [],
                "evidence_gap": "공개 근거가 부족하다.",
            },
            "data_loop_structure": {
                "text": "독점 데이터 루프가 완성됐다.",
                "references": [],
                "evidence_gap": "공개 근거가 부족하다.",
            },
            "product_summary": {
                "text": "유료 고객 50곳을 기반으로 빠르게 확산 중이다.",
                "references": [],
                "evidence_gap": "공개 근거가 부족하다.",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
                patch("agents.agent_report.service.get_chat_model", side_effect=RuntimeError("boom")),
            ):
                result = build_report_state(
                    selected_company=build_selected_company(),
                    force_report_generation=False,
                    company_search_summary="검색 요약",
                    selected_company_reason="선정 이유",
                    investigate_members_state=build_investigate_members_state(),
                    agent_product_market_analysis_state=product_market_state,
                    traction_state=build_traction_state(),
                    agent_risk_search_state=build_risk_state(),
                    review_state=build_review_state(),
                    eval_state=build_eval_state(),
                )

        self.assertNotIn("유료 고객 50곳", result["markdown"])
        self.assertNotIn("독점 데이터 루프가 완성됐다", result["markdown"])
        self.assertNotIn("글로벌 로보틱스 시장 전망", result["markdown"])
        self.assertNotIn("Robotics Benchmarking", result["markdown"])

    def test_build_report_state_skips_when_not_ready(self) -> None:
        result = build_report_state(
            selected_company=build_selected_company(),
            force_report_generation=False,
            company_search_summary="검색 요약",
            selected_company_reason="선정 이유",
            investigate_members_state=build_investigate_members_state(),
            agent_product_market_analysis_state=build_product_market_state(),
            traction_state=build_traction_state(),
            agent_risk_search_state=build_risk_state(),
            review_state=build_review_state(),
            eval_state=build_eval_state(ready_for_report=False),
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["report_path"], "")
        self.assertEqual(result["pdf_path"], "")
        self.assertEqual(result["markdown"], "")

    def test_build_report_state_generates_when_forced_even_if_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_root = Path(temp_dir) / "outputs" / "reports"
            with (
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
                patch("agents.agent_report.service.get_chat_model", side_effect=RuntimeError("boom")),
            ):
                result = build_report_state(
                    selected_company=build_selected_company(),
                    force_report_generation=True,
                    company_search_summary="검색 요약",
                    selected_company_reason="선정 이유",
                    investigate_members_state=build_investigate_members_state(),
                    agent_product_market_analysis_state=build_product_market_state(),
                    traction_state=build_traction_state(),
                    agent_risk_search_state=build_risk_state(),
                    review_state=build_review_state(),
                    eval_state=build_eval_state(ready_for_report=False),
                )
                self.assertEqual(result["status"], "completed")
                self.assertTrue(Path(result["report_path"]).exists())
                self.assertTrue(Path(result["pdf_path"]).exists())
                self.assertIn("- 보고서 모드: 강제 생성", result["markdown"])


if __name__ == "__main__":
    unittest.main()
