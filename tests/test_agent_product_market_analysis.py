from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from agents.agent_product_market_analysis.node import (
    product_market_analysis_node,
)
from agents.agent_product_market_analysis.result import (
    ProductMarketAnalysisResult,
)
from agents.agent_product_market_analysis.service import (
    normalize_result_references,
    render_available_sources,
    run_product_market_analysis,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)


def build_selected_company() -> dict[str, object]:
    return {
        "company_id": "CP_AGENT_PM",
        "company_name": "테스트컴퍼니",
        "product_name": "테스트 제품",
        "description": "로봇 자동화 솔루션 기업",
    }


def build_result() -> ProductMarketAnalysisResult:
    return ProductMarketAnalysisResult(
        target_kpi_logic={
            "text": "물류 자동화 인건비 절감과 처리량 개선이 핵심이다.",
            "references": ["(2026-03-12), 테스트 문서, 테스트 발행처, https://example.com/doc"],
            "evidence_gap": "",
        },
        technical_moat={
            "text": "현장 데이터 축적과 통합 난이도가 진입장벽이다.",
            "references": ["(2026-03-12), 테스트 문서, 테스트 발행처, https://example.com/doc"],
            "evidence_gap": "",
        },
        data_loop_structure={
            "text": "운영 데이터가 모델 개선에 일부 기여하나 범위는 제한적이다.",
            "references": ["(2026-03-12), 테스트 문서, 테스트 발행처, https://example.com/doc"],
            "evidence_gap": "",
        },
        product_summary={
            "text": "초기 PMF 신호는 있으나 해자의 지속성은 추가 검증이 필요하다.",
            "references": ["(2026-03-12), 테스트 문서, 테스트 발행처, https://example.com/doc"],
            "evidence_gap": "",
        },
    )


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY"))


def has_tavily_key() -> bool:
    return bool(os.getenv("TAVILY_API_KEY") or os.getenv("TRAVILY_API_KEY"))


class ProductMarketAnalysisServiceTests(unittest.TestCase):
    def test_skips_when_selected_company_is_missing(self) -> None:
        result = run_product_market_analysis(None)

        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["structured_output"])

    def test_completes_when_research_and_writer_succeed(self) -> None:
        expected_result = build_result()
        with (
            patch(
                "agents.agent_product_market_analysis.service.run_product_market_research",
                return_value=(
                    "[tool]\nresearch notes",
                    [
                        {
                            "source_type": "rag_document",
                            "tool_name": "domain_rag_search_tool",
                            "title": "테스트 문서",
                            "publisher": "테스트 발행처",
                            "published_at": "2026-03-12",
                            "url": "https://example.com/doc",
                        }
                    ],
                ),
            ),
            patch(
                "agents.agent_product_market_analysis.service.write_product_market_result",
                return_value=expected_result,
            ) as writer_mock,
        ):
            result = run_product_market_analysis(build_selected_company())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["input_company_id"], "CP_AGENT_PM")
        self.assertEqual(
            result["structured_output"]["product_summary"],
            expected_result.product_summary.model_dump(),
        )
        self.assertEqual(
            result["structured_output"]["product_summary"]["references"],
            expected_result.product_summary.references,
        )
        self.assertEqual(len(result["findings"]), 4)
        writer_mock.assert_called_once()
        self.assertEqual(
            writer_mock.call_args.args[2],
            [
                {
                    "source_type": "rag_document",
                    "tool_name": "domain_rag_search_tool",
                    "title": "테스트 문서",
                    "publisher": "테스트 발행처",
                    "published_at": "2026-03-12",
                    "url": "https://example.com/doc",
                }
            ],
        )

    def test_fails_gracefully_when_research_raises(self) -> None:
        with patch(
            "agents.agent_product_market_analysis.service.run_product_market_research",
            side_effect=RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다."),
        ):
            result = run_product_market_analysis(build_selected_company())

        self.assertEqual(result["status"], "failed")
        self.assertIn("오류 메시지", " ".join(result["findings"]))
        self.assertIn("실행 오류", result["structured_output"]["product_summary"]["text"])
        self.assertEqual(result["structured_output"]["product_summary"]["references"], [])

    def test_normalize_result_references_excludes_sources_without_url(self) -> None:
        result = build_result()
        result.product_summary.references = [
            "(2026-03-12), URL 있는 문서, 테스트 발행처, https://example.com/doc",
            "(2026-03-11), URL 없는 문서, 다른 발행처",
        ]
        normalized = normalize_result_references(
            result,
            [
                {
                    "source_type": "rag_document",
                    "tool_name": "domain_rag_search_tool",
                    "title": "URL 있는 문서",
                    "publisher": "테스트 발행처",
                    "published_at": "2026-03-12",
                    "url": "https://example.com/doc",
                },
                {
                    "source_type": "rag_document",
                    "tool_name": "domain_rag_search_tool",
                    "title": "URL 없는 문서",
                    "publisher": "다른 발행처",
                    "published_at": "2026-03-11",
                },
            ],
        )

        self.assertEqual(
            normalized.product_summary.references,
            ["(2026-03-12), URL 있는 문서, 테스트 발행처, https://example.com/doc"],
        )

    def test_render_available_sources_excludes_sources_without_url(self) -> None:
        rendered = render_available_sources(
            [
                {
                    "source_type": "rag_document",
                    "tool_name": "domain_rag_search_tool",
                    "title": "URL 있는 문서",
                    "publisher": "테스트 발행처",
                    "published_at": "2026-03-12",
                    "url": "https://example.com/doc",
                    "excerpt": "본문",
                },
                {
                    "source_type": "rag_document",
                    "tool_name": "domain_rag_search_tool",
                    "title": "URL 없는 문서",
                    "publisher": "다른 발행처",
                    "published_at": "2026-03-11",
                    "excerpt": "본문",
                },
            ]
        )

        self.assertIn("https://example.com/doc", rendered)
        self.assertNotIn("URL 없는 문서", rendered)

    def test_render_available_sources_excludes_url_only_sources(self) -> None:
        rendered = render_available_sources(
            [
                {
                    "source_type": "web_page",
                    "tool_name": "web_page_extract_tool",
                    "url": "https://example.com/url-only",
                    "excerpt": "본문",
                },
                {
                    "source_type": "web_search",
                    "tool_name": "web_benchmark_search_tool",
                    "title": "정상 문서",
                    "publisher": "테스트 발행처",
                    "published_at": "2026-03-12",
                    "url": "https://example.com/doc",
                    "excerpt": "본문",
                },
            ]
        )

        self.assertIn("정상 문서", rendered)
        self.assertNotIn("https://example.com/url-only", rendered)


class ProductMarketAnalysisNodeTests(unittest.TestCase):
    def test_node_reads_selected_company_and_previous_state(self) -> None:
        previous_state = {
            "agent_name": "agent_product_market_analysis",
            "status": "failed",
            "attempt_count": 1,
            "input_company_id": "CP_AGENT_PM",
            "summary": "이전 실행",
            "findings": ["이전 findings"],
            "sources": [],
            "structured_output": None,
        }
        expected_payload = {
            "agent_name": "agent_product_market_analysis",
            "status": "completed",
            "attempt_count": 2,
            "input_company_id": "CP_AGENT_PM",
            "summary": "새 실행",
            "findings": ["새 findings"],
            "sources": [{"source_type": "rag_document", "title": "테스트 문서"}],
            "structured_output": {
                "product_summary": {
                    "text": "요약",
                    "references": ["[1] 테스트 문서"],
                    "evidence_gap": "",
                },
            },
        }

        with patch(
            "agents.agent_product_market_analysis.node.run_product_market_analysis",
            return_value=expected_payload,
        ) as run_mock:
            result = product_market_analysis_node(
                {
                    "selected_company": build_selected_company(),
                    "agent_product_market_analysis_state": previous_state,
                }
            )

        run_mock.assert_called_once_with(
            build_selected_company(),
            previous_state,
        )
        self.assertEqual(
            result,
            {"agent_product_market_analysis_state": expected_payload},
        )

    def test_node_passes_none_when_previous_state_is_missing(self) -> None:
        expected_payload = {
            "agent_name": "agent_product_market_analysis",
            "status": "skipped",
            "attempt_count": 1,
            "input_company_id": None,
            "summary": "건너뜀",
            "findings": ["selected_company 없음"],
            "sources": [],
            "structured_output": None,
        }

        with patch(
            "agents.agent_product_market_analysis.node.run_product_market_analysis",
            return_value=expected_payload,
        ) as run_mock:
            result = product_market_analysis_node({})

        run_mock.assert_called_once_with(None, None)
        self.assertEqual(
            result["agent_product_market_analysis_state"]["status"],
            "skipped",
        )


@unittest.skipUnless(
    has_openai_key() and has_tavily_key(),
    "Live integration test requires OPENAI_API_KEY/OPEN_AI_KEY and TAVILY_API_KEY/TRAVILY_API_KEY.",
)
class ProductMarketAnalysisLiveTests(unittest.TestCase):
    def test_live_node_executes_with_real_external_dependencies(self) -> None:
        result = product_market_analysis_node(
            {
                "selected_company": {
                    "company_id": "CP00001693",
                    "company_name": "뉴빌리티",
                    "product_name": "뉴비",
                    "description": "카메라 기반 자율주행 로봇",
                }
            }
        )
        agent_state = result["agent_product_market_analysis_state"]

        self.assertEqual(agent_state["agent_name"], "agent_product_market_analysis")
        self.assertEqual(agent_state["input_company_id"], "CP00001693")
        self.assertEqual(agent_state["status"], "completed")
        self.assertTrue(agent_state["summary"])
        self.assertEqual(len(agent_state["findings"]), 4)
        self.assertTrue(agent_state["sources"])
        self.assertIsInstance(agent_state["structured_output"], dict)
        self.assertIn("target_kpi_logic", agent_state["structured_output"])
        self.assertIn("technical_moat", agent_state["structured_output"])
        self.assertIn("data_loop_structure", agent_state["structured_output"])
        self.assertIn("product_summary", agent_state["structured_output"])
        self.assertIn("references", agent_state["structured_output"]["target_kpi_logic"])
        self.assertIn("references", agent_state["structured_output"]["technical_moat"])
        self.assertIn("references", agent_state["structured_output"]["data_loop_structure"])
        self.assertIn("references", agent_state["structured_output"]["product_summary"])


if __name__ == "__main__":
    unittest.main()
