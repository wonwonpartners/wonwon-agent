from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from agents.agent_traction.node import traction_node
from agents.agent_traction.service import TractionAgent
from tools import ToolDocument


class TractionAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_promotes_tooldocument_results_to_evidence_sources(self) -> None:
        agent = TractionAgent(llm=Mock(), tool=Mock())
        query_contexts = {
            "partnership": ToolDocument(
                content="파트너십 근거",
                source="firecrawl",
                metadata={
                    "query": '"테스트로보틱스" 파트너십',
                    "results": [
                        {
                            "title": "파트너십 기사",
                            "source": "https://traction.example.com/partnership",
                            "source_type": "web",
                            "published_at": "2026-03-09",
                            "url": "https://traction.example.com/partnership",
                            "score": 0.91,
                        }
                    ],
                },
            )
        }
        agent._build_queries = AsyncMock(return_value={"partnership": '"테스트로보틱스" 파트너십'})
        agent._query_signal_contexts = AsyncMock(return_value=query_contexts)
        agent._assess_vector_sufficiency = AsyncMock(return_value=(True, "충분", []))
        agent._assess_search_quality = AsyncMock(return_value=(True, "양호", []))

        with patch(
            "agents.agent_traction.service.run_structured_from_llm",
            new=AsyncMock(
                return_value={
                    "partnerships": ["물류사와 파트너십"],
                    "hiring_analysis": {
                        "field_engineer_ratio": 0.2,
                        "field_engineer_count": 2,
                        "hiring_trend_3m": 1,
                    },
                    "funding_velocity": ["시리즈 A 투자"],
                    "traction_summary": "상용화 신호가 있습니다.",
                }
            ),
        ):
            result = await agent({"startup_name": "테스트로보틱스"})

        self.assertIn("traction", result)
        evidence_sources = result["traction"]["evidence_sources"]
        self.assertEqual(len(evidence_sources), 1)
        self.assertEqual(
            evidence_sources[0]["url"],
            "https://traction.example.com/partnership",
        )
        self.assertEqual(evidence_sources[0]["signal_type"], "partnership")

    async def test_uses_heuristic_sufficiency_when_llm_check_is_unavailable(self) -> None:
        agent = TractionAgent(llm=Mock(), tool=Mock())
        contexts = {
            "hiring": ToolDocument(
                content="뉴빌리티 hiring signal",
                source="vector",
                metadata={"results": [{"url": "https://example.com/hiring"}]},
            ),
            "funding": ToolDocument(
                content="뉴빌리티 funding signal",
                source="vector",
                metadata={"results": [{"url": "https://example.com/funding"}]},
            ),
            "partnership": ToolDocument(
                content="뉴빌리티 partnership signal",
                source="vector",
                metadata={"results": [{"url": "https://example.com/partnership"}]},
            ),
            "customer": ToolDocument(
                content="근거가 부족",
                source="vector_rag_empty",
                metadata={"results": []},
            ),
        }

        with patch(
            "agents.agent_traction.service.run_structured_from_llm",
            AsyncMock(return_value=None),
        ):
            is_sufficient, reason, missing_signals = await agent._assess_vector_sufficiency(
                startup_name="뉴빌리티",
                queries={key: key for key in contexts},
                vector_contexts=contexts,
                vector_context_text="context",
            )

        self.assertTrue(is_sufficient)
        self.assertIn("vector 신호 3개 확보", reason)
        self.assertEqual(missing_signals, ["customer"])

    async def test_uses_heuristic_quality_when_llm_check_is_unavailable(self) -> None:
        agent = TractionAgent(llm=Mock(), tool=Mock())
        contexts = {
            "hiring": ToolDocument(
                content="뉴빌리티 채용 정보",
                source="merged",
                metadata={"results": [{"url": "https://example.com/hiring"}]},
            ),
            "funding": ToolDocument(
                content="무관한 문서",
                source="merged",
                metadata={"results": [{"url": "https://example.com/funding"}]},
            ),
        }

        with patch(
            "agents.agent_traction.service.run_structured_from_llm",
            AsyncMock(return_value=None),
        ):
            is_acceptable, reason, low_quality_signals = await agent._assess_search_quality(
                startup_name="뉴빌리티",
                queries={key: key for key in contexts},
                contexts=contexts,
                context_text="context",
            )

        self.assertFalse(is_acceptable)
        self.assertIn("funding", low_quality_signals)
        self.assertIn("기업 관련성", reason)

    async def test_normalizes_llm_output_before_validation(self) -> None:
        agent = TractionAgent(llm=Mock(), tool=Mock())
        tool = agent.tool
        tool.query_traction_vector = AsyncMock(
            side_effect=[
                ToolDocument(
                    content="뉴빌리티 파트너십과 투자, 월 2건 채용",
                    source="vector",
                    metadata={"results": [{"url": "https://example.com/a"}]},
                ),
                ToolDocument(
                    content="뉴빌리티 파트너십과 투자, 월 2건 채용",
                    source="vector",
                    metadata={"results": [{"url": "https://example.com/b"}]},
                ),
                ToolDocument(
                    content="뉴빌리티 파트너십과 투자, 월 2건 채용",
                    source="vector",
                    metadata={"results": [{"url": "https://example.com/c"}]},
                ),
                ToolDocument(
                    content="뉴빌리티 파트너십과 투자, 월 2건 채용",
                    source="vector",
                    metadata={"results": [{"url": "https://example.com/d"}]},
                ),
            ]
        )

        with patch(
            "agents.agent_traction.service.run_structured_from_llm",
            AsyncMock(
                side_effect=[
                    {
                        "is_sufficient": True,
                        "reason": "충분",
                        "missing_signals": [],
                    },
                    {
                        "is_acceptable": True,
                        "reason": "양호",
                        "low_quality_signals": [],
                    },
                    {
                        "is_sufficient": True,
                        "reason": "충분",
                        "missing_signals": [],
                    },
                    {
                        "partnerships": ["뉴빌리티 파트너십"],
                        "hiring_analysis": {
                            "field_engineer_ratio": 0.2,
                            "field_engineer_count": 2,
                            "hiring_trend_3m": 6,
                        },
                        "funding_velocity": [],
                        "traction_summary": "뉴빌리티 traction 요약",
                    },
                ]
            ),
        ):
            result = await agent({"startup_name": "뉴빌리티"})

        self.assertIn("traction", result)
        self.assertTrue(result["traction"]["funding_velocity"])

    def test_retry_queries_do_not_append_quality_reason_text(self) -> None:
        agent = TractionAgent(llm=None, tool=Mock())

        retry_queries = agent._build_retry_queries(
            startup_name="뉴빌리티",
            signal_types=["funding"],
            quality_reason="무관 문서가 많음",
        )

        self.assertIn("funding", retry_queries)
        self.assertNotIn("품질 보강 목적", retry_queries["funding"])

    def test_extract_hiring_trend_interprets_monthly_signal_as_three_month_estimate(self) -> None:
        agent = TractionAgent(llm=None, tool=Mock())

        self.assertEqual(agent._extract_hiring_trend("월 2건 채용"), 6)
        self.assertEqual(agent._extract_hiring_trend("최근 3개월 5건 채용"), 5)


class TractionNodeTests(unittest.TestCase):
    def test_node_uses_evidence_sources_as_research_sources(self) -> None:
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
                "score": 0.91,
            }
        ]

        class DummyTractionAgent:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            async def __call__(self, state: dict[str, str]) -> dict[str, object]:
                del state
                return {
                    "traction": {
                        "partnerships": ["파트너십 1건"],
                        "hiring_analysis": {
                            "field_engineer_ratio": 0.2,
                            "field_engineer_count": 2,
                            "hiring_trend_3m": 1,
                        },
                        "funding_velocity": ["시리즈 A 투자"],
                        "traction_summary": "상용화 신호가 있습니다.",
                        "evidence_sources": evidence_sources,
                    }
                }

        with (
            patch(
                "agents.agent_traction.node.get_chat_model",
                return_value=Mock(),
            ),
            patch(
                "agents.agent_traction.node.TractionAgent",
                DummyTractionAgent,
            ),
        ):
            result = traction_node(
                {
                    "selected_company": {
                        "company_id": "CP_TRACTION",
                        "company_name": "테스트로보틱스",
                    }
                }
            )

        self.assertEqual(
            result["traction_state"]["sources"],
            evidence_sources,
        )
        self.assertEqual(
            result["traction_state"]["structured_output"]["evidence_sources"],
            evidence_sources,
        )

if __name__ == "__main__":
    unittest.main()
