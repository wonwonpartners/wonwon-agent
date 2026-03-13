from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from agents.agent_traction.node import traction_node
from agents.agent_traction.service import TractionAgent
from tools.tools import ToolDocument


class TractionServiceTests(unittest.IsolatedAsyncioTestCase):
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
