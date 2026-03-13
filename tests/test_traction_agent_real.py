import json
import unittest

from agents.agent_traction import TractionAgent
from langchain_openai import ChatOpenAI
from state import TractionInputState
from tools import FirecrawlTractionSearchTool, TractionWebVectorTool, VectorTractionSearchTool


class TractionAgentRealTest(unittest.IsolatedAsyncioTestCase):
    async def test_newility_real_search_returns_normalized_traction_state(self) -> None:
        agent = TractionAgent(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            tool=TractionWebVectorTool(
                vector_tool=VectorTractionSearchTool(),
                web_tool=FirecrawlTractionSearchTool(),
            ),
        )
        state: TractionInputState = {"startup_name": "뉴빌리티"}
        startup_name = state["startup_name"]

        queries = await agent._build_queries(startup_name)
        vector_contexts = await agent._query_signal_contexts(startup_name, queries, mode="vector")
        vector_context_text = agent._join_context_blocks(
            startup_name=startup_name,
            queries=queries,
            contexts=vector_contexts,
            channel_label="vector",
        )
        is_sufficient, sufficiency_reason, missing_signals = await agent._assess_vector_sufficiency(
            startup_name=startup_name,
            queries=queries,
            vector_contexts=vector_contexts,
            vector_context_text=vector_context_text,
        )

        merged_contexts = dict(vector_contexts)
        if not is_sufficient:
            target_queries = {
                signal_type: queries[signal_type]
                for signal_type in (missing_signals or list(queries.keys()))
                if signal_type in queries
            }
            web_contexts = await agent._query_signal_contexts(startup_name, target_queries, mode="web")
            merged_contexts = agent._merge_context_maps(merged_contexts, web_contexts)

        merged_context_text = agent._join_context_blocks(
            startup_name=startup_name,
            queries=queries,
            contexts=merged_contexts,
            channel_label="merged",
        )
        quality_ok, quality_reason, low_quality_signals = await agent._assess_search_quality(
            startup_name=startup_name,
            queries=queries,
            contexts=merged_contexts,
            context_text=merged_context_text,
        )

        retry_queries = {}
        if not quality_ok:
            retry_queries = agent._build_retry_queries(
                startup_name=startup_name,
                signal_types=low_quality_signals or list(queries.keys()),
                quality_reason=quality_reason,
            )
            retry_contexts = await agent._query_signal_contexts(startup_name, retry_queries, mode="web")
            merged_contexts = agent._merge_context_maps(merged_contexts, retry_contexts)
            merged_context_text = agent._join_context_blocks(
                startup_name=startup_name,
                queries={**queries, **retry_queries},
                contexts=merged_contexts,
                channel_label="merged",
            )
            quality_ok, quality_reason, low_quality_signals = await agent._assess_search_quality(
                startup_name=startup_name,
                queries=queries,
                contexts=merged_contexts,
                context_text=merged_context_text,
            )

        final_is_sufficient, final_sufficiency_reason, final_missing_signals = agent._heuristic_vector_sufficiency(
            queries=queries,
            vector_contexts=merged_contexts,
        )

        result = await agent(state)

        print(json.dumps(result, ensure_ascii=False, indent=2))

        self.assertIsInstance(result, dict)
        self.assertIn(
            "traction",
            result,
            json.dumps(
                {
                    "final_sufficiency": final_is_sufficient,
                    "final_sufficiency_reason": final_sufficiency_reason,
                    "final_missing_signals": final_missing_signals,
                    "initial_sufficiency": is_sufficient,
                    "initial_sufficiency_reason": sufficiency_reason,
                    "initial_missing_signals": missing_signals,
                    "quality_ok": quality_ok,
                    "quality_reason": quality_reason,
                    "low_quality_signals": low_quality_signals,
                    "retry_queries": retry_queries,
                    "result": result,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        traction = result["traction"]
        self.assertIsInstance(traction["partnerships"], list)
        self.assertGreater(len(traction["partnerships"]), 0)
        self.assertIsInstance(traction["hiring_analysis"], dict)
        self.assertIn("field_engineer_ratio", traction["hiring_analysis"])
        self.assertIn("field_engineer_count", traction["hiring_analysis"])
        self.assertIn("hiring_trend_3m", traction["hiring_analysis"])
        self.assertIsInstance(traction["funding_velocity"], list)
        self.assertGreater(len(traction["funding_velocity"]), 0)
        self.assertIsInstance(traction["traction_summary"], str)
        self.assertTrue(traction["traction_summary"].strip())


if __name__ == "__main__":
    unittest.main()
