from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from utils._shared import run_structured_from_llm


class OpenAIFallbackTests(unittest.TestCase):
    def test_run_structured_from_llm_uses_fallback_model_on_rate_limit(self) -> None:
        primary_runnable = Mock()
        primary_runnable.ainvoke = AsyncMock(
            side_effect=RuntimeError("429 rate limit exceeded")
        )
        primary_llm = Mock()
        primary_llm.temperature = 0
        primary_llm.with_structured_output.return_value = primary_runnable

        fallback_runnable = Mock()
        fallback_runnable.ainvoke = AsyncMock(return_value={"summary": "fallback"})
        fallback_llm = Mock()
        fallback_llm.with_structured_output.return_value = fallback_runnable

        with (
            patch("utils._shared.build_chat_model", return_value=fallback_llm),
            patch(
                "utils._shared.get_fallback_openai_model_name",
                return_value="gpt-4.1-nano",
            ),
        ):
            result = asyncio.run(run_structured_from_llm(primary_llm, dict, "prompt"))

        self.assertEqual(result, {"summary": "fallback"})
        fallback_runnable.ainvoke.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
