from __future__ import annotations

import unittest

from agents.agent_risk_search.prompts import get_system_prompt


class RiskPromptTests(unittest.TestCase):
    def test_system_prompt_renders_json_schema_without_format_error(self) -> None:
        prompt = get_system_prompt()

        self.assertIn('"legal_regulatory"', prompt)
        self.assertIn('"risk_summary"', prompt)


if __name__ == "__main__":
    unittest.main()
