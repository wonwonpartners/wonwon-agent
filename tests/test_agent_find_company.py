from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.agent_find_company.input import FindCompanySearchInput
from agents.agent_find_company.node import find_company_node
from agents.agent_find_company.service import run_search


class FindCompanyTests(unittest.TestCase):
    def test_run_search_passes_excluded_company_ids(self) -> None:
        with (
            patch("agents.agent_find_company.service.get_engine", return_value=object()),
            patch(
                "agents.agent_find_company.service.search_companies_query",
                return_value=[],
            ) as search_mock,
        ):
            run_search(
                FindCompanySearchInput(query="뉴빌리티", limit=5),
                excluded_company_ids=["CP001", "CP002"],
            )

        self.assertEqual(
            search_mock.call_args.kwargs["excluded_company_ids"],
            ["CP001", "CP002"],
        )

    def test_find_company_node_uses_evaluated_company_ids_as_exclusions(self) -> None:
        with (
            patch(
                "agents.agent_find_company.node.parse_search_query",
                return_value=FindCompanySearchInput(query="뉴빌리티", limit=5),
            ),
            patch(
                "agents.agent_find_company.node.run_search",
                return_value={
                    "results": [],
                    "applied_filters": {"query": "뉴빌리티"},
                    "summary": "0개의 회사 후보를 찾았습니다.",
                },
            ) as run_search_mock,
        ):
            find_company_node(
                {
                    "user_query": "뉴빌리티",
                    "evaluated_company_ids": ["CP_OLD"],
                }
            )

        self.assertEqual(
            run_search_mock.call_args.kwargs["excluded_company_ids"],
            ["CP_OLD"],
        )


if __name__ == "__main__":
    unittest.main()
