from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agents.agent_find_company.input import FindCompanySearchInput
from agents.agent_find_company.node import find_company_node
from agents.agent_find_company.result import CompanySelectionResult
from agents.agent_find_company.service import pick_company


def build_search_input() -> FindCompanySearchInput:
    return FindCompanySearchInput(query="로봇 자동화", limit=8)


def build_candidate(
    *,
    company_id: str,
    company_name: str,
) -> dict[str, object]:
    return {
        "company_id": company_id,
        "company_name": company_name,
        "product_name": f"{company_name} 제품",
        "description": f"{company_name} 설명",
        "employees": 42,
        "revenue": None,
        "invest_count": 1,
        "invest_level": "seed",
        "hiring": False,
        "categories": ["robotics"],
        "keywords": ["automation"],
    }


class FindCompanyServiceTests(unittest.TestCase):
    def test_pick_company_returns_llm_selection(self) -> None:
        candidates = [
            build_candidate(company_id="CP_1", company_name="알파"),
            build_candidate(company_id="CP_2", company_name="베타"),
        ]
        selector = MagicMock()
        selector.invoke.return_value = CompanySelectionResult(
            company_id="CP_2",
            reason="제품 방향성이 질의와 더 직접적으로 맞습니다.",
        )

        with patch(
            "agents.agent_find_company.service.get_chat_model",
            return_value=MagicMock(with_structured_output=MagicMock(return_value=selector)),
        ):
            selection = pick_company("로봇 회사", candidates)

        self.assertEqual(selection.company_id, "CP_2")
        self.assertIn("질의", selection.reason)

    def test_pick_company_falls_back_to_first_candidate_on_exception(self) -> None:
        candidates = [
            build_candidate(company_id="CP_1", company_name="알파"),
            build_candidate(company_id="CP_2", company_name="베타"),
        ]
        selector = MagicMock()
        selector.invoke.side_effect = RuntimeError("LLM unavailable")

        with patch(
            "agents.agent_find_company.service.get_chat_model",
            return_value=MagicMock(with_structured_output=MagicMock(return_value=selector)),
        ):
            selection = pick_company("로봇 회사", candidates)

        self.assertEqual(selection.company_id, "CP_1")
        self.assertIn("첫 번째 후보", selection.reason)


class FindCompanyNodeTests(unittest.TestCase):
    def test_returns_none_when_search_result_is_empty(self) -> None:
        search_input = build_search_input()
        with (
            patch(
                "agents.agent_find_company.node.parse_search_query",
                return_value=search_input,
            ),
            patch(
                "agents.agent_find_company.node.run_search",
                return_value={
                    "results": [],
                    "applied_filters": search_input.model_dump(),
                },
            ),
        ):
            result = find_company_node({"user_query": "로봇 회사"})

        self.assertIsNone(result["selected_company"])
        self.assertEqual(result["selected_company_reason"], "검색 결과가 없습니다.")

    def test_selects_only_candidate_without_llm_selection(self) -> None:
        search_input = build_search_input()
        only_candidate = build_candidate(company_id="CP_ONLY", company_name="원컴퍼니")

        with (
            patch(
                "agents.agent_find_company.node.parse_search_query",
                return_value=search_input,
            ),
            patch(
                "agents.agent_find_company.node.run_search",
                return_value={
                    "results": [only_candidate],
                    "applied_filters": search_input.model_dump(),
                },
            ),
            patch(
                "agents.agent_find_company.node.pick_company",
            ) as pick_company_mock,
        ):
            result = find_company_node({"user_query": "로봇 회사"})

        pick_company_mock.assert_not_called()
        self.assertEqual(result["selected_company"], only_candidate)
        self.assertEqual(
            result["selected_company_reason"],
            "검색 결과가 1건이라 해당 회사를 바로 선택했습니다.",
        )

    def test_selects_candidate_chosen_by_llm_when_multiple_candidates_exist(self) -> None:
        search_input = build_search_input()
        candidates = [
            build_candidate(company_id="CP_1", company_name="알파"),
            build_candidate(company_id="CP_2", company_name="베타"),
            build_candidate(company_id="CP_3", company_name="감마"),
        ]
        selection = CompanySelectionResult(
            company_id="CP_2",
            reason="베타가 사용자 질의와 가장 잘 맞습니다.",
        )

        with (
            patch(
                "agents.agent_find_company.node.parse_search_query",
                return_value=search_input,
            ),
            patch(
                "agents.agent_find_company.node.run_search",
                return_value={
                    "results": candidates,
                    "applied_filters": search_input.model_dump(),
                },
            ),
            patch(
                "agents.agent_find_company.node.pick_company",
                return_value=selection,
            ) as pick_company_mock,
        ):
            result = find_company_node({"user_query": "로봇 회사"})

        pick_company_mock.assert_called_once_with("로봇 회사", candidates)
        self.assertEqual(result["selected_company"], candidates[1])
        self.assertEqual(result["selected_company_reason"], selection.reason)
        self.assertIn("베타", result["company_search_summary"])

    def test_falls_back_to_first_candidate_when_llm_selection_does_not_match(self) -> None:
        search_input = build_search_input()
        candidates = [
            build_candidate(company_id="CP_1", company_name="알파"),
            build_candidate(company_id="CP_2", company_name="베타"),
        ]
        selection = CompanySelectionResult(
            company_id="CP_UNKNOWN",
            reason="목록 밖 id를 잘못 반환한 케이스",
        )

        with (
            patch(
                "agents.agent_find_company.node.parse_search_query",
                return_value=search_input,
            ),
            patch(
                "agents.agent_find_company.node.run_search",
                return_value={
                    "results": candidates,
                    "applied_filters": search_input.model_dump(),
                },
            ),
            patch(
                "agents.agent_find_company.node.pick_company",
                return_value=selection,
            ),
        ):
            result = find_company_node({"user_query": "로봇 회사"})

        self.assertEqual(result["selected_company"], candidates[0])
        self.assertEqual(
            result["selected_company_reason"],
            "모델 응답을 후보 목록과 정확히 매칭하지 못해 첫 번째 후보를 선택했습니다.",
        )


if __name__ == "__main__":
    unittest.main()
