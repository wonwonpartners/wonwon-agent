from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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


class CompanyResearchGraphTests(unittest.TestCase):
    def test_ends_early_when_no_company_is_selected(self) -> None:
        investigate_members_mock = Mock(return_value={})
        agent_a_mock = Mock(return_value={})
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
            patch("company_research_graph.agent_a_node", agent_a_mock),
            patch("company_research_graph.review_investigate_members_node", review_mock),
            patch("company_research_graph.eval_node", eval_mock),
            patch("company_research_graph.report_node", report_mock),
        ):
            result = run_company_research("로봇 회사")

        self.assertIsNone(result["selected_company"])
        self.assertNotIn("graph_error", result)
        self.assertNotIn("report_state", result)
        investigate_members_mock.assert_not_called()
        agent_a_mock.assert_not_called()
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
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")
                report_path = Path(result["report_state"]["report_path"])
                self.assertTrue(report_path.exists())

        self.assertEqual(result["eval_state"]["status"], "completed")
        self.assertEqual(result["report_state"]["status"], "completed")
        self.assertTrue(result["report_state"]["report_path"].endswith("CP_SUCCESS.md"))
        self.assertIn("## 회사 선정 결과", result["report_state"]["markdown"])
        self.assertIn("## investigate_members", result["report_state"]["markdown"])
        self.assertIn("## eval 요약", result["report_state"]["markdown"])

    def test_retry_once_then_approve_reaches_success(self) -> None:
        selected_company = build_selected_company(company_id="CP_RETRY")
        review_outputs = [
            {
                "investigate_members_review": {
                    "reviewed_agent": "investigate_members",
                    "decision": "rejected",
                    "reason": "첫 번째 reject",
                    "review_count": 1,
                }
            },
            {
                "investigate_members_review": {
                    "reviewed_agent": "investigate_members",
                    "decision": "approved",
                    "reason": "두 번째 시도 통과",
                    "review_count": 2,
                }
            },
        ]
        review_mock = Mock(side_effect=review_outputs)

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
                    "company_research_graph.review_investigate_members_node",
                    review_mock,
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(review_mock.call_count, 2)
        self.assertEqual(result["investigate_members_state"]["attempt_count"], 2)
        self.assertEqual(result["report_state"]["status"], "completed")
        self.assertNotIn("graph_error", result)

    def test_second_reject_terminates_graph_with_error_state(self) -> None:
        selected_company = build_selected_company(company_id="CP_FAIL")
        review_outputs = [
            {
                "investigate_members_review": {
                    "reviewed_agent": "investigate_members",
                    "decision": "rejected",
                    "reason": "첫 번째 reject",
                    "review_count": 1,
                }
            },
            {
                "investigate_members_review": {
                    "reviewed_agent": "investigate_members",
                    "decision": "rejected",
                    "reason": "두 번째 reject",
                    "review_count": 2,
                }
            },
        ]
        review_mock = Mock(side_effect=review_outputs)

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
                    "company_research_graph.review_investigate_members_node",
                    review_mock,
                ),
                patch("agents.agent_report.service.REPORTS_ROOT", reports_root),
            ):
                result = run_company_research("로봇 회사")

        self.assertEqual(result["graph_error"]["stage"], "review")
        self.assertEqual(result["graph_error"]["agent_name"], "investigate_members")
        self.assertIn("두 번째 reject", result["graph_error"]["message"])
        self.assertNotIn("eval_state", result)
        self.assertNotIn("report_state", result)

    def test_report_path_is_stable_and_overwrites_existing_file(self) -> None:
        selected_company = build_selected_company(company_id="CP_STABLE")
        eval_v1 = {
            "eval_state": {
                "status": "completed",
                "ready_for_report": True,
                "summary": "첫 번째 평가 요약",
                "agent_summaries": {
                    "investigate_members": "v1",
                    "agent_a": "v1",
                    "agent_b": "v1",
                    "agent_c": "v1",
                },
            }
        }
        eval_v2 = {
            "eval_state": {
                "status": "completed",
                "ready_for_report": True,
                "summary": "두 번째 평가 요약",
                "agent_summaries": {
                    "investigate_members": "v2",
                    "agent_a": "v2",
                    "agent_b": "v2",
                    "agent_c": "v2",
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
