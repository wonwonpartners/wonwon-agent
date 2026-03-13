from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.agent_investigate_members.result import (
    InvestigateMemberExtraction,
    InvestigateMembersExtractionResult,
)
from agents.agent_investigate_members.service import (
    extract_search_results,
    build_company_profile,
    build_search_queries,
    collect_investigate_member_signals,
    run_investigate_members,
)


def build_selected_company() -> dict[str, object]:
    return {
        "company_id": "CP_MEMBER",
        "company_name": "테스트컴퍼니",
        "product_name": "테스트 제품",
        "description": "로봇 자동화 솔루션 기업",
    }


def build_signal(
    *,
    source_id: str,
    url: str,
    title: str,
    snippet: str,
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "snippet": snippet,
        "published_at": "2026-03-01",
        "query": '"테스트컴퍼니" leadership core team 경영진',
        "source_kind": "web",
        "domain": url.split("/")[2],
    }


def build_signals(*, distinct_urls: int = 2) -> list[dict[str, str]]:
    signals = [
        build_signal(
            source_id="S1",
            url="https://news.example.com/ceo",
            title="테스트컴퍼니 홍대표 인터뷰",
            snippet="홍대표는 로봇 AI 상용화를 이끌었다.",
        ),
        build_signal(
            source_id="S2",
            url="https://company.example.com/team",
            title="테스트컴퍼니 핵심팀 소개",
            snippet="김총괄 COO가 시스템 통합과 사업개발을 맡고 있다.",
        ),
    ]
    if distinct_urls <= 1:
        signals[1]["url"] = signals[0]["url"]
        signals[1]["domain"] = signals[0]["domain"]
    return signals


def build_completed_extraction() -> InvestigateMembersExtractionResult:
    return InvestigateMembersExtractionResult(
        ceo=InvestigateMemberExtraction(
            name="홍대표",
            current_role="CEO",
            is_founder=True,
            experience_tags=["robot_sw_ai", "productization_deployment"],
            evidence_summary="로봇 AI 제품화 경력이 확인된다.",
            source_ids=["S1"],
            confidence=0.9,
        ),
        key_members=[
            InvestigateMemberExtraction(
                name="김총괄",
                current_role="COO",
                is_founder=False,
                experience_tags=["system_integration", "business_development"],
                evidence_summary="시스템 통합과 사업 운영 총괄 경험이 드러난다.",
                source_ids=["S2"],
                confidence=0.84,
            )
        ],
        strengths=["대표와 운영 핵심인력에 대한 공개 근거가 있다."],
        evidence_gaps=["제조 리더십은 추가 확인이 필요하다."],
        assessment_summary="대표와 COO 중심의 핵심팀 근거가 확인된다.",
        evidence_quality="서로 다른 URL 2건을 통해 교차 확인했다.",
    )


def build_ceo_only_extraction() -> InvestigateMembersExtractionResult:
    return InvestigateMembersExtractionResult(
        ceo=InvestigateMemberExtraction(
            name="홍대표",
            current_role="CEO",
            is_founder=True,
            experience_tags=["robot_sw_ai"],
            evidence_summary="대표의 로봇 AI 경력이 공개 자료에 언급된다.",
            source_ids=["S1"],
            confidence=0.88,
        ),
        key_members=[],
        strengths=[],
        evidence_gaps=["핵심팀 공개 자료가 부족하다."],
        assessment_summary="대표 외 핵심팀 근거가 없다.",
        evidence_quality="근거가 제한적이다.",
    )


def build_tag_rich_extraction() -> InvestigateMembersExtractionResult:
    return InvestigateMembersExtractionResult(
        ceo=InvestigateMemberExtraction(
            name="홍대표",
            current_role="CEO",
            is_founder=True,
            experience_tags=["robot_hw", "robot_sw_ai", "control_perception"],
            evidence_summary="로봇 HW/SW와 제어 경험이 확인된다.",
            source_ids=["S1"],
            confidence=0.92,
        ),
        key_members=[
            InvestigateMemberExtraction(
                name="김총괄",
                current_role="COO",
                is_founder=False,
                experience_tags=[
                    "system_integration",
                    "productization_deployment",
                    "manufacturing_operations",
                    "business_development",
                ],
                evidence_summary="통합, 배치, 운영, 사업 경험이 확인된다.",
                source_ids=["S2"],
                confidence=0.86,
            )
        ],
        strengths=[],
        evidence_gaps=[],
        assessment_summary="",
        evidence_quality="",
    )


class InvestigateMembersServiceTests(unittest.TestCase):
    def test_extract_search_results_accepts_json_string_payload(self) -> None:
        raw = '[{"title":"테스트","url":"https://example.com","content":"본문"}]'

        result = extract_search_results(raw)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "테스트")

    def test_skips_when_selected_company_is_missing(self) -> None:
        result = run_investigate_members(None)

        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["structured_output"])

    def test_completes_when_ceo_key_member_and_two_urls_exist(self) -> None:
        with (
            patch(
                "agents.agent_investigate_members.service.collect_investigate_member_signals",
                return_value=build_signals(),
            ),
            patch(
                "agents.agent_investigate_members.service.extract_investigate_members",
                return_value=build_completed_extraction(),
            ),
        ):
            result = run_investigate_members(build_selected_company())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["structured_output"]["ceo"]["name"], "홍대표")
        self.assertEqual(len(result["structured_output"]["key_members"]), 1)

    def test_fails_when_only_ceo_is_found(self) -> None:
        with (
            patch(
                "agents.agent_investigate_members.service.collect_investigate_member_signals",
                return_value=build_signals(),
            ),
            patch(
                "agents.agent_investigate_members.service.extract_investigate_members",
                return_value=build_ceo_only_extraction(),
            ),
        ):
            result = run_investigate_members(build_selected_company())

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["structured_output"]["key_members"], [])

    def test_fails_when_distinct_url_threshold_is_not_met(self) -> None:
        with (
            patch(
                "agents.agent_investigate_members.service.collect_investigate_member_signals",
                return_value=build_signals(distinct_urls=1),
            ),
            patch(
                "agents.agent_investigate_members.service.extract_investigate_members",
                return_value=build_completed_extraction(),
            ),
        ):
            result = run_investigate_members(build_selected_company())

        self.assertEqual(result["status"], "failed")
        self.assertIn("서로 다른 URL 2건 이상", " ".join(result["findings"]))

    def test_fails_gracefully_when_search_or_config_raises(self) -> None:
        with patch(
            "agents.agent_investigate_members.service.collect_investigate_member_signals",
            side_effect=RuntimeError("TAVILY_API_KEY 환경변수가 필요합니다."),
        ):
            result = run_investigate_members(build_selected_company())

        self.assertEqual(result["status"], "failed")
        self.assertIn("오류 메시지", " ".join(result["findings"]))
        self.assertIn("실행 오류", " ".join(result["structured_output"]["evidence_gaps"]))

    def test_role_coverage_maps_experience_tags_deterministically(self) -> None:
        with (
            patch(
                "agents.agent_investigate_members.service.collect_investigate_member_signals",
                return_value=build_signals(),
            ),
            patch(
                "agents.agent_investigate_members.service.extract_investigate_members",
                return_value=build_tag_rich_extraction(),
            ),
        ):
            result = run_investigate_members(build_selected_company())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["structured_output"]["role_coverage"],
            {
                "robot_hw": True,
                "robot_sw_ai": True,
                "control_perception": True,
                "system_integration": True,
                "productization_deployment": True,
                "manufacturing_operations": True,
                "business_development": True,
            },
        )

    def test_collect_signals_prefers_official_and_news_sources(self) -> None:
        company_profile = build_company_profile(build_selected_company())
        search_queries = build_search_queries(company_profile)

        class FakeSearch:
            def invoke(self, payload: dict[str, str]) -> object:
                query = payload["query"]
                if "회사 소개 대표 CEO" in query:
                    return {
                        "results": [
                            {
                                "title": "테스트컴퍼니 회사 소개 | 홍대표 CEO",
                                "url": "https://testcompany.com/about/leadership",
                                "content": "테스트컴퍼니는 홍대표 CEO가 이끄는 로봇 자동화 기업이다.",
                                "raw_content": "홍대표 CEO는 창업자로서 로봇 자동화 제품 상용화를 총괄한다.",
                                "published_date": "2026-03-01",
                            },
                            {
                                "title": "테스트컴퍼니 채용",
                                "url": "https://testcompany.com/careers",
                                "content": "테스트컴퍼니 채용 공고",
                                "raw_content": "로봇 엔지니어 채용 중",
                                "published_date": "2026-03-01",
                            },
                        ]
                    }
                if "대표 인터뷰 창업자 기사" in query:
                    return {
                        "results": [
                            {
                                "title": "테스트컴퍼니 홍대표 인터뷰",
                                "url": "https://news.example.com/testcompany-ceo",
                                "content": "홍대표는 창업자이자 CEO로 로봇 AI 상용화를 이끌고 있다.",
                                "raw_content": "홍대표는 창업자이자 CEO이며 현장 배치 경험을 설명했다.",
                                "published_date": "2026-02-27",
                            }
                        ]
                    }
                if "팀 소개 리더십 경영진" in query:
                    return {
                        "results": [
                            {
                                "title": "테스트컴퍼니 핵심팀",
                                "url": "https://testcompany.com/team",
                                "content": "김총괄 COO가 시스템 통합과 사업개발을 맡고 있다.",
                                "raw_content": "핵심팀에는 김총괄 COO와 박리드 Head of Product가 포함된다.",
                                "published_date": "2026-03-02",
                            }
                        ]
                    }
                if "site:linkedin.com" in query:
                    return {
                        "results": [
                            {
                                "title": "김총괄 - 테스트컴퍼니 COO",
                                "url": "https://www.linkedin.com/in/test-coo",
                                "content": "테스트컴퍼니 COO",
                                "raw_content": "김총괄 COO at 테스트컴퍼니",
                                "published_date": "",
                            }
                        ]
                    }
                return "No search results found"

        with patch(
            "agents.agent_investigate_members.service.get_web_search_tool",
            return_value=FakeSearch(),
        ):
            signals = collect_investigate_member_signals(company_profile, search_queries)

        self.assertTrue(signals)
        self.assertIn("official", {signal["source_type"] for signal in signals})
        self.assertIn("news", {signal["source_type"] for signal in signals})
        self.assertNotIn(
            "https://testcompany.com/careers",
            {signal["url"] for signal in signals},
        )

    def test_collect_signals_tolerates_string_tool_responses(self) -> None:
        company_profile = build_company_profile(build_selected_company())
        search_queries = build_search_queries(company_profile)

        class FakeSearch:
            def invoke(self, payload: dict[str, str]) -> object:
                return "No search results found"

        with patch(
            "agents.agent_investigate_members.service.get_web_search_tool",
            return_value=FakeSearch(),
        ):
            signals = collect_investigate_member_signals(company_profile, search_queries)

        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
