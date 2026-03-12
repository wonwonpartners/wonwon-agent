from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ResearchAgentState


class InvestigateMemberProfile(TypedDict):
    name: str
    current_role: str
    is_founder: bool
    experience_tags: list[str]
    evidence_summary: str
    source_ids: list[str]
    confidence: float


class InvestigateMembersRoleCoverage(TypedDict):
    robot_hw: bool
    robot_sw_ai: bool
    control_perception: bool
    system_integration: bool
    productization_deployment: bool
    manufacturing_operations: bool
    business_development: bool


class InvestigateMembersPayload(TypedDict):
    ceo: InvestigateMemberProfile | None
    key_members: list[InvestigateMemberProfile]
    role_coverage: InvestigateMembersRoleCoverage
    strengths: list[str]
    evidence_gaps: list[str]
    assessment_summary: str
    evidence_quality: str
    search_queries: list[str]


class InvestigateMembersNodeOutput(TypedDict):
    investigate_members_state: ResearchAgentState
    leadership_research: InvestigateMembersPayload | None
