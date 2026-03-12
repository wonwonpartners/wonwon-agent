from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ExperienceTag = Literal[
    "robot_hw",
    "robot_sw_ai",
    "control_perception",
    "system_integration",
    "productization_deployment",
    "manufacturing_operations",
    "business_development",
]


class InvestigateMemberExtraction(BaseModel):
    name: str = Field(description="팀원의 이름")
    current_role: str = Field(description="현재 직책 또는 역할")
    is_founder: bool = Field(description="창업자 또는 공동창업자인지 여부")
    experience_tags: list[ExperienceTag] = Field(
        default_factory=list,
        description="허용된 taxonomy 안에서만 선택한 경험 태그",
    )
    evidence_summary: str = Field(description="근거 기반 경력 요약")
    source_ids: list[str] = Field(
        default_factory=list,
        description="근거로 사용한 source_id 목록",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="추출 결과에 대한 신뢰도",
    )


class InvestigateMembersExtractionResult(BaseModel):
    ceo: InvestigateMemberExtraction | None = Field(
        default=None,
        description="대표 또는 CEO 1인. 찾지 못하면 null",
    )
    key_members: list[InvestigateMemberExtraction] = Field(
        default_factory=list,
        description="CEO를 제외한 핵심 구성원 목록",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="핵심팀 강점 요약",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="현재 공개 자료 기준으로 부족한 근거",
    )
    assessment_summary: str = Field(
        default="",
        description="C.1 관점의 조사 요약",
    )
    evidence_quality: str = Field(
        default="",
        description="공개 근거의 충실도 요약",
    )
