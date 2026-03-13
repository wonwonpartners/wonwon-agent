from __future__ import annotations

from pydantic import BaseModel, Field

from agents.agent_find_company.common import MAX_COMPANY_CANDIDATES


class FindCompanySearchInput(BaseModel):
    query: str = Field(
        default="",
        description="기업명, 제품명, 설명, keyword를 기준으로 찾는 자유 텍스트 검색어",
    )
    invest_level: str | None = Field(
        default=None,
        description="canonical 투자 단계 값. 예: seed, pre-A, series A, series B",
    )
    employees_min: int | None = Field(
        default=None,
        description="직원 수 하한",
    )
    employees_max: int | None = Field(
        default=None,
        description="직원 수 상한",
    )
    hiring: bool | None = Field(
        default=None,
        description="채용 중 여부. 채용 중인 회사를 찾는 조건이면 true",
    )
    categories: list[str] | None = Field(
        default=None,
        description="canonical category_name 목록. 정확한 taxonomy 값을 사용",
    )
    limit: int = Field(
        default=MAX_COMPANY_CANDIDATES,
        ge=1,
        le=20,
        description="가져올 최대 후보 수",
    )
