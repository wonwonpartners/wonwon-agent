from __future__ import annotations

from pydantic import BaseModel, Field


class CompanySelectionResult(BaseModel):
    company_id: str = Field(description="후보 목록에서 선택한 회사의 company_id")
    reason: str = Field(description="해당 회사를 선택한 이유")
