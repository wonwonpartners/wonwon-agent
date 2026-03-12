from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisFieldResult(BaseModel):
    text: str = Field(description="해당 평가 항목의 결론 텍스트")
    references: list[str] = Field(
        default_factory=list,
        description="리포트에 바로 사용할 수 있는 참고문헌 문자열 목록",
    )
    evidence_gap: str = Field(
        default="",
        description="근거 부족 또는 외부 검증 한계를 설명하는 짧은 문장",
    )


class ProductMarketAnalysisResult(BaseModel):
    target_kpi_logic: AnalysisFieldResult = Field(
        description="고객 pain point, KPI, ROI 논리 평가",
    )
    technical_moat: AnalysisFieldResult = Field(
        description="기술 해자와 경쟁 비교 평가",
    )
    data_loop_structure: AnalysisFieldResult = Field(
        description="AI 활용, 데이터 루프, fallback 구조 평가",
    )
    product_summary: AnalysisFieldResult = Field(
        description="제품/시장 종합 투자 관점 요약",
    )
