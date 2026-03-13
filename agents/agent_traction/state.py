from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class TractionInputState(TypedDict):
    startup_name: str


class HiringAnalysisOutput(BaseModel):
    field_engineer_ratio: float = Field(default=0.0)
    field_engineer_count: int = Field(default=0)
    hiring_trend_3m: int = Field(default=0)


class EvidenceSourceOutput(BaseModel):
    source_type: str = Field(default="web")
    signal_type: str = Field(default="")
    query: str = Field(default="")
    title: str = Field(default="")
    publisher: str = Field(default="")
    published_at: str = Field(default="")
    url: str = Field(default="")
    source: str = Field(default="")
    score: float = Field(default=0.0)


class TractionState(BaseModel):
    partnerships: list[str] = Field(default_factory=list)
    hiring_analysis: HiringAnalysisOutput = Field(default_factory=HiringAnalysisOutput)
    funding_velocity: list[str] = Field(default_factory=list)
    traction_summary: str = Field(default="")
    evidence_sources: list[EvidenceSourceOutput] = Field(default_factory=list)
