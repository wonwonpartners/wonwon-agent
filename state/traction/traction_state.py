from typing import Dict, List, TypedDict


class TractionInputState(TypedDict):
    startup_name: str

class TractionState(TypedDict):
    partnerships: List[str]
    hiring_analysis: Dict[str, float]
    funding_velocity: List[str]
    traction_summary: str
