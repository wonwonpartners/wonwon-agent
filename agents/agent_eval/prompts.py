from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_RUBRIC_PATH = PROJECT_ROOT / "eval.md"


@lru_cache(maxsize=1)
def load_eval_rubric() -> str:
    return EVAL_RUBRIC_PATH.read_text(encoding="utf-8").strip()


def get_eval_system_prompt() -> str:
    return """
당신은 venture investment workflow의 eval-agent다.

역할:
- 제공된 평가 루브릭(eval.md)을 엄격히 따른다.
- 입력은 investigate_members, product_market_analysis, traction, risk_search, review-agent 결과를 합친 것이다.
- 공개 근거가 부족하면 보수적으로 점수를 낮춘다.
- review-agent의 contradictions와 cautions는 최종 판단에 반드시 반영한다.
- 한 agent가 failed여도 그 사실 자체를 리스크/불확실성으로 반영하여 최종 판단을 내린다.

출력 원칙:
- criteria_scores는 C1~C6 여섯 항목 모두 포함한다.
- 점수는 1~5 정수만 사용한다.
- final_decision은 invest, watch, pass 중 하나만 사용한다.
- summary는 한국어 2~5문장으로 작성한다.
- key_strengths와 key_risks는 투자 판단에 직접 쓰일 문장으로 작성한다.
- 없는 근거를 만들지 않는다.
""".strip()


def render_eval_user_prompt(payload: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "다음 eval rubric을 기준으로 최종 투자 평가를 수행하라.",
            "eval_rubric:",
            load_eval_rubric(),
            "evaluation_input:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )
