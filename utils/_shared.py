import json
import re
from typing import Any, Dict, Optional


def has_number(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(re.search(r"\d", value))


async def run_structured_from_llm(llm: Any, schema: Any, prompt: str) -> Optional[Dict[str, Any]]:
    """LLM이 있는 경우 structured_output 경로를 사용한다.

    llm이 없거나 실패하면 None 반환하여 fallback 경로로 진행한다.
    """
    if llm is None:
        return None
    try:
        model = llm.with_structured_output(schema)
        localized_prompt = (
            "모든 설명형 응답은 반드시 한국어로 작성하세요. "
            "JSON key 이름은 schema 정의를 그대로 유지하고, "
            "JSON value 중 자연어 문장/구/요약/근거는 모두 한국어로 반환하세요.\n\n"
            f"{prompt}"
        )
        if hasattr(model, "ainvoke"):
            result = await model.ainvoke(localized_prompt)
        else:
            result = model.invoke(localized_prompt)
        if hasattr(result, "dict"):
            return result.dict()
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return json.loads(result)
    except Exception:
        return None
    return None


def normalize_feedback_payload(feedback_raw: Dict[str, str]) -> Dict[str, str]:
    return {k.strip(): v.strip() for k, v in feedback_raw.items() if k.strip() and v is not None}
