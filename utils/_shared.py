import logging
import json
import re
from typing import Any, Dict, Optional

from utils.openai_fallback import (
    ainvoke_with_rate_limit_fallback,
    build_chat_model,
    get_fallback_openai_model_name,
    invoke_with_rate_limit_fallback,
)

logger = logging.getLogger(__name__)


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
        localized_prompt = (
            "모든 설명형 응답은 반드시 한국어로 작성하세요. "
            "JSON key 이름은 schema 정의를 그대로 유지하고, "
            "JSON value 중 자연어 문장/구/요약/근거는 모두 한국어로 반환하세요.\n\n"
            f"{prompt}"
        )
        if hasattr(llm.with_structured_output(schema), "ainvoke"):
            result = await ainvoke_with_rate_limit_fallback(
                payload=localized_prompt,
                primary_factory=lambda: llm.with_structured_output(schema),
                fallback_factory=lambda: build_fallback_structured_model(llm, schema),
                logger=logger,
                operation_name="utils.run_structured_from_llm",
            )
        else:
            result = invoke_with_rate_limit_fallback(
                payload=localized_prompt,
                primary_factory=lambda: llm.with_structured_output(schema),
                fallback_factory=lambda: build_fallback_structured_model(llm, schema),
                logger=logger,
                operation_name="utils.run_structured_from_llm",
            )
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return json.loads(result)
    except Exception:
        return None
    return None


def build_fallback_structured_model(llm: Any, schema: Any) -> Any:
    return build_chat_model(
        model=get_fallback_openai_model_name(),
        temperature=float(getattr(llm, "temperature", 0) or 0),
        api_key=extract_api_key(llm),
    ).with_structured_output(schema)


def extract_api_key(llm: Any) -> str | None:
    for attr_name in ("openai_api_key", "api_key"):
        value = getattr(llm, attr_name, None)
        if value:
            return str(value)
    return None


def normalize_feedback_payload(feedback_raw: Dict[str, str]) -> Dict[str, str]:
    return {k.strip(): v.strip() for k, v in feedback_raw.items() if k.strip() and v is not None}
