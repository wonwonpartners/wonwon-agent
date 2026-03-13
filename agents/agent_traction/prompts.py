from __future__ import annotations

from functools import lru_cache

from utils.prompt_templates import render_prompt_template


@lru_cache(maxsize=1)
def get_traction_state_prompt() -> str:
    return render_prompt_template("agent_traction/traction_state_prompt.md")


@lru_cache(maxsize=1)
def get_sufficiency_prompt() -> str:
    return render_prompt_template("agent_traction/sufficiency_prompt.md")


@lru_cache(maxsize=1)
def get_quality_prompt() -> str:
    return render_prompt_template("agent_traction/quality_prompt.md")


def render_traction_state_prompt(*, startup_name: str, context_text: str) -> str:
    return render_prompt_template(
        "agent_traction/traction_state_prompt.md",
        startup_name=startup_name,
        context_text=context_text,
    )


def render_sufficiency_prompt(
    *,
    startup_name: str,
    signal_names: str,
    context_text: str,
) -> str:
    return render_prompt_template(
        "agent_traction/sufficiency_prompt.md",
        startup_name=startup_name,
        signal_names=signal_names,
        context_text=context_text,
    )


def render_quality_prompt(
    *,
    startup_name: str,
    signal_names: str,
    context_text: str,
) -> str:
    return render_prompt_template(
        "agent_traction/quality_prompt.md",
        startup_name=startup_name,
        signal_names=signal_names,
        context_text=context_text,
    )
