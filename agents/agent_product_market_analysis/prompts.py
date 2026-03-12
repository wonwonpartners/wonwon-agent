from __future__ import annotations

from functools import lru_cache

from utils.prompt_templates import render_prompt_template


@lru_cache(maxsize=1)
def get_research_system_prompt() -> str:
    return render_prompt_template("agent_product_market_analysis/system_prompt.md")


@lru_cache(maxsize=1)
def get_writer_system_prompt() -> str:
    return render_prompt_template(
        "agent_product_market_analysis/writer_system_prompt.md"
    )


def render_research_user_prompt(company_profile_text: str) -> str:
    return render_prompt_template(
        "agent_product_market_analysis/user_prompt.md",
        company_profile=company_profile_text,
    )
