from __future__ import annotations

import os
from functools import lru_cache
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from utils.rdb import get_engine
from utils.rdb_queries import search_companies as search_companies_query
from utils.schema_prompt import generate_schema_prompt
from utils.taxonomy_prompt import generate_taxonomy_prompt

load_dotenv()