from __future__ import annotations

import os
from typing import Literal

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from retrieval import get_vector_store


def _build_metadata_lines(metadata: dict) -> list[str]:
    fields = [
        ("source", metadata.get("source_path")),
        ("title", metadata.get("title")),
        ("author", metadata.get("author")),
        ("organization", metadata.get("organization")),
        ("publisher", metadata.get("publisher")),
        ("journal", metadata.get("journal")),
        ("published_at", metadata.get("published_at")),
        ("url", metadata.get("url")),
    ]
    return [f"{key}: {value}" for key, value in fields if value]


def _format_retrieved_docs(docs: list, *, max_chars: int = 4000) -> str:
    if not docs:
        return "No relevant documents found."

    sections: list[str] = []
    remaining = max_chars

    for doc in docs:
        metadata_lines = _build_metadata_lines(doc.metadata)
        body = doc.page_content.strip()

        if remaining <= 0:
            break

        prefix = "\n".join(["[document]"] + metadata_lines)
        available_body_chars = max(0, remaining - len(prefix) - 2)
        snippet = body[:available_body_chars]
        sections.append(f"{prefix}\n{snippet}".rstrip())
        remaining -= len(prefix) + len(snippet) + 2

    return "\n\n".join(sections)


def _format_web_results(results: list[dict], *, max_chars: int = 4000) -> str:
    if not results:
        return "No relevant web results found."

    sections: list[str] = []
    remaining = max_chars

    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "unknown")
        published_date = result.get("published_date") or result.get("published_at")
        source_name = result.get("source") or result.get("site_name") or result.get("domain")
        content = (result.get("content") or result.get("snippet") or "").strip()
        section_lines = [
            f"[{index}] {title}",
            f"URL: {url}",
        ]
        if source_name:
            section_lines.append(f"Source: {source_name}")
        if published_date:
            section_lines.append(f"Published: {published_date}")
        section_lines.append(f"Snippet: {content}")
        section = "\n".join(section_lines)

        if remaining <= 0:
            break

        snippet = section[:remaining]
        sections.append(snippet)
        remaining -= len(snippet)

    return "\n\n".join(sections)


def _extract_webpage_text(html: str, *, max_chars: int = 6000) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text("\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())
    text = text[:max_chars]

    if title:
        return f"Title: {title}\n\n{text}"
    return text


@tool
def rag_search_tool(
    query: str,
    corpus: Literal["company", "domain"],
    top_k: int = 5,
) -> str:
    """
    Search the local vector database for grounded evidence.

    Use corpus='company' for company-authored materials such as the official homepage,
    product pages, whitepapers, and PR materials.

    Use corpus='domain' for external domain knowledge such as research papers,
    industry reports, market data, and benchmark references.

    Prefer this tool when you need reliable source-backed context before making
    claims about KPI logic, technical moat, or data loop structure.
    """
    vector_store = get_vector_store(corpus)
    docs = vector_store.similarity_search(query, k=top_k)
    return _format_retrieved_docs(docs)


@tool
def web_benchmark_search_tool(query: str, top_k: int = 5) -> str:
    """
    Search the web for competitor, comparable service, and market benchmark information.

    Use this tool for:
    - similar services and competitors
    - product positioning and feature comparison
    - recent external information not already covered in the local vector database

    Prefer this tool when you need market-facing comparisons rather than internal
    company claims or curated domain references.
    """
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TRAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set.")

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "topic": "general",
            "max_results": top_k,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    results = payload.get("results", [])
    return _format_web_results(results)


@tool
def web_page_extract_tool(url: str, max_chars: int = 6000) -> str:
    """
    Fetch and extract readable text from a specific web page URL.

    Use this tool after web_benchmark_search_tool when you need deeper evidence
    from a promising result page, such as a competitor product page, benchmark
    article, or external analysis page.
    """
    response = requests.get(
        url,
        headers={"User-Agent": "product-market-agent/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        return f"Unsupported content type for extraction: {content_type or 'unknown'}"

    extracted = _extract_webpage_text(response.text, max_chars=max_chars)
    return f"URL: {url}\n\n{extracted}"


PRODUCT_MARKET_TOOLS = [
    rag_search_tool,
    web_benchmark_search_tool,
    web_page_extract_tool,
]
