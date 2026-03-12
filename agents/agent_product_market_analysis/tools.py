from __future__ import annotations

import os
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

from retrieval import get_retriever


MIN_COMPANY_RAG_SCORE = 0.7
MIN_DOMAIN_RAG_SCORE = 0.35


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
        source_name = (
            result.get("source") or result.get("site_name") or result.get("domain")
        )
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


def _search_rag_corpus(
    *,
    query: str,
    corpus: str,
    top_k: int,
    score_threshold: float,
) -> str:
    retriever = get_retriever(
        corpus,
        k=top_k,
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold},
    )
    docs = retriever.invoke(query)
    return _format_retrieved_docs(docs)


@tool
def company_rag_search_tool(query: str, top_k: int = 5) -> str:
    """Search company-authored materials such as official homepage, product page, whitepaper, and PR materials."""
    return _search_rag_corpus(
        query=query,
        corpus="company",
        top_k=top_k,
        score_threshold=MIN_COMPANY_RAG_SCORE,
    )


@tool
def domain_rag_search_tool(query: str, top_k: int = 5) -> str:
    """Search external domain materials such as papers, reports, market data, and benchmark references."""
    return _search_rag_corpus(
        query=query,
        corpus="domain",
        top_k=top_k,
        score_threshold=MIN_DOMAIN_RAG_SCORE,
    )


@tool
def company_web_search_tool(query: str, top_k: int = 5) -> str:
    """Search the web for company-specific public information such as official website pages, press releases, interviews, blog posts, and news coverage."""
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TRAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 환경변수가 필요합니다.")

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
def web_benchmark_search_tool(query: str, top_k: int = 5) -> str:
    """Search the web for competitor, comparable service, and market benchmark information."""
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TRAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 환경변수가 필요합니다.")

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
    """Fetch and extract readable text from a specific web page URL."""
    response = requests.get(
        url,
        headers={"User-Agent": "agent-product-market-analysis/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        return f"Unsupported content type for extraction: {content_type or 'unknown'}"

    extracted = _extract_webpage_text(response.text, max_chars=max_chars)
    return f"URL: {url}\n\n{extracted}"


PRODUCT_MARKET_ANALYSIS_TOOLS = [
    company_rag_search_tool,
    domain_rag_search_tool,
    company_web_search_tool,
    web_benchmark_search_tool,
    web_page_extract_tool,
]
