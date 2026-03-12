from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever

from retrieval.vectorstore import get_vector_store


def get_retriever(
    store_name: str,
    *,
    k: int = 4,
    search_type: str = "similarity",
    search_kwargs: dict | None = None,
) -> VectorStoreRetriever:
    options = {"k": k, **(search_kwargs or {})}
    return get_vector_store(store_name).as_retriever(
        search_type=search_type,
        search_kwargs=options,
    )
