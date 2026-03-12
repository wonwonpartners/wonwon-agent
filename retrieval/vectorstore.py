from __future__ import annotations

from langchain_chroma import Chroma

from retrieval.config import VectorStoreConfig, get_vector_store_config
from retrieval.embeddings import get_embeddings


def get_vector_store(config_or_name: VectorStoreConfig | str) -> Chroma:
    config = (
        get_vector_store_config(config_or_name)
        if isinstance(config_or_name, str)
        else config_or_name
    )
    config.persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=get_embeddings(config),
        persist_directory=str(config.persist_dir),
    )
