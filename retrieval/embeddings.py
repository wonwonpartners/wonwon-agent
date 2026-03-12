from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from retrieval.config import VectorStoreConfig


def get_embeddings(config: VectorStoreConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.device},
        encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
    )
