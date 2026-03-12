from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VectorStoreConfig:
    key: str
    collection_name: str
    persist_dir: Path
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "mps"
    normalize_embeddings: bool = True


VECTOR_STORE_CONFIGS: dict[str, VectorStoreConfig] = {
    "company": VectorStoreConfig(
        key="company",
        collection_name="company_corpus",
        persist_dir=PROJECT_ROOT / "vectordb" / "company",
    ),
    "domain": VectorStoreConfig(
        key="domain",
        collection_name="domain_corpus",
        persist_dir=PROJECT_ROOT / "vectordb" / "domain",
    ),
}


def get_vector_store_config(name: str) -> VectorStoreConfig:
    try:
        return VECTOR_STORE_CONFIGS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(VECTOR_STORE_CONFIGS))
        raise ValueError(f"Unknown vector store config '{name}'. Supported: {supported}") from exc
