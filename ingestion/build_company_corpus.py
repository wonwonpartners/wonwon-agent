from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from retrieval import get_vector_store
from retrieval.config import VectorStoreConfig, get_vector_store_config

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".json"}
@dataclass(frozen=True)
class CompanyCorpusConfig:
    source_dir: Path
    persist_dir: Path
    collection_name: str
    vector_store_key: str = "company"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    batch_size: int = 32
    glob_pattern: str = "**/*"


def parse_args() -> CompanyCorpusConfig:
    vector_config = get_vector_store_config("company")
    parser = argparse.ArgumentParser(
        description="Build a company corpus vector DB from collected company-authored documents."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("../data/company"),
        help="Directory containing company document folders.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=vector_config.persist_dir,
        help="Directory where the Chroma DB will be stored.",
    )
    parser.add_argument(
        "--collection-name",
        default=vector_config.collection_name,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size for text splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for text splitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to embed and insert per batch.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/*",
        help="Glob pattern used to discover source files inside docs folders.",
    )

    args = parser.parse_args()
    return CompanyCorpusConfig(
        source_dir=args.source_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        glob_pattern=args.glob_pattern,
    )


def collect_source_files(config: CompanyCorpusConfig) -> list[Path]:
    if not config.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {config.source_dir}")

    files = [
        path
        for path in config.source_dir.glob(config.glob_pattern)
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "docs" in path.parts
    ]
    if not files:
        logger.warning("No supported company documents found under %s", config.source_dir)
    return sorted(files)


def load_documents(paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []

    for path in paths:
        suffix = path.suffix.lower()
        metadata = load_sidecar_metadata(path)

        if suffix in {".txt", ".md", ".html"}:
            documents.extend(load_text_like_document(path, metadata))
        elif suffix == ".pdf":
            documents.extend(load_pdf_document(path, metadata))
        elif suffix == ".json":
            documents.extend(load_json_document(path, metadata))
        else:
            logger.info("Skipping unsupported file: %s", path)

    return documents


def load_text_like_document(path: Path, metadata: dict) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata=merge_metadata(
                metadata,
                {
                    "source_path": str(path),
                    "doc_type": path.suffix.lower().lstrip("."),
                },
            ),
        )
    ]


def load_pdf_document(path: Path, metadata: dict) -> list[Document]:
    from langchain_community.document_loaders import PDFPlumberLoader

    loader = PDFPlumberLoader(str(path))
    docs = loader.load()
    for index, doc in enumerate(docs, start=1):
        doc.metadata.update(
            merge_metadata(
                metadata,
                {
                    "source_path": str(path),
                    "doc_type": "pdf",
                    "page": index,
                },
            )
        )
    return docs


def load_json_document(path: Path, metadata: dict) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        documents = []
        for index, item in enumerate(raw):
            text = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
            documents.append(
                Document(
                    page_content=text,
                    metadata=merge_metadata(
                        metadata,
                        {
                            "source_path": str(path),
                            "doc_type": "json",
                            "record_index": index,
                            **extract_optional_metadata(item),
                        },
                    ),
                )
            )
        return documents

    text = raw.get("text") or raw.get("content") or json.dumps(raw, ensure_ascii=False)
    return [
        Document(
            page_content=text,
            metadata=merge_metadata(
                metadata,
                {
                    "source_path": str(path),
                    "doc_type": "json",
                    **extract_optional_metadata(raw),
                },
            ),
        )
    ]


def load_sidecar_metadata(path: Path) -> dict:
    company_dir = path.parent.parent
    metadata_dir = company_dir / "metadata"
    metadata_path = metadata_dir / f"{path.stem}.json"

    if not metadata_path.exists():
        return build_fallback_metadata(path)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return merge_metadata(
        build_fallback_metadata(path),
        payload,
    )


def build_fallback_metadata(path: Path) -> dict:
    company_name = path.parent.parent.name
    return {
        "source_type": "company",
        "company": company_name,
        "document_type": "company_document",
        "title": path.stem,
        "url": "",
    }


def extract_optional_metadata(payload: dict) -> dict:
    keys = [
        "title",
        "source",
        "url",
        "publisher",
        "published_at",
        "region",
        "domain",
        "topic",
        "document_type",
        "company",
    ]
    return {key: payload[key] for key in keys if key in payload}


def merge_metadata(base: dict, extra: dict) -> dict:
    merged = dict(base)
    merged.update({key: value for key, value in extra.items() if value is not None})
    return merged


def split_documents(documents: list[Document], config: CompanyCorpusConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def batched(items: list[Document], batch_size: int) -> Iterable[list[Document]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_vector_store(chunks: list[Document], config: CompanyCorpusConfig):
    vector_config = VectorStoreConfig(
        key=config.vector_store_key,
        collection_name=config.collection_name,
        persist_dir=config.persist_dir,
    )
    logger.info("Loading embedding model: %s", vector_config.embedding_model)
    vector_store = get_vector_store(vector_config)

    total_batches = ceil(len(chunks) / config.batch_size)
    for batch in tqdm(
        batched(chunks, config.batch_size),
        total=total_batches,
        desc="Embedding and indexing",
        unit="batch",
    ):
        vector_store.add_documents(batch)

    return vector_store


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args()

    logger.info("Collecting source files from %s", config.source_dir)
    source_files = collect_source_files(config)

    logger.info("Loading %s company documents", len(source_files))
    documents = load_documents(source_files)
    if not documents:
        logger.warning("No documents loaded. Exiting without building vector DB.")
        return

    logger.info("Splitting documents into chunks")
    chunks = split_documents(documents, config)
    logger.info("Created %s chunks", len(chunks))

    logger.info(
        "Building Chroma collection '%s' with %s chunks (batch_size=%s)",
        config.collection_name,
        len(chunks),
        config.batch_size,
    )
    build_vector_store(chunks, config)

    logger.info("Done. Vector DB persisted to %s", config.persist_dir)


if __name__ == "__main__":
    main()
