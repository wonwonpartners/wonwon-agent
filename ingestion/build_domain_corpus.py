from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm
from retrieval import get_vector_store
from retrieval.config import PROJECT_ROOT, VectorStoreConfig, get_vector_store_config
logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".json"}


@dataclass(frozen=True)
class DomainCorpusConfig:
    source_dir: Path
    persist_dir: Path
    collection_name: str
    vector_store_key: str = "domain"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    batch_size: int = 32
    glob_pattern: str = "**/*"
    similarity_threshold: float = 0.97


def parse_args() -> DomainCorpusConfig:
    vector_config = get_vector_store_config("domain")
    parser = argparse.ArgumentParser(
        description="Build a domain corpus vector DB from local reports, papers, and benchmark documents."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "domain",
        help="Directory containing domain documents.",
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
        help="Glob pattern used to discover source files.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.97,
        help=(
            "Skip a chunk if the most similar existing chunk in Chroma has "
            "relevance score greater than or equal to this threshold."
        ),
    )

    args = parser.parse_args()
    return DomainCorpusConfig(
        source_dir=args.source_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        glob_pattern=args.glob_pattern,
        similarity_threshold=args.similarity_threshold,
    )


def collect_source_files(config: DomainCorpusConfig) -> list[Path]:
    if not config.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {config.source_dir}")

    files = [
        path
        for path in config.source_dir.glob(config.glob_pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "metadata" not in path.parts
    ]
    if not files:
        logger.warning("No supported files found under %s", config.source_dir)
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
                build_base_metadata(path, doc_type=path.suffix.lower().lstrip(".")),
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
                    **build_base_metadata(path, doc_type="pdf"),
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
            item_metadata = merge_metadata(
                metadata,
                {
                    **build_base_metadata(path, doc_type="json"),
                    "record_index": index,
                    **extract_optional_metadata(item),
                },
            )
            documents.append(Document(page_content=text, metadata=item_metadata))
        return documents

    text = raw.get("text") or raw.get("content") or json.dumps(raw, ensure_ascii=False)
    item_metadata = merge_metadata(
        metadata,
        {
            **build_base_metadata(path, doc_type="json"),
            **extract_optional_metadata(raw),
        },
    )
    return [Document(page_content=text, metadata=item_metadata)]


def load_sidecar_metadata(path: Path) -> dict:
    metadata_dir = path.parent / "metadata"
    metadata_path = metadata_dir / f"{path.stem}.json"

    if not metadata_path.exists():
        return build_fallback_metadata(path)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return merge_metadata(build_fallback_metadata(path), payload)


def build_fallback_metadata(path: Path) -> dict:
    return {
        "source_type": "domain",
        "title": path.stem,
        "url": "",
    }


def extract_optional_metadata(payload: dict) -> dict:
    keys = [
        "title",
        "author",
        "source",
        "url",
        "publisher",
        "journal",
        "organization",
        "published_at",
        "region",
        "domain",
        "topic",
        "document_type",
    ]
    return {key: payload[key] for key in keys if key in payload}


def build_base_metadata(path: Path, *, doc_type: str) -> dict:
    relative_path = str(path)
    return {
        "source_path": relative_path,
        "source_type": "domain",
        "doc_type": doc_type,
    }


def merge_metadata(base: dict, extra: dict) -> dict:
    merged = dict(base)
    merged.update({key: value for key, value in extra.items() if value is not None})
    return merged


def split_documents(documents: list[Document], config: DomainCorpusConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def normalize_chunk_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def build_chunk_content_id(chunk: Document) -> str:
    normalized = normalize_chunk_text(chunk.page_content)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"domain-chunk:{digest}"


def dedupe_chunks_in_run(chunks: list[Document]) -> tuple[list[Document], list[str], int]:
    deduped_chunks: list[Document] = []
    chunk_ids: list[str] = []
    seen_ids: set[str] = set()
    skipped_count = 0

    for chunk in chunks:
        chunk_id = build_chunk_content_id(chunk)
        chunk.metadata["content_hash"] = chunk_id.removeprefix("domain-chunk:")
        if chunk_id in seen_ids:
            skipped_count += 1
            continue
        seen_ids.add(chunk_id)
        deduped_chunks.append(chunk)
        chunk_ids.append(chunk_id)

    return deduped_chunks, chunk_ids, skipped_count


def find_existing_ids(vector_store, ids: list[str]) -> set[str]:
    if not ids:
        return set()

    raw = vector_store.get(ids=ids, include=[])
    existing_ids = raw.get("ids") or []
    return {str(item) for item in existing_ids if item}


def get_max_similarity_score(vector_store, text: str) -> float:
    normalized = normalize_chunk_text(text)
    if not normalized:
        return 0.0

    matches = vector_store.similarity_search_with_relevance_scores(
        normalized,
        k=1,
    )
    if not matches:
        return 0.0

    _, score = matches[0]
    return float(score)


def batched(items: list[Document], batch_size: int) -> Iterable[list[Document]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_vector_store(chunks: list[Document], config: DomainCorpusConfig):
    vector_config = VectorStoreConfig(
        key=config.vector_store_key,
        collection_name=config.collection_name,
        persist_dir=config.persist_dir,
    )
    logger.info("Loading embedding model: %s", vector_config.embedding_model)
    vector_store = get_vector_store(vector_config)

    deduped_chunks, chunk_ids, skipped_in_run = dedupe_chunks_in_run(chunks)
    if skipped_in_run:
        logger.info("Skipped %s duplicate chunks within this run", skipped_in_run)

    existing_ids = find_existing_ids(vector_store, chunk_ids)
    skipped_existing_exact = 0
    skipped_existing_similar = 0
    inserted_count = 0

    for chunk, chunk_id in tqdm(
        zip(deduped_chunks, chunk_ids, strict=False),
        total=len(deduped_chunks),
        desc="Checking and indexing chunks",
        unit="chunk",
    ):
        if chunk_id in existing_ids:
            skipped_existing_exact += 1
            continue

        similarity_score = get_max_similarity_score(vector_store, chunk.page_content)
        if similarity_score >= config.similarity_threshold:
            skipped_existing_similar += 1
            continue

        vector_store.add_documents([chunk], ids=[chunk_id])
        inserted_count += 1

    if skipped_existing_exact:
        logger.info("Skipped %s chunks already stored in Chroma by exact content", skipped_existing_exact)
    if skipped_existing_similar:
        logger.info(
            "Skipped %s near-duplicate chunks with similarity >= %.3f",
            skipped_existing_similar,
            config.similarity_threshold,
        )
    logger.info("Inserted %s new chunks into Chroma", inserted_count)

    return vector_store


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args()

    logger.info("Collecting source files from %s", config.source_dir)
    source_files = collect_source_files(config)

    logger.info("Loading %s documents", len(source_files))
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
