import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

def _normalize_startup_name(name: str) -> str:
    return "".join((name or "").strip().lower().split())


@dataclass(frozen=True)
class ToolDocument:
    content: str
    source: str = "vector_rag"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TractionVectorIndex:
    def __init__(
        self,
        index_dir: Path,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        dimensions: int = 256,
    ):
        self.index_dir = index_dir
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.dimensions = int(dimensions)
        self.entries: List[Dict[str, Any]] = []
        self._client = None
        self._collection = None
        self._setup_runtime()

    def _setup_runtime(self):
        try:
            import chromadb
        except Exception:
            self._client = None
            self._collection = None
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chromadb = chromadb
        self._client = chromadb.PersistentClient(path=str(self.index_dir))
        self._collection = self._client.get_or_create_collection(
            name="traction_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, docs: Optional[List[Dict[str, Any]]] = None, force: bool = False):
        del docs
        if self._client is None or self._collection is None:
            self._setup_runtime()
        if force and self._client is not None:
            try:
                self._client.delete_collection("traction_chunks")
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name="traction_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        if self._collection is not None:
            self.load()

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().split()).lower()

    def load(self):
        if self._client is None or self._collection is None:
            self._setup_runtime()
        if self._collection is None:
            return False
        try:
            raw = self._collection.get(include=["metadatas", "documents"])
            metadatas = raw.get("metadatas") or []
            documents = raw.get("documents") or []
            entries: List[Dict[str, Any]] = []
            for metadata, document in zip(metadatas, documents):
                if not isinstance(metadata, dict):
                    continue
                entry = dict(metadata)
                entry["content"] = document or ""
                tags_raw = entry.get("tags", "[]")
                if isinstance(tags_raw, str):
                    try:
                        entry["tags"] = json.loads(tags_raw)
                    except Exception:
                        entry["tags"] = []
                entries.append(entry)
            self.entries = entries
            return True
        except Exception:
            return False
        return False

    def search(
        self,
        startup_name: str,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self._collection is None:
            return []
        startup_key = _normalize_startup_name(startup_name)
        query_text = (query or "").strip()
        normalized_query = self._normalize_text(query_text)
        tokens = [token for token in normalized_query.split(" ") if token]
        candidates: List[Dict[str, Any]] = []
        if not query_text and self._collection is not None:
            try:
                raw = self._collection.get(
                    where={"startup_key": startup_key},
                    include=["metadatas", "documents"],
                )
            except Exception:
                raw = self._collection.get(include=["metadatas", "documents"])
            candidates.extend(self._materialize_candidates(raw, startup_key, query_tokens=tokens))
        if query_text:
            try:
                raw = self._collection.query(
                    query_texts=[query_text],
                    where={"startup_key": startup_key},
                    n_results=min(max(k * 3, k), 30),
                    include=["metadatas", "documents", "distances"],
                )
                metadatas = (raw.get("metadatas") or [[]])[0]
                documents = (raw.get("documents") or [[]])[0]
                distances = (raw.get("distances") or [[]])[0]
                for metadata, document, distance in zip(metadatas, documents, distances):
                    entry = dict(metadata or {})
                    entry["content"] = document
                    entry["score"] = float(max(0.0, 1.0 - float(distance or 0.0)))
                    tags_raw = entry.get("tags", "[]")
                    if isinstance(tags_raw, str):
                        try:
                            entry["tags"] = json.loads(tags_raw)
                        except Exception:
                            entry["tags"] = []
                    candidates.append(entry)
            except Exception:
                pass
        if not candidates:
            try:
                raw = self._collection.get(
                    where={"startup_key": startup_key},
                    include=["metadatas", "documents"],
                )
            except Exception:
                raw = self._collection.get(include=["metadatas", "documents"])
            candidates.extend(self._materialize_candidates(raw, startup_key, query_tokens=tokens))
        return self._rank(startup_key, candidates, k)

    def _materialize_candidates(
        self,
        raw: Dict[str, Any],
        startup_key: str,
        query_tokens: List[str],
    ) -> List[Dict[str, Any]]:
        metadatas = raw.get("metadatas") or []
        documents = raw.get("documents") or []
        candidates: List[Dict[str, Any]] = []
        for metadata, document in zip(metadatas, documents):
            if not isinstance(metadata, dict):
                continue
            candidate_key = (metadata.get("startup_key") or "").strip()
            if candidate_key and candidate_key != startup_key:
                continue
            entry = dict(metadata)
            entry["content"] = document or ""
            tags_raw = entry.get("tags", "[]")
            if isinstance(tags_raw, str):
                try:
                    entry["tags"] = json.loads(tags_raw)
                except Exception:
                    entry["tags"] = []
            score = 0.0
            content = (document or "").lower()
            for token in query_tokens:
                if token and token in content:
                    score += 1.0
            if query_tokens:
                score = score / float(len(query_tokens))
            entry["score"] = score
            candidates.append(entry)
        return candidates

    def _rank(self, startup_key: str, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        exact = [c for c in candidates if c.get("startup_key") == startup_key]
        if not exact:
            exact = [c for c in candidates if startup_key in c.get("startup_key", "")]
        if not exact:
            exact = candidates
        exact.sort(key=lambda x: (x.get("score", 0.0), x.get("published_at", "")), reverse=True)
        return exact[:k]


class VectorTractionSearchTool:
    """Traction 분석용 벡터 검색 도구. query_traction만 외부 인터페이스로 유지."""

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        force_build: bool = False,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        dimensions: int = 256,
    ):
        self.corpus_dir = Path(corpus_dir or os.getenv("TRACTION_DATA_DIR", "data/traction"))
        self.index_dir = Path(index_dir or os.getenv("TRACTION_INDEX_PATH", ".data/traction_index"))
        self.force_build = force_build
        self.index = TractionVectorIndex(self.index_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap, dimensions=dimensions)
        self._initialized = False

    async def query_traction(self, startup_name: str, query: str) -> ToolDocument:
        await self._ensure_initialized()
        return self._query_internal(startup_name, query)

    async def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True
        loaded = self.index.load()
        if not loaded or not self.index.entries:
            self._rebuild()

    def _rebuild(self):
        self.index.build(force=True)

    def _query_internal(self, startup_name: str, query: str) -> ToolDocument:
        results = self.index.search(startup_name=startup_name, query=query, k=4)
        if not results:
            return ToolDocument(
                content=(
                    f"{startup_name}에 대한 traction 근거 데이터가 현재 corpus에서 조회되지 않습니다. "
                    f"추가 검색 또는 인덱스 재생성이 필요합니다."
                ),
                source="vector_rag_empty",
                metadata={
                    "startup_name": startup_name,
                    "query": query,
                    "results": [],
                },
            )

        snippets: List[str] = []
        result_meta: List[Dict[str, Any]] = []
        for i, item in enumerate(results, start=1):
            snippets.append(
                f"[{i}] source={item.get('source','')} date={item.get('published_at','')} "
                f"score={item.get('score',0.0):.3f} title={item.get('title','')}\n{item.get('content','')}"
            )
            result_meta.append(
                {
                    "title": item.get("title", ""),
                    "source": item.get("source", ""),
                    "source_type": item.get("source_type", ""),
                    "published_at": item.get("published_at", ""),
                    "url": item.get("url", ""),
                    "chunk_no": item.get("chunk_no"),
                    "score": item.get("score", 0.0),
                    "content": item.get("content", ""),
                }
            )
        return ToolDocument(
            content="\n\n".join(snippets),
            source="vector_rag",
            metadata={
                "startup_name": startup_name,
                "query": query,
                "results": result_meta,
            },
        )


class MockSearchTool:
    """Mock 기반 임시 더미 응답 대신 실제 검색 도구를 그대로 노출한다."""

    def __init__(self, vector_tool: Optional["VectorTractionSearchTool"] = None):
        self.vector_tool = vector_tool or VectorTractionSearchTool()

    async def query_team(self, startup_name: str, query: str) -> ToolDocument:
        return await self.vector_tool.query_traction(startup_name=startup_name, query=query)

    async def query_product(self, startup_name: str, query: str) -> ToolDocument:
        return await self.vector_tool.query_traction(startup_name=startup_name, query=query)

    async def query_risk(self, startup_name: str, query: str) -> ToolDocument:
        return await self.vector_tool.query_traction(startup_name=startup_name, query=query)

    async def query_traction(self, startup_name: str, query: str) -> ToolDocument:
        return await self.vector_tool.query_traction(startup_name=startup_name, query=query)


class FirecrawlTractionSearchTool:
    """Firecrawl 기반 traction 검색 도구. 검색 후 상위 URL을 다시 scrape한다."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        scrape_formats: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.max_results = int(max_results)
        self.scrape_formats = scrape_formats or ["markdown"]
        self.client = None
        self.ready = False

        if not self.api_key:
            return
        try:
            from firecrawl import Firecrawl

            self.client = Firecrawl(api_key=self.api_key)
            self.ready = True
        except Exception:
            self.client = None
            self.ready = False

    async def query_traction(self, startup_name: str, query: str) -> ToolDocument:
        if not self.ready:
            return ToolDocument(
                content=(
                    "Firecrawl API 키 미설정이거나 Firecrawl 라이브러리를 로드하지 못해 실시간 검색이 불가능합니다. "
                    "FIRECRAWL_API_KEY를 확인하거나 VectorTractionSearchTool 경로로 폴백 설정하세요."
                ),
                source="firecrawl_disabled",
                metadata={"startup_name": startup_name, "query": query, "results": []},
            )

        def _search_and_scrape():
            raw = self.client.search(query=query, limit=self.max_results)
            results = self._normalize_results(raw)
            enriched: List[Dict[str, Any]] = []
            for item in results[: self.max_results]:
                url = item.get("url", "") or item.get("source", "")
                if not url:
                    continue
                scraped_text = item.get("markdown", "") or item.get("content", "") or item.get("description", "")
                if not scraped_text:
                    try:
                        scraped = self.client.scrape(url, formats=self.scrape_formats)
                        scraped_text = self._extract_scrape_text(scraped)
                    except Exception:
                        scraped_text = item.get("description", "") or ""
                enriched.append({**item, "scraped_text": scraped_text})
            return enriched

        try:
            started_at = time.perf_counter()
            results = await asyncio.to_thread(_search_and_scrape)
            elapsed = time.perf_counter() - started_at
            print(
                f"[timing] firecrawl_query startup={startup_name} "
                f"elapsed={elapsed:.3f}s results={len(results)}"
            )
        except Exception as exc:
            return ToolDocument(
                content=f"Firecrawl 검색 실행 중 오류: {exc}",
                source="firecrawl_error",
                metadata={"startup_name": startup_name, "query": query, "results": []},
            )

        if not results:
            return ToolDocument(
                content=f"{startup_name}에 대한 실시간 traction 검색 결과가 없습니다.",
                source="firecrawl_empty",
                metadata={"startup_name": startup_name, "query": query, "results": []},
            )

        snippets: List[str] = []
        result_meta: List[Dict[str, Any]] = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "")
            source = item.get("url", "") or item.get("source", "")
            score = item.get("score")
            score_text = "0"
            if isinstance(score, (int, float)):
                score_text = f"{float(score):.3f}"
            elif isinstance(score, str):
                try:
                    score_text = f"{float(score):.3f}"
                except ValueError:
                    score_text = "0"
            content = item.get("scraped_text", "") or item.get("content", "") or item.get("description", "")
            snippets.append(
                f"[{idx}] source={source} title={title} score={score_text}\n{content}"
            )
            result_meta.append(
                {
                    "title": title,
                    "source": source,
                    "source_type": item.get("type", "web"),
                    "published_at": item.get("published_date", "") or item.get("published_at", ""),
                    "url": item.get("url", ""),
                    "chunk_no": item.get("chunk_no", None),
                    "score": score if isinstance(score, (int, float)) else 0.0,
                }
            )

        return ToolDocument(
            content="\n\n".join(snippets),
            source="firecrawl",
            metadata={"startup_name": startup_name, "query": query, "results": result_meta},
        )

    def _normalize_results(self, raw: Any) -> List[Dict[str, Any]]:
        raw = self._to_plain_object(raw)
        if not raw:
            return []
        if isinstance(raw, dict):
            for key in ("web", "news", "images", "data", "results"):
                rows = raw.get(key)
                if isinstance(rows, list):
                    normalized_rows: List[Dict[str, Any]] = []
                    for row in rows:
                        row_obj = self._to_plain_object(row)
                        if isinstance(row_obj, dict):
                            normalized_rows.append(row_obj)
                    return normalized_rows
            return []
        if isinstance(raw, list):
            normalized_rows = []
            for row in raw:
                row_obj = self._to_plain_object(row)
                if isinstance(row_obj, dict):
                    normalized_rows.append(row_obj)
            return normalized_rows
        return []

    def _extract_scrape_text(self, scraped: Any) -> str:
        scraped = self._to_plain_object(scraped)
        if isinstance(scraped, dict):
            for key in ("markdown", "content", "text", "html"):
                value = scraped.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            data = scraped.get("data")
            if isinstance(data, dict):
                for key in ("markdown", "content", "text", "html"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        return ""

    def _to_plain_object(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return value
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                return value
        return value


class TractionWebVectorTool:
    """traction 조회는 웹(Firecrawl) 우선, 없으면 VectorRAG로 폴백."""

    def __init__(
        self,
        vector_tool: VectorTractionSearchTool,
        web_tool: Optional[FirecrawlTractionSearchTool] = None,
        use_web_first: bool = True,
        fallback_to_vector: bool = True,
    ):
        self.vector_tool = vector_tool
        self.web_tool = web_tool
        self.use_web_first = use_web_first
        self.fallback_to_vector = fallback_to_vector


    async def query_traction_vector(self, startup_name: str, query: str) -> ToolDocument:
        return await self.vector_tool.query_traction(startup_name=startup_name, query=query)

    async def query_traction_web(self, startup_name: str, query: str) -> ToolDocument:
        if self.web_tool is None:
            return ToolDocument(
                content=f"{startup_name} traction 웹 검색 도구가 비활성 상태입니다.",
                source="firecrawl_disabled",
                metadata={"startup_name": startup_name, "query": query, "results": []},
            )
        return await self.web_tool.query_traction(startup_name=startup_name, query=query)

    async def query_traction(self, startup_name: str, query: str) -> ToolDocument:
        if self.use_web_first and self.web_tool is not None:
            web_doc = await self.web_tool.query_traction(startup_name=startup_name, query=query)
            if web_doc.source not in {"firecrawl_empty", "firecrawl_disabled", "firecrawl_error"}:
                return web_doc

        if self.fallback_to_vector:
            return await self.vector_tool.query_traction(startup_name=startup_name, query=query)

        return ToolDocument(
            content=f"{startup_name} traction 실시간 검색/벡터 검색 모두 비활성 상태입니다.",
            source="traction_search_unavailable",
            metadata={"startup_name": startup_name, "query": query, "results": []},
        )
