import re
import time
from typing import Any, Dict, List, Optional, TypedDict, cast

from tools import (
    FirecrawlTractionSearchTool,
    ToolDocument,
    TractionWebVectorTool,
    VectorTractionSearchTool,
)
from utils._shared import run_structured_from_llm

from .state import TractionInputState, TractionState


class TractionSufficiencyCheck(TypedDict):
    is_sufficient: bool
    reason: str
    missing_signals: List[str]


class TractionSearchQualityCheck(TypedDict):
    is_acceptable: bool
    reason: str
    low_quality_signals: List[str]


class TractionAgent:
    NODE_NAME = "traction_agent"
    STATE_KEY = "traction"

    def __init__(self, llm: Optional[Any] = None, tool: Optional[Any] = None):
        self.llm = llm
        self.tool = tool or TractionWebVectorTool(
            vector_tool=VectorTractionSearchTool(),
            web_tool=FirecrawlTractionSearchTool(),
        )

    def _build_name_variants(self, startup_name: str) -> List[str]:
        variants: List[str] = []
        normalized = (startup_name or "").strip()
        if normalized:
            variants.append(normalized)
        compact = normalized.replace(" ", "")
        if compact and compact not in variants:
            variants.append(compact)
        return variants

    async def _build_queries(self, startup_name: str) -> Dict[str, str]:
        name_variants = self._build_name_variants(startup_name)
        exact_name_clause = " OR ".join(f'"{name}"' for name in name_variants) or f'"{startup_name}"'
        return {
            "hiring": f'({exact_name_clause}) (채용 OR hiring OR careers OR "Field Engineer" OR "field service engineer")',
            "funding": f'({exact_name_clause}) (투자 OR funding OR 라운드 OR seed OR series OR 조달 OR 매출 OR 수주)',
            "partnership": f'({exact_name_clause}) (파트너십 OR partnership OR 협력 OR 제휴 OR MOU)',
            "customer": f'({exact_name_clause}) (고객사 OR 도입 OR 납품 OR 사용처 OR 파일럿 OR PoC OR 상용화)',
        }

    async def __call__(self, state: TractionInputState) -> Dict[str, Any]:
        startup_name = state["startup_name"]
        queries = await self._build_queries(startup_name)
        vector_contexts = await self._query_signal_contexts(startup_name, queries, mode="vector")
        vector_context_text = self._join_context_blocks(
            startup_name=startup_name,
            queries=queries,
            contexts=vector_contexts,
            channel_label="vector",
        )

        is_sufficient, sufficiency_reason, missing_signals = await self._assess_vector_sufficiency(
            startup_name=startup_name,
            queries=queries,
            vector_contexts=vector_contexts,
            vector_context_text=vector_context_text,
        )

        merged_contexts = dict(vector_contexts)

        if not is_sufficient:
            target_queries = {
                signal_type: queries[signal_type]
                for signal_type in (missing_signals or list(queries.keys()))
                if signal_type in queries
            }
            web_contexts = await self._query_signal_contexts(startup_name, target_queries, mode="web")
            merged_contexts = self._merge_context_maps(merged_contexts, web_contexts)

        merged_context_text = self._join_context_blocks(
            startup_name=startup_name,
            queries=queries,
            contexts=merged_contexts,
            channel_label="merged",
        )
        quality_ok, quality_reason, low_quality_signals = await self._assess_search_quality(
            startup_name=startup_name,
            queries=queries,
            contexts=merged_contexts,
            context_text=merged_context_text,
        )
        retry_queries: Dict[str, str] = {}
        if not quality_ok:
            retry_queries = self._build_retry_queries(
                startup_name=startup_name,
                signal_types=low_quality_signals or list(queries.keys()),
                quality_reason=quality_reason,
            )
            retry_contexts = await self._query_signal_contexts(startup_name, retry_queries, mode="web")
            merged_contexts = self._merge_context_maps(merged_contexts, retry_contexts)
            merged_context_text = self._join_context_blocks(
                startup_name=startup_name,
                queries={**queries, **retry_queries},
                contexts=merged_contexts,
                channel_label="merged",
            )
            quality_ok, quality_reason, low_quality_signals = await self._assess_search_quality(
                startup_name=startup_name,
                queries=queries,
                contexts=merged_contexts,
                context_text=merged_context_text,
            )

        final_is_sufficient, final_sufficiency_reason, final_missing_signals = await self._assess_vector_sufficiency(
            startup_name=startup_name,
            queries=queries,
            vector_contexts=merged_contexts,
            vector_context_text=merged_context_text,
        )

        context_blocks: List[str] = []
        if not final_is_sufficient:
            context_blocks.append(
                f"[sufficiency_check] 최종 traction 충분성 판단: 불충분 / "
                f"사유: {final_sufficiency_reason or sufficiency_reason or '판단 근거 미제공'} / "
                f"부족 신호: {', '.join(final_missing_signals or missing_signals) if (final_missing_signals or missing_signals) else '없음'}"
            )
        context_blocks.append(
            f"[quality_check] 검색 품질 판단: {'양호' if quality_ok else '보강 필요'} / "
            f"사유: {quality_reason or '판단 근거 미제공'} / "
            f"저품질 신호: {', '.join(low_quality_signals) if low_quality_signals else '없음'}"
        )
        if retry_queries:
            context_blocks.append(
                "[retry_queries] "
                + " | ".join(f"{signal}={query}" for signal, query in retry_queries.items())
            )
        context_blocks.append(merged_context_text)
        context_text = "\n\n".join(block for block in context_blocks if block.strip())
        evidence_sources = self._collect_evidence_sources(merged_contexts)
        prompt = (
            "Return JSON matching TractionState keys: partnerships, hiring_analysis, funding_velocity, traction_summary.\n"
            "평가 기준은 다소 완화해서 적용하라. 회사명과 직접 연결된 외부 근거가 일부라도 확인되면 "
            "보수적으로 전부 부족하다고 쓰지 말고, 확인된 traction 신호를 중심으로 구조화하라. "
            "모든 축에서 정량 지표가 완비될 필요는 없다. 파트너십, 투자/펀딩, 고객/도입, 채용 중 "
            "2개 이상 신호가 확인되거나 1개의 강한 신호와 보조 신호가 있으면 충분히 traction 결과를 작성하라. "
            "다음 정도의 결과보다 다소 낮은 품질이어도 통과로 보라: "
            "파트너십 3~4건, 채용 공고 수 4건 내외, 시리즈 B 251억원 및 누적 550억원 수준의 투자 이력, "
            "그리고 '자율주행 로봇 서비스 상용화 역량을 입증하고 있으며 다양한 파트너십을 통해 사업을 확장하고 있다' 수준의 summary. "
            "'외부 트랙션 데이터가 부족하다'는 표현은 회사 직접 관련 근거가 대부분 없거나 신호가 거의 비어 있을 때만 사용하라.\n"
            f"Startup: {startup_name}\n"
            f"Context:\n{context_text}"
        )
        started_at = time.perf_counter()
        llm_result = await run_structured_from_llm(self.llm, TractionState, prompt)
        elapsed = time.perf_counter() - started_at
        print(
            f"[timing] llm_traction_state startup={startup_name} "
            f"elapsed={elapsed:.3f}s result={'ok' if isinstance(llm_result, dict) else 'empty'}"
        )
        if isinstance(llm_result, dict) and self._valid(llm_result):
            return {
                self.STATE_KEY: self._coerce(
                    llm_result,
                    context_text,
                    evidence_sources=evidence_sources,
                )
            }

        return {}

    async def _query_signal_contexts(
        self,
        startup_name: str,
        queries: Dict[str, str],
        mode: str,
    ) -> Dict[str, ToolDocument]:
        contexts: Dict[str, ToolDocument] = {}
        for signal_type, query in queries.items():
            contexts[signal_type] = await self._query_single_context(
                startup_name=startup_name,
                query=query,
                mode=mode,
            )
        return contexts

    async def _query_single_context(self, startup_name: str, query: str, mode: str) -> ToolDocument:
        if mode == "vector" and hasattr(self.tool, "query_traction_vector"):
            return await self.tool.query_traction_vector(startup_name, query)
        if mode == "web" and hasattr(self.tool, "query_traction_web"):
            return await self.tool.query_traction_web(startup_name, query)
        return await self.tool.query_traction(startup_name, query)

    def _join_context_blocks(
        self,
        startup_name: str,
        queries: Dict[str, str],
        contexts: Dict[str, ToolDocument],
        channel_label: str,
    ) -> str:
        blocks: List[str] = []
        for signal_type, query in queries.items():
            context = contexts.get(signal_type)
            if context is None:
                continue
            blocks.append(
                self._build_context(
                    startup_name=startup_name,
                    context=context,
                    query=query,
                    signal_type=signal_type,
                    channel_label=channel_label,
                )
            )
        return "\n\n".join(block for block in blocks if block.strip())

    async def _assess_vector_sufficiency(
        self,
        startup_name: str,
        queries: Dict[str, str],
        vector_contexts: Dict[str, ToolDocument],
        vector_context_text: str,
    ) -> tuple[bool, str, List[str]]:
        if self.llm is not None:
            prompt = (
                "다음은 스타트업 traction 평가를 위해 현재까지 수집한 검색 결과다. "
                "이 결과만으로 외부 traction을 작성하기에 충분한지 판단하라. "
                "판정 기준은 다소 완화해서 적용한다. hiring, funding, partnership, customer 네 신호가 "
                "모두 완벽하게 채워질 필요는 없고, 회사와 직접 관련된 근거가 일부라도 확인되면 충분으로 볼 수 있다. "
                "특히 파트너십/투자 같은 강한 신호가 1~2개 있고, 나머지 신호가 약하게라도 뒷받침되면 충분으로 판단하라. "
                "반대로 대부분이 빈 결과이거나 회사 관련성이 약할 때만 불충분으로 판단하라.\n"
                "부족하면 missing_signals에 hiring/funding/partnership/customer 중 실제로 추가 검색이 필요한 항목만 넣어라. "
                "애매한 경우에는 지나치게 보수적으로 불충분 판정을 내리지 마라.\n"
                f"Startup: {startup_name}\n"
                f"Signals: {list(queries.keys())}\n"
                f"Collected Context:\n{vector_context_text}"
            )
            started_at = time.perf_counter()
            llm_result = await run_structured_from_llm(self.llm, TractionSufficiencyCheck, prompt)
            elapsed = time.perf_counter() - started_at
            print(
                f"[timing] llm_sufficiency startup={startup_name} "
                f"elapsed={elapsed:.3f}s result={'ok' if isinstance(llm_result, dict) else 'empty'}"
            )
            if isinstance(llm_result, dict):
                is_sufficient = bool(llm_result.get("is_sufficient"))
                reason = str(llm_result.get("reason", "")).strip()
                missing_signals = [
                    str(item).strip()
                    for item in (llm_result.get("missing_signals") or [])
                    if str(item).strip() in queries
                ]
                if is_sufficient or missing_signals:
                    return is_sufficient, reason or "LLM sufficient check", missing_signals

        return False, "LLM sufficient check unavailable", list(queries.keys())

    async def _assess_search_quality(
        self,
        startup_name: str,
        queries: Dict[str, str],
        contexts: Dict[str, ToolDocument],
        context_text: str,
    ) -> tuple[bool, str, List[str]]:
        if self.llm is not None:
            prompt = (
                "다음은 스타트업 traction 검색 결과다. "
                "기업 관련성, 신호별 구체성, 중복/노이즈 여부를 기준으로 검색 품질을 평가하라. "
                "단, 기준은 다소 완화해서 적용한다. 일부 중복이나 메타데이터 노이즈가 있어도 "
                "회사와 직접 관련된 투자, 파트너십, 고객 도입, 채용 정보가 읽히면 양호로 본다. "
                "검색 결과가 완벽히 정돈되어 있지 않아도 실제 회사 관련 근거가 있으면 통과시켜라. "
                "거의 전부 무관하거나 빈 결과일 때만 품질이 낮다고 판단하라. "
                "검색 품질이 낮다면 low_quality_signals에 hiring/funding/partnership/customer 중 "
                "재검색이 필요한 항목만 넣어라. 애매한 경우에는 불필요한 재검색을 줄이는 방향으로 판단하라.\n"
                f"Startup: {startup_name}\n"
                f"Signals: {list(queries.keys())}\n"
                f"Context:\n{context_text}"
            )
            started_at = time.perf_counter()
            llm_result = await run_structured_from_llm(self.llm, TractionSearchQualityCheck, prompt)
            elapsed = time.perf_counter() - started_at
            print(
                f"[timing] llm_quality startup={startup_name} "
                f"elapsed={elapsed:.3f}s result={'ok' if isinstance(llm_result, dict) else 'empty'}"
            )
            if isinstance(llm_result, dict):
                is_acceptable = bool(llm_result.get("is_acceptable"))
                reason = str(llm_result.get("reason", "")).strip()
                low_quality_signals = [
                    str(item).strip()
                    for item in (llm_result.get("low_quality_signals") or [])
                    if str(item).strip() in queries
                ]
                if is_acceptable or low_quality_signals:
                    return is_acceptable, reason or "LLM quality check", low_quality_signals

        return True, "LLM quality check unavailable", []

    def _heuristic_vector_sufficiency(
        self,
        queries: Dict[str, str],
        vector_contexts: Dict[str, ToolDocument],
    ) -> tuple[bool, str, List[str]]:
        missing_signals: List[str] = []
        strong_signal_count = 0
        for signal_type in queries:
            context = vector_contexts.get(signal_type)
            if context is None:
                missing_signals.append(signal_type)
                continue
            content = (context.content or "").strip()
            results = context.metadata.get("results", []) if isinstance(context.metadata, dict) else []
            limited_markers = ("조회되지 않습니다", "추가 검색", "정보가 제한적", "근거가 부족")
            if context.source == "vector_rag_empty" or not results or any(marker in content for marker in limited_markers):
                missing_signals.append(signal_type)
                continue
            strong_signal_count += 1

        is_sufficient = strong_signal_count >= 3 and len(missing_signals) <= 1
        reason = (
            f"vector 신호 {strong_signal_count}개 확보, 부족 신호 {len(missing_signals)}개"
            if is_sufficient
            else f"vector 신호가 부족하여 web 보강 필요: {', '.join(missing_signals) if missing_signals else '전반적 부족'}"
        )
        return is_sufficient, reason, missing_signals

    def _heuristic_search_quality(
        self,
        startup_name: str,
        queries: Dict[str, str],
        contexts: Dict[str, ToolDocument],
    ) -> tuple[bool, str, List[str]]:
        low_quality_signals: List[str] = []
        name_variants = self._build_name_variants(startup_name)
        for signal_type in queries:
            context = contexts.get(signal_type)
            if context is None:
                low_quality_signals.append(signal_type)
                continue
            content = (context.content or "").strip()
            results = context.metadata.get("results", []) if isinstance(context.metadata, dict) else []
            if not content or not results:
                low_quality_signals.append(signal_type)
                continue
            lowered_content = content.lower()
            matched = any(name.lower() in lowered_content for name in name_variants if name)
            if not matched:
                low_quality_signals.append(signal_type)

        is_acceptable = not low_quality_signals
        reason = (
            "모든 신호에서 기업명 관련 근거가 확인됨"
            if is_acceptable
            else f"기업 관련성 또는 근거 밀도가 약한 신호 존재: {', '.join(low_quality_signals)}"
        )
        return is_acceptable, reason, low_quality_signals

    def _build_retry_queries(
        self,
        startup_name: str,
        signal_types: List[str],
        quality_reason: str,
    ) -> Dict[str, str]:
        name_variants = self._build_name_variants(startup_name)
        primary_name = name_variants[0] if name_variants else startup_name
        compact_name = name_variants[1] if len(name_variants) > 1 else primary_name.replace(" ", "")
        retry_queries: Dict[str, str] = {}
        for signal_type in signal_types:
            if signal_type == "hiring":
                retry_queries[signal_type] = (
                    f'"{primary_name}" "{compact_name}" 채용 공고 현장 엔지니어 Field Engineer careers jobs'
                )
            elif signal_type == "funding":
                retry_queries[signal_type] = (
                    f'"{primary_name}" 투자 유치 funding seed series 매출 수주 기사 보도자료'
                )
            elif signal_type == "partnership":
                retry_queries[signal_type] = (
                    f'"{primary_name}" 파트너십 제휴 협력 MOU 보도자료 레퍼런스'
                )
            elif signal_type == "customer":
                retry_queries[signal_type] = (
                    f'"{primary_name}" 고객사 도입 납품 사용처 파일럿 PoC 상용화 사례'
                )
        if quality_reason:
            retry_queries = {
                signal_type: f"{query} 품질 보강 목적: {quality_reason}"
                for signal_type, query in retry_queries.items()
            }
        return retry_queries

    def _build_context(
        self,
        startup_name: str,
        context: ToolDocument,
        query: str,
        signal_type: str,
        channel_label: str,
    ) -> str:
        if not context.content:
            return (
                f"[{channel_label}:{signal_type}] {startup_name} traction 근거가 부족합니다. "
                f"쿼리({query}) 기준으로 재검색 또는 데이터 보강이 필요합니다."
            )
        evidence = context.metadata.get("results", [])
        if not evidence:
            return f"[{channel_label}:{signal_type}] query={query}\nsource={context.source}\n{context.content}"
        lines = [f"[{channel_label}:{signal_type}] query={query}", f"source={context.source}", context.content]
        lines.append("")
        lines.append("근거 메타데이터:")
        for idx, item in enumerate(evidence, start=1):
            score = item.get("score")
            if isinstance(score, float):
                score_text = f"{score:.3f}"
            else:
                score_text = str(score or "0")
            lines.append(
                f"{idx}) source={item.get('source','')} source_type={item.get('source_type','')} "
                f"date={item.get('published_at','-')} score={score_text} url={item.get('url','-')}"
            )
        return "\n".join(lines)

    def _merge_source_channels(self, sources: List[str]) -> str:
        normalized: List[str] = []
        for source in sources:
            cleaned = str(source or "").strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return "+".join(normalized) if normalized else "unknown"

    def _merge_documents(self, base: ToolDocument, extra: ToolDocument) -> ToolDocument:
        merged_content = "\n\n".join(part for part in [base.content, extra.content] if part)
        merged_results = []
        seen = set()
        for document in (base, extra):
            results = document.metadata.get("results", []) if isinstance(document.metadata, dict) else []
            for item in results:
                key = (
                    item.get("url", ""),
                    item.get("title", ""),
                    item.get("published_at", ""),
                    item.get("source", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_results.append(item)
        return ToolDocument(
            content=merged_content,
            source=self._merge_source_channels([base.source, extra.source]),
            metadata={"results": merged_results},
        )

    def _merge_context_maps(
        self,
        base_contexts: Dict[str, ToolDocument],
        extra_contexts: Dict[str, ToolDocument],
    ) -> Dict[str, ToolDocument]:
        merged = dict(base_contexts)
        for signal_type, context in extra_contexts.items():
            if signal_type in merged:
                merged[signal_type] = self._merge_documents(merged[signal_type], context)
            else:
                merged[signal_type] = context
        return merged

    def _collect_evidence_sources(
        self,
        contexts: Dict[str, ToolDocument],
    ) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        for signal_type, context in contexts.items():
            results = context.metadata.get("results", []) if isinstance(context.metadata, dict) else []
            query = str(context.metadata.get("query", "")) if isinstance(context.metadata, dict) else ""
            for item in results:
                result = cast(Dict[str, Any], item)
                url = str(result.get("url") or result.get("source") or "").strip()
                title = str(result.get("title", "")).strip()
                published_at = str(result.get("published_at", "")).strip()
                source = str(result.get("source", "")).strip()
                dedupe_key = (url, title, published_at, signal_type)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                collected.append(
                    {
                        "source_type": str(result.get("source_type", "web")).strip() or "web",
                        "signal_type": signal_type,
                        "query": query,
                        "title": title,
                        "publisher": self._extract_publisher(url or source),
                        "published_at": published_at,
                        "url": url,
                        "source": source,
                        "score": result.get("score", 0.0),
                    }
                )

        return collected

    def _valid(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload.get("partnerships"), list):
            return False
        if not isinstance(payload.get("hiring_analysis"), dict):
            return False
        funding_velocity = payload.get("funding_velocity")
        if not isinstance(funding_velocity, list) or not any(str(item).strip() for item in funding_velocity):
            return False
        return bool(payload.get("traction_summary"))

    def _coerce(
        self,
        payload: Dict[str, Any],
        raw_text: str = "",
        *,
        evidence_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> TractionState:
        partnerships = payload.get("partnerships", [])
        if not partnerships:
            partnerships = ["공개 파트너십 정보가 제한적입니다."]
        partnerships = [p for p in partnerships if str(p).strip()]
        if not partnerships:
            partnerships = ["공개 파트너십 정보가 제한적입니다."]

        hiring = payload.get("hiring_analysis", {}) or {}
        if not isinstance(hiring, dict):
            hiring = {}
        if "field_engineer_ratio" not in hiring:
            ratio = self._extract_ratio(raw_text)
            hiring["field_engineer_ratio"] = ratio
        if "field_engineer_count" not in hiring:
            hiring["field_engineer_count"] = self._extract_field_count(raw_text)
        if "hiring_trend_3m" not in hiring:
            hiring["hiring_trend_3m"] = self._extract_hiring_trend(raw_text)

        if not isinstance(hiring.get("field_engineer_ratio"), (int, float)):
            hiring["field_engineer_ratio"] = max(0.0, min(1.0, 0.1))

        raw_funding_velocity = payload.get("funding_velocity", [])
        if isinstance(raw_funding_velocity, list):
            funding_velocity = [str(item).strip() for item in raw_funding_velocity if str(item).strip()]
        elif isinstance(raw_funding_velocity, str) and raw_funding_velocity.strip():
            funding_velocity = [raw_funding_velocity.strip()]
        else:
            funding_velocity = self._extract_funding_velocity_signals(raw_text)
        if not funding_velocity:
            funding_velocity = ["채용/투자/수주 신호가 제한적이며 추가 확인 필요"]

        summary = (
            payload.get("traction_summary", "")
            or f"채널/고객 기반, 채용, 투자 흐름이 부분적으로 확인됨. 핵심 수치: {self._extract_funding_signal(raw_text)}"
        )
        return TractionState(
            partnerships=[str(p).strip() for p in partnerships][:5],
            hiring_analysis={
                "field_engineer_ratio": round(float(hiring.get("field_engineer_ratio", 0.0)), 3),
                "field_engineer_count": int(hiring.get("field_engineer_count", 0) or 0),
                "hiring_trend_3m": hiring.get("hiring_trend_3m", 0),
            },
            funding_velocity=funding_velocity,
            traction_summary=summary.strip() or "traction 분석 보완 필요",
            evidence_sources=list(evidence_sources or []),
        )

    def _extract_publisher(self, url: str) -> str:
        if not url:
            return ""
        normalized = re.sub(r"^https?://", "", url)
        normalized = normalized.split("/", 1)[0]
        return normalized.replace("www.", "")

    def _parse_context(self, content: str) -> Dict[str, Any]:
        ratio = self._extract_ratio(content)
        hiring_count = self._extract_field_count(content)
        partnerships = self._extract_partnerships(content)
        return {
            "partnerships": partnerships,
            "hiring_analysis": {
                "field_engineer_ratio": ratio,
                "field_engineer_count": hiring_count,
                "hiring_trend_3m": self._extract_hiring_trend(content),
            },
            "funding_velocity": self._extract_funding_velocity_signals(content),
            "traction_summary": self._build_summary(content, partnerships, ratio, hiring_count),
        }

    def _extract_ratio(self, text: str) -> float:
        match = re.search(r"Field Engineer 채용 비중은 ([0-9]+(?:\.[0-9]+)?)%?", text)
        if match:
            v = float(match.group(1))
            if v <= 1:
                return float(v)
            return round(v / 100.0, 3)
        match = re.search(r"비중은\s*([0-9]+(?:\.[0-9]+)?)%", text)
        if match:
            return round(float(match.group(1)) / 100.0, 3)
        return 0.0

    def _extract_field_count(self, text: str) -> int:
        match = re.search(r"Field Engineer[^\d]{0,20}(\d+)", text)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return 0
        return 0

    def _extract_hiring_trend(self, text: str) -> int:
        match = re.search(r"월 (\d+)건", text)
        if match:
            return int(match.group(1))
        return 0

    def _extract_velocity(self, text: str) -> float:
        match = re.search(r"(\d+\.?\d*)x", text)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return 0.0
        return 0.0

    def _extract_funding_signal(self, text: str) -> str:
        for token in ("시리즈", "투자", "전략적", "펀딩", "라운드", "협의"):
            if token in text:
                return token
        return "채용/파트너십 위주"

    def _extract_partnerships(self, text: str) -> List[str]:
        parts = []
        for line in re.split(r"[;,.]", text):
            cleaned = line.strip()
            if not cleaned:
                continue
            if any(k in cleaned for k in ("파트너", "협력", "협업", "파트너십", "MOU")):
                parts.append(cleaned)
        if not parts:
            return ["공개 파트너십 정보가 제한적입니다."]
        return parts[:4]

    def _extract_funding_velocity_signals(self, text: str) -> List[str]:
        signals: List[str] = []
        normalized_text = text or ""

        if "Pre-seed" in normalized_text and "매출" in normalized_text:
            signals.append("Pre-seed 대비 매출 성장 추정치 존재")
        if "전략적" in normalized_text and "투자자" in normalized_text:
            signals.append("전략적 투자 협의 단계 관측")
        if "조달" in normalized_text:
            signals.append("운영 자금 조달 공고 확인")
        if "수주" in normalized_text:
            signals.append("수주 잔고 또는 납품 일정 관련 신호 확인")

        for sentence in re.split(r"[\n.!?]", normalized_text):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            if any(token in cleaned for token in ("투자", "라운드", "펀딩", "전략적", "조달", "매출", "수주", "성장")):
                signals.append(cleaned)

        if not signals:
            fallback_signal = self._extract_funding_signal(normalized_text)
            if fallback_signal:
                signals.append(fallback_signal)

        deduped: List[str] = []
        for item in signals:
            if item not in deduped:
                deduped.append(item)
        return deduped[:5]

    def _build_summary(self, content: str, partnerships: List[str], ratio: float, hiring_count: int) -> str:
        p_text = ", ".join(partnerships[:3]) if partnerships else "파트너십 미확보"
        return (
            f"채널 측면에서 {p_text}가 확인되며, Field Engineer 비중은 {ratio:.0%} "
            f"(최근 월 {hiring_count}건 채용 언급) 기반의 상용화 단계를 검토한다."
        )
