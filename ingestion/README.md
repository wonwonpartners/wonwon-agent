# Ingestion

`ingestion/` 폴더는 RAG용 문서를 수집하고, 청크로 나누고, vector DB로 저장하는 스크립트를 모아둔 곳입니다.

## 구성

- `collect_company_docs.py`
  - 회사 웹사이트에서 공개 문서를 수집해 `data/company/<company>/` 아래에 저장합니다.
- `build_company_corpus.py`
  - 수집된 회사 문서를 청크로 분리하고 `vectordb/company/` Chroma DB를 생성합니다.
- `build_domain_corpus.py`
  - `data/domain/` 아래의 산업/기술 문서를 청크로 분리하고 `vectordb/domain/` Chroma DB를 생성합니다.

## 실행 위치

프로젝트 루트에서 실행합니다.

```bash
cd /Users/hyobin/skala/workspace/skala-gen-ai-project/wonwon-agent
```

`retrieval` 패키지를 import하므로 `python ingestion/...` 대신 `python -m ...` 형식으로 실행해야 합니다.

`build_company_corpus.py`와 `build_domain_corpus.py`의 기본 source 경로는 프로젝트 루트 기준으로 설정되어 있습니다.

## 1. 회사 문서 수집

예시:

```bash
python -m ingestion.collect_company_docs \
  --company-name xyz \
  --start-url https://xyz.ai \
  --output-dir data/company
```

주요 옵션:

- `--company-name`: 회사명 또는 slug
- `--start-url`: 크롤링 시작 URL
- `--output-dir`: 저장 루트. `python -m ingestion.collect_company_docs`로 실행할 때는 `data/company`처럼 프로젝트 루트 기준 경로를 명시하는 것을 권장
- `--max-pages`: 최대 크롤링 페이지 수, 기본값 `40`
- `--timeout-sec`: 요청 타임아웃, 기본값 `20`

출력:

- 문서 원본: `data/company/<company>/docs/`
- 메타데이터: `data/company/<company>/metadata/`

## 2. 회사 corpus 인덱싱

예시:

```bash
python -m ingestion.build_company_corpus
```

기본 source 경로:

- `data/company`

특정 경로나 컬렉션명을 지정하려면:

```bash
python -m ingestion.build_company_corpus \
  --source-dir data/company \
  --persist-dir vectordb/company \
  --collection-name company_corpus
```

주요 옵션:

- `--source-dir`: 회사 문서 루트
- `--persist-dir`: Chroma 저장 경로
- `--collection-name`: Chroma 컬렉션 이름
- `--chunk-size`: 청크 크기, 기본값 `1200`
- `--chunk-overlap`: 청크 overlap, 기본값 `200`
- `--batch-size`: 임베딩 배치 크기, 기본값 `32`

출력:

- `vectordb/company/`

## 3. 도메인 corpus 인덱싱

예시:

```bash
python -m ingestion.build_domain_corpus
```

기본 source 경로:

- `data/domain`

특정 경로나 컬렉션명을 지정하려면:

```bash
python -m ingestion.build_domain_corpus \
  --source-dir data/domain \
  --persist-dir vectordb/domain \
  --collection-name domain_corpus
```

출력:

- `vectordb/domain/`
- sidecar metadata는 `data/domain/metadata/*.json`에서 읽어 각 문서 metadata에 병합됩니다.

## 전체 흐름

회사 문서를 새로 수집하고 인덱싱할 때는 보통 아래 순서로 실행합니다.

```bash
python -m ingestion.collect_company_docs \
  --company-name xyz \
  --start-url https://xyz.ai

python -m ingestion.build_company_corpus
```

도메인 문서를 다시 인덱싱할 때는 아래처럼 실행합니다.

```bash
python -m ingestion.build_domain_corpus
```

## 참고

- `build_*_corpus.py`는 공용 `retrieval/` 설정을 사용합니다.
- 현재 vector store backend는 로컬 `Chroma` persist 디렉터리입니다.
- embedding 모델 기본값은 `Qwen/Qwen3-Embedding-0.6B`입니다.
- 현재 `build_*_corpus.py`는 기존 collection을 비우지 않고 `add_documents(...)`를 수행합니다. 같은 corpus를 다시 빌드하면 중복 chunk가 쌓일 수 있으므로 재색인 전 기존 `vectordb/company` 또는 `vectordb/domain` 정리 여부를 먼저 결정해야 합니다.
