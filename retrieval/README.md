# Retrieval

`retrieval/` 패키지는 ingestion과 agents가 공통으로 사용하는 retrieval 계층입니다.

이 패키지의 역할은 다음과 같습니다.

- vector store 설정을 한곳에서 관리
- embedding 생성 로직 공통화
- Chroma vector store 생성 방식 통일
- agent용 retriever 생성 진입점 제공

## 파일 구성

- `config.py`
  - vector store별 설정 정의
  - 현재 `company`, `domain` 설정 포함
- `embeddings.py`
  - 공용 HuggingFace embedding 객체 생성
- `vectorstore.py`
  - Chroma vector store 생성 factory
- `retrievers.py`
  - LangChain retriever 생성 factory
- `__init__.py`
  - 외부에서 사용할 public API export

## 현재 설정

기본 제공 store:

- `company`
  - collection: `company_corpus`
  - persist dir: `vectordb/company`
- `domain`
  - collection: `domain_corpus`
  - persist dir: `vectordb/domain`

기본 embedding 모델:

- `Qwen/Qwen3-Embedding-0.6B`

## 사용 방법

### 1. 설정 가져오기

```python
from retrieval import get_vector_store_config

config = get_vector_store_config("company")
print(config.collection_name)
print(config.persist_dir)
```

### 2. Vector store 열기

```python
from retrieval import get_vector_store

vector_store = get_vector_store("company")
```

### 3. Retriever 만들기

```python
from retrieval import get_retriever

retriever = get_retriever("company", k=5)
docs = retriever.invoke("이 회사의 주요 제품은?")
```

### 4. search kwargs 전달하기

```python
from retrieval import get_retriever

retriever = get_retriever(
    "company",
    k=5,
    search_kwargs={"filter": {"company": "xyz"}},
)
```

## 의존 방향

의도한 의존 방향은 아래와 같습니다.

- `ingestion/*` -> `retrieval/*`
- `agents/*` -> `retrieval/*`

반대로 `retrieval/`이 `agents/`나 `ingestion/`를 import하지 않도록 유지해야 합니다.

## 주의사항

- ingestion과 retrieval은 같은 embedding 모델을 사용해야 합니다.
- collection name이나 persist dir를 바꾸면 ingestion과 agent 조회 쪽이 함께 맞춰져야 합니다.
- 현재 backend는 서버형 DB가 아니라 로컬 `Chroma` persist 디렉터리입니다.

## 확장 방법

새 store를 추가하려면 [config.py](/Users/hyobin/skala/workspace/skala-gen-ai-project/wonwon-agent/retrieval/config.py)에 `VectorStoreConfig`를 하나 더 등록하면 됩니다.
