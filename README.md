# WONWON Partners Investment Evaluation Agent

Robotics/Physical AI 도메인 스타트업의 투자 가능성을 자동 평가하는 멀티에이전트 실습 프로젝트입니다.

## 개요

- Objective: 기술력, 시장성, 팀 구성, 공개 리스크를 종합 분석해 투자 리포트 생성
- Method: LangGraph 기반 멀티에이전트 워크플로 + Agentic RAG + 웹 조사
- Tools: OpenAI API, Tavily Search, Firecrawl, PostgreSQL/SQLAlchemy, Chroma

## 핵심 기능

- 자연어 질의를 기반으로 후보 회사를 검색하고 최종 조사 대상 1개를 선정
- 회사 문서/PDF, 도메인 문서, 회사 DB를 결합해 조사
- 4개 조사 에이전트 결과를 Review 에이전트가 교차 검토
- 평가 기준에 따라 최종 투자 판단 생성
- 최종 결과를 Markdown 보고서로 저장

## Tech Stack

| Category | Details |
| --- | --- |
| Framework | Python, LangGraph, LangChain |
| LLM | `gpt-4o-mini` via OpenAI API (default, configurable by `OPENAI_MODEL`) |
| Retrieval | Local Chroma vector store (`vectordb/company`, `vectordb/domain`) |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` |
| Data/Infra | PostgreSQL, SQLAlchemy, Tavily Search, Firecrawl, PDFPlumber |
| Output | Markdown report generation |

## 에이전트 구성

| Agent | 역할 |
| --- | --- |
| find_company | 사용자 질의를 구조화 검색 입력으로 변환하고 PostgreSQL에서 후보 검색 후 최종 조사 대상 1개 선택 |
| investigate_members | Tavily 기반 웹 검색으로 CEO/핵심 팀원 조사, 리더십 구성 및 역할 커버리지 정리 |
| product_market_analysis | 회사 문서 RAG + 도메인 문서 RAG + 웹 검색으로 제품/시장 분석, 참조 출처 정규화 |
| traction | 회사 문서 웹 검색(Firecrawl)을 조합해 파트너십/채용/투자/성장 신호 수집 및 구조화 |
| risk_search | 뉴스/웹 검색으로 인증, 특허, 규제, 부정 이슈 등 공개 리스크 탐지 및 구조화 |
| review | 4개 에이전트 결과를 교차 검토해 상충 주장과 추가 확인 포인트 요약 |
| eval | 평가 기준을 적용해 최종 투자 판단, 점수, 강점/위험 산출 |
| report | 모든 에이전트 완료 시 최종 Markdown 보고서 생성 (미완료 에이전트가 있으면 종료) |

## 아키텍처
### 다이어그램
![Image](https://github.com/user-attachments/assets/225bd388-5ac9-4d80-a907-82c2dd4de0f3)

### 디렉터리 구조

```text
.
├── agents/                            # 에이전트 모듈
│   ├── agent_find_company/            # 회사 검색 및 선정
│   ├── agent_investigate_members/     # CEO/핵심팀 조사
│   ├── agent_product_market_analysis/ # 제품/시장 분석
│   ├── agent_traction/                # 파트너십/채용/투자 신호 분석
│   ├── agent_risk_search/             # 규제/인증/부정 이슈 리스크 탐지
│   ├── agent_review/                  # 병렬 조사 결과 교차 검토
│   ├── agent_eval/                    # 최종 투자 판단 생성
│   ├── agent_report/                  # Markdown 보고서 생성
│   └── workflow_common.py             # 공통 state 타입 정의
├── data/                              # 회사 문서 및 도메인 PDF/메타데이터
├── ingestion/                         # 문서 수집 및 Chroma 인덱싱 스크립트
├── init-scripts/                      # PostgreSQL 초기화 SQL
├── outputs/                           # 평가 결과 보고서 저장
├── prompts/                           # 에이전트별 프롬프트 템플릿
├── retrieval/                         # 공용 embedding/vector store/retriever 계층
├── tests/                             # 그래프 및 에이전트 테스트
├── tools/                             # 검색 도구 구현
├── utils/                             # DB 조회, 프롬프트 보조, 공통 유틸
├── vectordb/                          # 로컬 Chroma persist 디렉터리
├── company_research_graph.py          # LangGraph 워크플로 정의
├── docker-compose.yml                 # PostgreSQL 컨테이너 설정
├── main.py                            # CLI 실행 엔트리포인트
└── README.md
```

## 실행 방법

### 1) 환경 변수 설정

`.env` 예시:

```env
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypass
POSTGRES_DB=mydb
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

OPENAI_API_KEY=KEY
TRAVILY_API_KEY=KEY
```

### 2) 의존성 설치 및 인프라 실행

```bash
docker compose up -d
uv sync
```

### 3) 실행

```bash
python main.py "AI 로봇 스타트업 찾아서 투자 평가해줘"
```

성공 시 보고서는 `outputs/reports/{company_id}.md`에 저장됩니다.

## 투자 평가 기준

| No | 비중 | 평가항목 | 평가 포인트 |
| --- | --- | --- | --- |
| 1 | 20% | 창업자 및 핵심팀 신뢰도 | 도메인 전문성, 제품화 경험, 팀 구성 완결성, 이전 실적 |
| 2 | 15% | 시장 문제 및 도입 논리 명확성 | 고객 명확성, 작업 정의, 도입 필요성, ROI/KPI, 구매 논리 |
| 3 | 20% | 제품 완성도 및 시스템 차별성 | 기술 성숙도, 통합도, 실환경 검증, 반복 성능, 경쟁 우위 |
| 4 | 20% | 상용화 진전도 및 시장 검증 | PoC/파일럿, 유료 고객, 레퍼런스, 반복 계약, 최근 진전 |
| 5 | 15% | AI 자율성 및 데이터 운영 우위 | AI 핵심성, 데이터 루프, 운영성, fallback, 지속 개선 구조 |
| 6 | 10% | 공개 리스크 및 안전/규제 대응 | 안전사고, 법적 리스크, 보안 이슈, 인증/규제 대응 |

## Contributors

- 최재민: 트랙션/모멘텀 에이전트, 리뷰 에이전트, 투자 종합 평가 에이전트 개발
- 이준수: 리스크 탐지 에이전트 개발, README 작성
- 장효빈: RAG 문서 수집 및 Embedding 적재, Chroma DB 구성, 제품-시장 분석 에이전트 개발
- 김경록: 루브릭 평가 기준 작성, 기업 탐색/창업팀 분석/보고서 생성 에이전트 개발