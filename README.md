# 텍스트 ↔ 표 변환 파이프라인

임의의 장문(기사·논문·소설·일기·보고서)을 구조화된 표로 변환하는 파이프라인.
표 → 장문 round-trip을 최종 산출물이 아니라 **추출 파이프라인을 디버깅하는 진단
도구**로 쓰는 것이 핵심 아이디어다. 이 진단 도구는 표 1개짜리 추출뿐 아니라,
한 문서에서 표를 몇 개까지 뽑을지 정하는 정지 조건(F단계)으로도 재사용된다.

---

## 실행 환경 준비 (Ollama Cloud)

이 파이프라인은 기본값으로 `gpt-oss:120b-cloud` 모델을 쓴다. 이름에 `-cloud`가 붙은
모델은 로컬 GPU에서 돌아가는 게 아니라, **로컬에 설치된 Ollama가 요청을 Ollama의
클라우드 인프라로 그대로 전달**하는 방식으로 동작한다. 즉 120B급 모델 가중치를
로컬에 받을 필요는 없지만, Ollama 계정 로그인은 필요하다.

1. **Ollama 설치**: https://ollama.com/download
2. **클라우드 로그인**: 터미널에서 `ollama signin` 실행 (브라우저로 ollama.com 로그인)
3. **클라우드 모델 등록**: `ollama pull gpt-oss:120b-cloud` — 실제 가중치가 아니라
   메타데이터만 받아오므로 몇 초 안에 끝난다.
4. **로컬 서버 확인**: `ollama serve`가 `http://localhost:11434`에서 떠 있어야 한다
   (설치 시 보통 자동 실행됨). 이 파이프라인의 모든 스크립트는 이 주소로 요청을 보낸다.
5. **정상 동작 확인**:
   ```bash
   curl http://localhost:11434/api/chat -d '{"model": "gpt-oss:120b-cloud", "messages": [{"role":"user","content":"안녕"}], "stream": false}'
   ```
   응답이 오면 준비 완료.

### 자주 겪는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `You need to be signed in to Ollama to run Cloud models` | 로그인 안 됨 | `ollama signin` |
| `502 Bad Gateway` | 클라우드 게이트웨이 일시 오류 (로컬 문제 아님) | 파이프라인이 5xx는 자동 재시도함(`--max-retries`로 횟수 조절). 계속되면 잠시 후 다시 시도 |
| `UnicodeDecodeError` (입력 `.txt` 읽을 때) | 파일이 UTF-8이 아니라 CP949(한국어 Windows "ANSI" 저장)로 되어 있음 | `text_to_table.py`가 UTF-8/CP949를 자동 순차 시도함 |

---

## 빠른 실행

가장 단순한 경로 — 임의의 텍스트 파일 하나를 표로 바꾸기:

```bash
python text_to_table.py --input essay.txt
```

(C→D만 실행한다. round-trip 검증까지 보고 싶으면 `roundtrip_verify.py`를 별도로
돌린다 — 입력 형식이 달라서 `text_to_table.py`의 출력을 바로 넣을 수는 없고,
`tables_longtext.json` 형식의 배치 입력이 필요하다. 아래 "PDF부터 시작하는 전체
경로" 참고.)

PDF부터 시작하는 전체 경로:

```bash
# 1) PDF → 표(Markdown) + round-trip 진단용 long_text 생성
python pdf_table_extractor.py --input-dir samples --output-dir output_docs
python table_to_longtext.py --input output_docs --output-dir longtext_out --check-coverage

# 2) 장문 → 표 (여러 문서를 배치로 돌릴 때)
python preset_classifier.py --input longtext_out/tables_longtext.json --output preset_classification.json
python chunk_orchestrator.py --longtext longtext_out/tables_longtext.json --classification preset_classification.json --resume
python roundtrip_verify.py --longtext longtext_out/tables_longtext.json --extracted chunk_extract_result.json --resume
```

텍스트 레이어가 있는 PDF 하나를 바로 표로 바꾸고 싶을 때(docling 없이, 가볍게):

```bash
python pdf_to_table.py --input novel.pdf                        # 표 1개
python pdf_to_table.py --input novel.pdf --multi                # 표 여러 개(F단계)
python pdf_to_table.py --input novel.pdf --multi --max-tables 0 --genre narrative  # 무제한, 서사체
```

한 장문에서 표를 여러 개(F단계) 뽑고 싶을 때:

```bash
python multi_table_extractor.py --input novel.txt --max-tables 0
```

---

## 파이프라인 한눈에 보기

| 단계 | 스크립트 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| PDF 추출 | `pdf_table_extractor.py` | PDF → 표(Markdown) | PDF 파일 | `{stem}_tables.md/json` |
| 진단용 역변환 | `table_to_longtext.py` | 표 → 장문 (round-trip 데이터 생성) | `{stem}_tables.md` | `tables_longtext.json` |
| **C** | `preset_classifier.py` | 장문 → 축 분류 → preset 선택 | `tables_longtext.json` | `preset_classification.json` |
| **A-1** | `schema_scan.py` | 확장 열 후보 스캔 *(D가 내부 호출)* | preset + long_text | 채택된 컬럼 후보 |
| **A-2** | `schema_extract.py` | 최종 표 추출 *(D가 내부 호출)* | 스키마 + long_text | `table_markdown` |
| **D** | `chunk_orchestrator.py` | A-1/A-2 실행 진입점, 긴 문서 청크 분할·병합 | `preset_classification.json` | `chunk_extract_result.json` |
| **E** | `roundtrip_verify.py` | round-trip 자기검증 (환각/누락) | `chunk_extract_result.json` | `roundtrip_verify_result.json/md` |
| **F** | `multi_table_extractor.py` | 한 장문에서 표 여러 개 반복 추출 (E의 누락 체크를 정지 신호로 재사용) | `.txt` 파일 | `{name}.multitable.md/json` |
| **통합** | `text_to_table.py` | 텍스트 파일 하나에 C→D만 실행 (E는 관심사 분리로 별도 실행) | `.txt` 파일 | `{name}.table.md/json` |
| **통합** | `pdf_to_table.py` | 텍스트 레이어가 있는 PDF → 텍스트 → `text_to_table.py`/`multi_table_extractor.py` | PDF 파일 | `{name}.table.md/json` 또는 `.multitable.md/json` |

> `schema_scan.py`/`schema_extract.py`의 CLI는 실제 실행 경로에서 안 쓴다.
> `chunk_orchestrator.py`가 두 파일의 함수를 직접 import해서 쓰기 때문 —
> 이 둘의 CLI는 A-1/A-2만 따로 떼어 디버깅할 때 쓰는 격리 도구다.

```
PDF(표 추출용) ─▶ 표(Markdown) ─▶ long_text ┐
                                              │  (진짜 목표는 여기서부터)
PDF(텍스트 레이어) ─▶ pdf_to_table.py ─▶ text ┤
                                              │
                     순수 .txt ───────────────┘
                                              ▼
                                C: preset 분류
                                              ▼
                                D: 스키마 제안(A-1) → 추출(A-2) → [청크 병합]
                                    │                        ▼
                                    │           E: round-trip 자기검증 (선택 · 별도 실행)
                                    ▼
                     F: 표 여러 개 반복 추출 (내부에서 C→D→E 반복, coverage로 정지)
```

> `text_to_table.py`는 C→D까지만 자동 실행한다. 검증(E)이 필요하면
> `roundtrip_verify.py`를 별도로 돌린다 — 표 변환과 검증을 분리해서, 검증이
> 필요 없는 경우 LLM 호출을 아낄 수 있게 했다.
> `multi_table_extractor.py`(F)는 회차마다 C→D를 새로 돌리고, E의 누락 체크
> 로직(재-verbalize 후 coverage 측정)을 표를 더 뽑을지 멈출지 정하는 정지
> 신호로 재사용한다. `pdf_to_table.py`는 텍스트 레이어가 있는 PDF를 pypdf로
> 가볍게 텍스트만 뽑아 `text_to_table.py`(표 1개) 또는
> `multi_table_extractor.py`(표 여러 개, `--multi`)로 넘기는 얇은 통합
> 스크립트다 — docling 기반 표 추출(`pdf_table_extractor.py`)과는 별개
> 경로이며, 스캔본(이미지 PDF)은 지원하지 않는다.

---

## Preset 5종

행/열의 "모양"은 초점 단위 × 시간구조 두 축으로 거의 결정되고, 여기에 속성 개수로
한 번 더 갈라져 5개로 수렴한다. 도메인(재무/법률/인물…)은 preset 개수에 영향을
주지 않는다 — extension column의 "내용"에만 영향을 준다.

| preset_id | 행의 의미 | 열 | 기존 연구 대응 용어 |
|---|---|---|---|
| `vertical_entity` | 속성 1개 | 고정 2열 (속성명, 값) | Vertical Table (Lautert et al.) |
| `event_timeline` | 사건/시점 1개 | 가변 | — |
| `horizontal_relational_static` | 개체 1개 | 가변 | Horizontal Relational Table |
| `horizontal_relational_timeseries` | 개체×시점 1건 | 가변 | Horizontal Relational + 시간축 |
| `listing` | 개체 1개, 속성 1개뿐 | 고정 2열 (항목, 값) | Vertical Listing (Crestan & Pantel) |

`extension_budget_default=None`인 `vertical_entity`/`listing`은 확장 열이 구조적으로
없어 A-1 스캔 자체를 스킵한다.

---

## 파일별 핵심 동작

<details>
<summary><b>pdf_table_extractor.py</b> — PDF → 표(Markdown)</summary>

- docling 기반 (다른 프로젝트의 `pdf2json.py`를 이식)
- 표를 tidy JSON이 아니라 **Markdown**으로 직렬화
- 이미지 추출/VLM 이미지-표 해석 제거 (표 추출 실패 시 폴백 이미지 크롭만 유지)
- `.md` 출력에 `<!-- table_id: ... -->` 숨김 주석 → 재파싱 가능한 단일 소스

</details>

<details>
<summary><b>table_to_longtext.py</b> — 표 → 장문</summary>

- `{stem}_tables.md`를 읽어 표를 로컬 Ollama LLM으로 한국어 장문 서술
- `--check-coverage`: 표의 모든 값이 장문에 최소 1회 언급됐는지 검증
- 원본 `table_markdown`도 결과에 저장 → E 단계에서 "정답"으로 재사용됨
- `--genre {informative,narrative}`: 서술 문체 전환. `informative`(기본)는
  보고서/논문체 설명 문단, `narrative`는 시간·인과관계가 드러나는 이야기체
  문장으로 풀어 쓴다. 같은 옵션이 E(`roundtrip_verify.py`)와
  F(`multi_table_extractor.py`)의 재-verbalize 단계에도 그대로 전달된다

</details>

<details>
<summary><b>preset_library.py / preset_classifier.py</b> — [C] 분류 + preset 선택</summary>

- `preset_library.py`: preset 5개 정의 + `select_preset(focus, time_structure, attribute_count)` 결정 트리 (LLM 의존성 없는 순수 로직)
- `preset_classifier.py`: LLM으로 3개 축을 분류 → `select_preset()`으로 preset 확정
- 신뢰도가 낮으면 `preset_id=None` + `fallback_reason` 반환 (자유 스키마 경로 신호, 현재 미구현)

</details>

<details>
<summary><b>schema_scan.py</b> — [A-1] 확장 열 후보 스캔</summary>

- preset의 `core_columns`/`few_shot`/`row_unit_desc`를 프롬프트에 그대로 삽입
- LLM은 각 후보의 recurrence(반복 횟수)만 정직하게 보고
- budget 적용은 코드가 결정론적으로 수행: recurrence<2 제외, budget의 2배 초과 시 `header_proliferation_risk` 플래그

</details>

<details>
<summary><b>schema_extract.py</b> — [A-2] 스키마 확정 후 추출</summary>

- core + A-1 채택 컬럼을 프롬프트에 순서까지 고정해서 명시
- `validate_extraction()`: 누락/추가 컬럼, `key_uniqueness_columns` 기준 row_key 중복, 컬럼별 채움 비율(completeness)을 코드로 검증

</details>

<details>
<summary><b>chunk_orchestrator.py</b> — [D] 청크 처리 (A-1/A-2 실행 진입점)</summary>

- 컨텍스트 한도(gpt-oss:120b-cloud 기준 128K 토큰, `--model-context-tokens`로 변경 가능) 초과 여부 판단
  - 안 넘으면: 청크 없이 `scan_schema()` + `extract_table()` 그대로
  - 넘으면: 문단/문장 경계로 분할(overlap 포함) → **스키마는 대표 청크로 1회만** → 청크마다 동일 스키마로 추출 → 느슨한 정규화(공백/기호 제거, 임베딩 없음)로 병합 → 병합 표 재검증
- `--resume` + `.jsonl` 체크포인트: 항목 처리 즉시 저장, 중단돼도 이어서 처리
- `--schema-scan-sample-chunks 0`: 앞부분 청크 샘플만 보는 대신, 모든 청크를 개별
  스캔한 뒤 느슨한 정규화로 후보를 합쳐 recurrence를 합산 (문서 전체 대상 스캔,
  청크마다 컨텍스트 예산 안에서 호출되므로 안전)
- `--quality-chunk-tokens`: 컨텍스트 오버플로 여부와 무관하게 청크 크기를 강제
  지정 가능. "컨텍스트에 다 들어간다"와 "그 안에서 다 정확히 처리한다"는 다른
  문제라는 점(lost-in-the-middle, context rot)에 따라, 오버플로 방지 상한과
  품질 목표 청크 크기를 분리했다

</details>

<details>
<summary><b>roundtrip_verify.py</b> — [E] round-trip 자기검증</summary>

- 새 LLM 호출을 만들지 않고 `table_to_longtext.py`의 verbalization을 반대 방향으로 재사용
- **환각 체크**: D가 만든 표의 값이 원문(long_text)에 실제로 있었는가
- **누락 체크(진짜 round-trip)**: 표를 다시 장문으로 복원 → 원문의 숫자/날짜/퍼센트가 복원문에 남아있는가. **원본 표 없이도 동작**하는 유일한 검증이라 "임의의 장문 → 표"에 일반화됨
- `--resume`, `--report-only`(LLM 재호출 없이 `.jsonl`만으로 리포트 재생성), `Ctrl+C` 시에도 그때까지 결과 저장
- `.md` 리포트: 원문 / 정답 표(원본) / 재구성된 표 / 복원된 장문 / 검증 수치를 항목별로 나란히 표시
- `--genre {informative,narrative}`: 누락 체크용 재-verbalize 문체 (기본 `informative`)
- `--no-ner`: salient token 추출 시 숫자/날짜/퍼센트 외에 개체명(NER)까지 포함할지 여부 (기본 포함)

</details>

<details>
<summary><b>multi_table_extractor.py</b> — [F] 한 장문에서 표 여러 개 반복 추출</summary>

- 지금까지의 파이프라인(C→D)은 "텍스트 1개 → 표 1개"로 고정돼 있었다. 인물/사건/시간축이
  여러 겹인 문서는 표 하나로 다 담으면 정보가 눌려서 결과가 얕아진다는 문제의식에서 출발
- 표를 하나 뽑을 때마다 **E(round-trip)의 누락 체크를 정지 신호로 재활용**한다 — 표를 다시
  장문으로 복원해, 원문의 salient token(숫자/날짜/개체명)이 지금까지 뽑은 표들로 얼마나
  커버됐는지 측정하고, 목표 coverage(`--coverage-stop-threshold`, 기본 0.9)에 도달하면 멈춘다.
  새 검증 장치를 만들지 않고 이미 있는 E 인프라를 반복 실행의 정지 조건으로 쓰는 것
- "몇 개의 표가 필요한지"를 LLM 판단에 맡기지 않고 측정된 coverage 수치로 결정 — "LLM
  자기절제를 믿지 않는다"는 설계 원칙을 그대로 잇는다
- 회차마다 다른 preset을 유도하기 위해, 직전까지 뽑은 표들의 preset/컬럼(그리고
  `vertical_entity`/`listing`/`event_timeline`처럼 컬럼명이 고정된 preset은 행 라벨 예시까지)을
  프롬프트에 `avoid_note`로 넣어 같은 관점의 표 재추출을 회피시킴
- 분류용 샘플을 문서 전체에서 구간별로 순환시켜(`--classification-sample-chars`), 매 회차
  똑같은 앞부분만 보고 뒷부분 정보를 놓치는 문제를 막음
- coverage 정체(새로 커버된 토큰 0개)만으로 멈추면 숫자가 적은 서사 텍스트에서 오판할 수 있어,
  이 표의 행 라벨이 지금까지 하나도 안 나온 새로운 것인지(`row_key_novelty_ratio`)를 보조
  신호로 같이 봄
- `--max-tables`(기본 10, 0=무제한)는 비용 상한이고 coverage 기반 정지와는 별개 레이어 —
  0을 줘도 내부 안전판(`DEFAULT_HARD_SAFETY_LIMIT`, 50개)은 항상 걸림
- `--genre {informative,narrative}`, 그리고 D로 그대로 전달되는 청크 관련 옵션
  (`--model-context-tokens`, `--quality-chunk-tokens` 등)을 모두 지원

</details>

<details>
<summary><b>text_to_table.py</b> — 통합: .txt 하나 → 표 (C→D)</summary>

- `preset_classifier`/`chunk_orchestrator`의 함수를 직접 import해서 C→D만 실행
  (E는 관심사 분리를 위해 빠짐 — 검증이 필요하면 `roundtrip_verify.py`를 별도로 돌린다)
- 출처 표가 없는 순수 텍스트가 입력이므로 `context_before/after`는 빈 문자열
- C가 폴백(`preset_id=None`)을 반환하면 자유 스키마 경로가 아직 없어 명확한 사유와 함께 종료
- 인코딩 자동 판별(UTF-8→UTF-8-BOM→CP949), HTTP 5xx 자동 재시도
- `.md` 리포트의 원문은 줄마다 blockquote 처리해, 여러 문단짜리 텍스트도 중간에
  끊긴 것처럼 안 보이고 전체가 다 보이게 함

</details>

<details>
<summary><b>pdf_to_table.py</b> — 통합: 텍스트 레이어가 있는 PDF → 표</summary>

- 새로 만든 부분은 "PDF → 텍스트"뿐이다. `pypdf`로 페이지 순서대로 텍스트 레이어만 가볍게
  긁어 이어붙이고, 이후 오케스트레이션은 `text_to_table.py`의 `convert_text_to_table()`
  (표 1개, 기본) 또는 `multi_table_extractor.py`의 `extract_multiple_tables()`(`--multi`,
  표 여러 개/F단계)를 그대로 가져다 쓴다 — 같은 로직을 두 번 짜지 않음
- **`pdf_table_extractor.py`(docling 기반)와는 다른 경로**다. 그쪽은 PDF 안의 표 구조 자체를
  뽑아내는 게 목적(→ round-trip 진단용 long_text 생성)이고, 이쪽은 "이미 표가 아니라 장문으로
  서술된 PDF"를 텍스트로 펼쳐서 C→D(→F)에 바로 태우는 게 목적
- 스캔본(이미지 PDF, 텍스트 레이어 없음)은 지원 범위 밖 — 페이지당 평균 추출 문자 수가
  임계치(20자) 미만이면 스캔본으로 의심해 경고와 함께 중단한다 (`--force`로 강행 가능)
- 추출된 텍스트를 `{name}.extracted.txt`로도 저장해, 필요하면 `text_to_table.py`/
  `multi_table_extractor.py`를 이 파일에 직접 돌려 더 세밀한 옵션을 쓸 수 있게 함
- `--multi`, `--max-tables`, `--coverage-stop-threshold`, `--genre` 등 F단계 옵션과
  D단계(청크) 옵션을 모두 그대로 전달받아 넘김

</details>

---

## 설계 원칙 (한눈에)

- **초기 접근(임베딩+클러스터링)이 기대만큼 안 나와서**, round-trip을 파이프라인
  디버깅용 진단 도구로 전환했다.
- **전체를 한 번에 튜닝하지 않고 단계별로 그리디하게 검증**한다.
- **LLM의 자기절제에 기대지 않는다.** budget 적용, row_key 유일성, 완성도, 환각/누락
  체크는 전부 LLM 출력을 후처리하는 결정론적 코드로 한다.
- **표의 "모양"은 도메인과 무관하게 유한하다.** 기존 웹 테이블 연구를 채택해 preset을
  5개로 수렴시켰다.

---

## 설계 과정: 시행착오와 핵심 통찰

시간 순서대로 정리한 설계 로그. "왜 지금 이 구조인지"가 궁금할 때 참고용.

### 1. 출발점 — 왜 round-trip인가
최초 접근은 텍스트에서 정보를 추출한 뒤 **벡터 임베딩(BGE-M3) + HDBSCAN 클러스터링
+ LLM 그룹핑**으로 표 행을 묶어내는 방식이었다. 결과가 기대만큼 나오지 않았고,
"어디가 문제인지" 자체를 특정하기 어려웠다. 그래서 접근을 바꿔, **표 → 장문 → 표**로
한 바퀴 돌렸을 때 재추출된 표가 원래 표와 다르면 그 차이는 원문의 모호함이 아니라
**파이프라인 자체의 불안정성**이라는 가설을 세우고, round-trip을 최종 산출물이
아니라 진단 도구로 쓰기 시작했다. 이 인프라(`table_to_longtext.py`)가 나중에
E 단계에서 그대로 재활용된다는 게 이번 설계 전체를 관통하는 실이다.

### 2. 표현 포맷 전환 — JSON에서 Markdown으로
표를 tidy record(행/열을 분해한 JSON)로 표현하던 것을 전면 **Markdown**으로
바꿨다. "LLM이 Markdown 표를 더 잘 이해한다"는 문제의식에서 시작된 결정이었고,
PDF 표 추출기(`pdf_table_extractor.py`, 다른 프로젝트의 `pdf2json.py`를 이식)의
직렬화 방식도 함께 바꿨다. 이미지 안에 있는 표를 VLM으로 인식하는 부분도
"HTML로 뽑은 뒤 파싱"이 아니라 **VLM이 Markdown을 직접 출력**하도록 단순화했다.
이 과정에서 실제 버그를 하나 잡았다 — 단위(예: "백만원")를 정규화 *이전* 컬럼명으로
저장해두고 정규화 *이후* 이름으로 조회하고 있어서, 항상 매칭에 실패해 단위가
헤더에 안 붙던 문제였다. 테스트를 짜다가 발견했다.

나중에 "LLM은 JSON이 아니라 이미 Markdown만 보고 있다"는 점을 재확인하는 계기가
있었다 — `table_to_longtext.py`가 `.json`을 읽어도 프롬프트에는 `table_markdown`
필드(순수 텍스트)만 들어가고 있었다. 그래도 `.md` 파일 하나를 사람이 검수하는
산출물이자 프로그램 입력으로 동시에 쓰고 싶다는 요구에 따라, `.md`에
`<!-- table_id: ... -->` 같은 숨김 주석을 심어 재파싱 가능한 단일 소스로
재설계했다.

### 3. 진짜 목표를 다시 정의 — "임의의 장문 → 표"
표에서 장문을 만드는 round-trip은 "원래 표가 존재한다"는 전제가 있었지만, 최종
목표(소설·일기·기사 같은 순수 장문을 표로 바꾸는 것)에는 그 전제가 없다는 걸
명확히 했다. 여기서 핵심 난제로 떠오른 것이 **"행이 무엇을 의미하는가가 텍스트마다
정해져 있지 않다"**는 문제였다(인물 중심? 사건 중심? 시간 중심?). 이 문제를
어떻게 다룰지에 대해 6개 후보 접근(스키마 선결정 2단계, bottom-up
개체-속성-값 추출, 장르별 프리셋, 청크 map-reduce, 자기검증 루프, 다중 후보
샘플링)을 비교했고, **C(장르/축 분류+프리셋) → A(프리셋을 초안으로 스키마
제안) → D(청크 처리) → E(round-trip 자기검증)** 조합으로 결정했다.

### 4. Preset이 뭘 담아야 하는가 — 오해와 정정
설계 초반에 "preset에 실제 행/열 이름이 들어가는 것 아니냐"는 오해가 있었다.
정정하자면 preset은 **표의 "모양(shape)"에 대한 구조적 규칙**일 뿐이고, 실제
헤더/값은 항상 그 문서를 봐야 정해진다 — 도메인은 preset 개수에 영향을 주지
않고, extension column의 "내용"에만 영향을 준다.

그다음 "그럼 preset 조합이 사실상 무한한 것 아니냐"는 질문이 나왔는데, 분석해보니
축들이 서로 곱해지는 게 아니라 **하나가 다른 하나를 결정하는 lookup 관계**였다
(`row_unit = f(초점단위, 시간구조)`). 곱셈이 아니라 덧셈 구조로 보면 5개로
수렴한다는 걸 확인했다.

이 지점에서 "이미 연구된 게 있지 않을까"라는 질문에 웹 테이블 추출 연구
(Lautert et al., Crestan & Pantel, Lehmberg et al.)를 검색해봤고, 우리가 독자적으로
도출한 축이 기존 분류체계(Horizontal/Vertical/Matrix Relational Table, Listing)와
거의 1:1로 대응한다는 걸 확인했다. 그래서 이름을 새로 짓지 않고 기존 용어를
채택했다.

실제 데이터(밸류에이션 리포트)에 적용해보는 과정에서, "단일주제+비시계열엔
자연스러운 행 단위가 없다"던 초기 판단이 **틀렸다**는 걸 발견했다 — "구분/내용"
형태로 속성을 세로로 나열하는 표(예: 평가계약 정보)가 정확히 이 칸에 해당했고,
이게 `vertical_entity` preset이 추가된 계기다. 처음 세운 가설이 실제 데이터
앞에서 뒤집힌 사례.

### 5. A-1/A-2 — LLM의 self-control을 믿지 않는다
A-1(`schema_scan.py`) 설계 때, LLM에게 "N개만 골라라"처럼 개수를 직접 강제하지
않기로 했다. 대신 각 후보의 recurrence(반복 횟수)만 정직하게 보고하게 하고,
budget 적용(recurrence<2 제외, budget의 2배 초과 시 경고 플래그)은 코드가
결정론적으로 수행한다. 테스트 중 mock이 실제로는 3개 개체 모두에 있는 속성의
recurrence를 1로 낮게 보고하는 사례가 나왔는데, 이게 바로 이 체크포인트가 잡아야
할 "정보 손실 조짐"이라는 걸 실증하는 계기가 됐다 — 설계 의도가 테스트에서
그대로 재현된 경우다.

A-2(`schema_extract.py`)에서도 같은 원칙을 이어서, `validation_rule`을 사람이
읽는 설명 문자열로만 두지 않고 `key_uniqueness_columns`라는 실제 코드 체크로
구현했다.

### 6. D의 정체성 — "5번째 단계"가 아니라 "실행 진입점"
"C → A → D → E"를 순차적인 4~5단계처럼 설명했다가 실제로 혼란이 생긴 지점이다.
정정하면, **D(`chunk_orchestrator.py`)는 A-1/A-2를 감싸는 실행 진입점**이고
실제 실행 그래프는 "C(무슨 preset인지) → D(그 preset으로 필요하면 청크까지
알아서 처리)" 2단계다. `schema_scan.py`/`schema_extract.py`의 독립 CLI는
프로덕션 경로가 아니라 A-1/A-2만 따로 디버깅할 때 쓰는 격리 도구로 재정리했다.

청크 분할 시 **"스키마는 문서당 1회만 결정하고, 추출은 청크마다 반복한다"**는
원칙을 타협하지 않았다 — 안 그러면 청크마다 컬럼이 갈라져서, 병합 단계가
예전에 실패했던 임베딩 클러스터링 문제를 그대로 다시 짊어지게 되기 때문이다.
병합 시 정규화 방식은 임베딩 유사도 대신 **느슨한 정규화(공백/기호 제거)**만
쓰기로 결정했다 — 실제로 정확 일치로 안 잡히는 사례가 쌓이면 그때 임베딩을
검토하기로 하고, 처음부터 무겁게 가지 않았다.

테스트 중 실제 버그를 하나 더 잡았다: 정규화로 이미 매칭된 키 컬럼 자체가
표기 차이(예: "주식회사 에이피에스" vs "주식회사에이피에스") 때문에 값 충돌로
오탐되던 문제였다. 키로 이미 매칭된 컬럼은 충돌 검사에서 제외하도록 고쳤다.

### 7. E — 새 인프라를 안 만들고 기존 걸 반대로 쓴다
E를 설계할 때, 1번 항목(왜 round-trip을 시작했는지)으로 다시 돌아가
`table_to_longtext.py`의 verbalization을 그대로 재사용했다. **환각 체크**(표의
값이 원문에 실제로 있었는가)와 **누락 체크**(표를 다시 장문으로 복원했을 때
원문의 사실이 남아있는가) 두 가지로 분리했는데, 후자는 **원본 표(ground truth)
없이도 동작**하는 유일한 방법이라 "임의의 장문 → 표"라는 최종 목표에 실제로
일반화되는 검증이다.

실사용 중 사용자가 "표 결과가 안 보인다"고 지적해서 확인해보니,
`verify_extraction()`이 검증만 하고 정작 검증 대상 `table_markdown`을 결과에
담아 반환하지 않고 있었다 — 실제 누락 버그였고, 고치면서 `.md` 리포트도
새로 추가했다. 이후 원본 표(정답)를 리포트에 나란히 넣었더니, 값은 정확히
맞았는데 헤더 이름 자체(`Unnamed | 금액` vs `항목 | 값`)는 원본과 다르게 나온
실제 사례가 눈에 보였다 — "라운드트립은 헤더 문자열이 아니라 구조와 값으로
평가해야 한다"는 원칙을 실측 데이터로 재확인한 순간이다.

### 8. 실행부 분리, 그리고 "컨텍스트에 들어간다"와 "품질"은 다른 문제
`text_to_table.py`는 처음엔 C→D→E를 한 번에 실행했다. 그런데 표만 뽑으면 되고
검증까지는 필요 없는 경우엔 E의 LLM 호출이 그냥 낭비였다 — 그래서 E를 분리해
`text_to_table.py`는 C→D만 하고, 검증이 필요하면 `roundtrip_verify.py`를 별도로
돌리는 구조로 바꿨다. 표 변환과 검증을 서로 다른 관심사로 나눈 것.

또 하나, "gpt-oss:120b-cloud가 컨텍스트가 아무리 커도 나눠 넣는 게 낫지 않냐"는
지적이 있었다. 실제로 찾아보니 이건 근거가 있는 우려였다 — **lost-in-the-middle**
(정보가 컨텍스트 중간에 있으면 정확도가 U자형으로 떨어지는 현상)과 **context rot**
(위치와 무관하게 입력이 길어지기만 해도 정확도가 떨어지는 현상)은 서로 다른, 둘 다
입증된 열화 현상이고, 특히 "빠짐없이 다 세고 다 뽑아야 하는" 우리의 A-1
recurrence 집계나 A-2 전체 추출 같은 작업이 여기 가장 취약한 유형이다. "컨텍스트
한도 안에 들어간다"와 "그 안에서 다 정확히 처리한다"는 별개 문제라는 뜻이다.

그래서 "오버플로를 막기 위한 상한"(`model_context_tokens`/`reserved_tokens`)과
"품질을 위한 목표 청크 크기"(`quality_chunk_tokens`, 신규)를 분리했다. 후자를
지정하면 컨텍스트에 여유가 있어도 그보다 작게 강제로 나눠서 처리한다. 다만 아주
짧은 텍스트까지 무조건 나누게 만들지는 않았다 — 필요 없는 병합 오버헤드만 늘 뿐,
새로운 이점이 없기 때문이다. 이 값을 지정 안 하면 예전과 동일하게 동작하도록
기본값은 `None`으로 뒀다.

### 9. F — 표 1개의 한계, 그리고 새 인프라 대신 E를 반대로 다시 쓴다
C→D는 처음부터 "텍스트 1개 → 표 1개"를 전제로 설계됐다. 그런데 인물도 여러 명,
사건도 여러 개, 시간 흐름도 있는 소설·보고서를 표 하나에 눌러 담으면 결과가
얕아 보인다는 문제가 실제 데이터에서 나왔다. "표를 몇 개 뽑아야 충분한가"를
LLM의 자기 판단에 맡기지 않는다는 원칙(5번 항목)을 여기서도 지키려면, 정지
조건을 측정 가능한 수치로 만들어야 했다.

7번 항목에서 만든 E의 누락 체크(표를 다시 장문으로 복원해 원문 salient
token이 얼마나 남아있는지 재는 것)가 정확히 이 역할을 할 수 있었다 — 표를 하나
뽑을 때마다 그 표를 복원해서 "지금까지 뽑은 표들이 원문을 얼마나 커버했는가"를
누적으로 재고, 목표 coverage에 도달하거나 더 이상 새 정보가 안 늘어나면 멈춘다.
새 검증 인프라를 만들지 않고 기존 걸 반복 실행의 정지 신호로 재활용한 것(8번
항목과 같은 패턴).

실제로 돌려보니 두 가지 실패 모드가 나왔다. 하나는 `vertical_entity`/`listing`/
`event_timeline`처럼 컬럼명이 문서 내용과 무관하게 항상 고정된 preset에서,
"같은 구조(컬럼명 동일)면 중복"으로 판정하는 로직이 실제 내용이 완전히 달라도
1개에서 멈춰버리는 문제였다(해리포터 텍스트로 재현) — 컬럼명 대신 행 라벨(내용)의
겹치는 비율로 중복을 재도록 고쳤다. 다른 하나는 coverage만으로 정체를 판단하면
숫자·날짜가 적은 서사 텍스트에서 오판할 수 있다는 점이라, 이 표가 실제로 새로운
행 라벨(개체/속성)을 다뤘는지를 보조 신호(`row_key_novelty_ratio`)로 같이 보게
했다. 회차마다 분류용 샘플을 문서의 다른 구간으로 순환시킨 것도 같은 맥락 —
안 그러면 `preset_classifier`의 샘플 길이 제한 때문에 매 회차 똑같은 앞부분만
보고, 뒷부분에 있는 서로 다른 정보를 영영 못 보게 된다.

### 10. PDF 입력 경로 분리 — "표 추출용"과 "텍스트 추출용"은 다른 문제다
`pdf_table_extractor.py`(docling)는 PDF 안에 이미 있는 표의 구조 자체를
뽑아내는 게 목적이라 무겁다(레이아웃 분석, OCR 폴백 등). 그런데 "표가 아니라
장문으로 서술된 PDF를 표로 바꾸고 싶다"는 요구에는 이 무게가 필요 없다 — 텍스트
레이어만 그대로 펼치면 되는 문제라서, `pypdf`로 가볍게 텍스트만 뽑는
`pdf_to_table.py`를 별도로 뒀다. 오케스트레이션(C→D, 필요하면 F까지)은
`text_to_table.py`/`multi_table_extractor.py`의 함수를 그대로 재사용해서
로직이 두 군데로 갈라지지 않게 했다. 스캔본(이미지 PDF)은 텍스트 레이어가
없어 이 가벼운 경로로 처리할 수 없으므로, 페이지당 평균 추출 문자 수가
너무 낮으면 스캔본으로 의심하고 명확히 중단한다 — 조용히 빈 표를 내보내는
대신, 처리 범위 밖임을 알리는 쪽을 택했다.

### 핵심 통찰 요약

- **Round-trip은 산출물이 아니라 디버깅 신호다.** 원본 표 없이도 "복원문에서
  원문의 사실이 사라졌는가"로 추출 품질을 잴 수 있다는 게 이 프로젝트 전체를
  관통하는 아이디어다.
- **Preset은 내용이 아니라 구조(shape)만 고정한다.** 도메인은 preset 개수에
  영향을 주지 않는다.
- **축은 곱하지 않는다.** 하나가 다른 하나를 결정하면 lookup으로 흡수하고,
  진짜 독립적인 것만 별도 축으로 둔다.
- **LLM에게 Self-control을 요구하지 않는다.** budget, 유일성, 완성도, 환각/누락은
  전부 후처리 코드로 검증한다.
- **새 기능이 필요하면 새 인프라부터 만들지 않는다.** 기존 걸 반대 방향으로
  못 쓰는지 먼저 본다 (E가 table_to_longtext.py를 재사용한 것처럼).
- **가설은 실제 데이터 앞에서 뒤집힐 수 있다.** `vertical_entity` preset은
  "이 칸엔 자연스러운 행 단위가 없다"던 초기 판단이 실제 데이터로 반박되면서
  추가됐다.
- **장시간 파이프라인은 중간 저장이 기본값이어야 한다.**
- **"컨텍스트에 들어간다"와 "그 안에서 다 정확히 처리한다"는 다른 문제다.**
  한도를 꽉 채우기보다, 처리 품질을 기준으로 별도의 청크 크기를 정하는 게 낫다.
- **"몇 개가 충분한가"도 LLM 판단이 아니라 측정치로 결정한다.** F단계에서
  "표를 몇 개 더 뽑을지"를 E의 누락 체크(coverage)로 재활용해 정지 조건을
  삼은 것도 결국 "LLM 자기절제를 믿지 않는다"는 같은 원칙의 확장이다.
- **"컨텍스트에 다 들어가는 문제"와 "표 구조 자체를 뽑는 문제"는 다른
  무게를 요구한다.** PDF에서 표 자체를 뽑아야 하면 docling(무거움)이,
  이미 서술된 텍스트를 표로 바꾸기만 하면 되면 텍스트 레이어 추출(가벼움)이
  맞는 도구다 — 목적에 맞지 않는 무거운 경로를 기본값으로 두지 않았다.

---

## 알려진 한계

- **자유 스키마(schema-free) 경로 없음**: C(`preset_classifier.py`)가 신뢰도
  부족으로 `preset_id=None`(폴백)을 반환하면, `text_to_table.py`/
  `pdf_to_table.py`/`multi_table_extractor.py` 모두 명확한 사유와 함께
  중단한다. preset 5종 중 어디에도 안 맞는 문서를 위한 "프리셋 없이 A만
  단독으로 스키마를 도출하는" 경로는 아직 구현돼 있지 않다.
- **스캔본(이미지) PDF 미지원**: `pdf_to_table.py`는 텍스트 레이어가 있는
  PDF만 다룬다. 이미지 기반 PDF는 `--force`로 강행해도 텍스트가 거의 안
  나와 의미 있는 결과를 못 만든다 — 표 구조까지 필요하면 docling 기반
  `pdf_table_extractor.py` 쪽을 검토해야 한다.
- **청크 병합은 느슨한 정규화(공백/기호 제거)만 쓴다**: 임베딩 유사도 매칭은
  의도적으로 아직 도입하지 않았다. 정확 일치로 안 잡히는 표기 차이 사례가
  실제로 쌓이면 그때 검토하기로 한 결정이다.
- **F단계 정지 조건은 휴리스틱이다**: coverage 임계치(`--coverage-stop-threshold`)와
  행 라벨 신선도(`row_key_novelty_ratio`) 두 신호로 정지 시점을 정하지만,
  둘 다 완벽한 판정 기준은 아니다. `--max-tables`/내부 안전판(50개)이 최종
  방어선 역할을 한다.

---

## 설치

```bash
pip install -r requirement.txt
```

- `requirement.txt`는 UTF-16(LE) 인코딩으로 저장돼 있다 — `pip`은 이를 문제
  없이 읽지만, 직접 열어서 편집할 때는 에디터가 UTF-16으로 인식하는지 확인할 것.
- PDF에서 표 구조 자체를 뽑는 경로(`pdf_table_extractor.py`)는 `docling`/
  `paddleocr`/`paddlepaddle` 등 무거운 의존성이 필요하다. 텍스트 레이어만
  가볍게 뽑는 `pdf_to_table.py`는 `pypdf`만 있으면 된다.
  주의: `pdf_to_table.py`는 `pypdf`를 import하지만 `requirement.txt`에는
  `pypdfium2`(다른 패키지)만 있고 `pypdf`가 빠져 있다 — `pdf_to_table.py`를
  쓰려면 `pip install pypdf`를 별도로 실행해야 한다.
- LLM 호출은 로컬에 설치된 Ollama가 담당한다 — 코드 의존성 설치와는 별개로
  위의 "실행 환경 준비(Ollama Cloud)" 절차를 반드시 따라야 한다.