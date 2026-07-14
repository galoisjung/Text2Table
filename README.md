# 텍스트 ↔ 표 변환 파이프라인

임의의 장문(기사·논문·소설·일기·보고서)을 구조화된 표로 변환하는 파이프라인.
표 → 장문 round-trip을 최종 산출물이 아니라 **추출 파이프라인을 디버깅하는 진단
도구**로 쓰는 것이 핵심 아이디어다.

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
| **통합** | `text_to_table.py` | 텍스트 파일 하나에 C→D→E를 전부 실행 | `.txt` 파일 | `{name}.table.md/json` |

> `schema_scan.py`/`schema_extract.py`의 CLI는 실제 실행 경로에서 안 쓴다.
> `chunk_orchestrator.py`가 두 파일의 함수를 직접 import해서 쓰기 때문 —
> 이 둘의 CLI는 A-1/A-2만 따로 떼어 디버깅할 때 쓰는 격리 도구다.

```
PDF ─▶ 표(Markdown) ─▶ long_text ┐
                                  │  (진짜 목표는 여기서부터)
                                  ▼
                    C: preset 분류
                                  ▼
                    D: 스키마 제안(A-1) → 추출(A-2) → [청크 병합]
                                  ▼
                    E: round-trip 자기검증
```

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

</details>

<details>
<summary><b>roundtrip_verify.py</b> — [E] round-trip 자기검증</summary>

- 새 LLM 호출을 만들지 않고 `table_to_longtext.py`의 verbalization을 반대 방향으로 재사용
- **환각 체크**: D가 만든 표의 값이 원문(long_text)에 실제로 있었는가
- **누락 체크(진짜 round-trip)**: 표를 다시 장문으로 복원 → 원문의 숫자/날짜/퍼센트가 복원문에 남아있는가. **원본 표 없이도 동작**하는 유일한 검증이라 "임의의 장문 → 표"에 일반화됨
- `--resume`, `--report-only`(LLM 재호출 없이 `.jsonl`만으로 리포트 재생성), `Ctrl+C` 시에도 그때까지 결과 저장
- `.md` 리포트: 원문 / 정답 표(원본) / 재구성된 표 / 복원된 장문 / 검증 수치를 항목별로 나란히 표시

</details>

<details>
<summary><b>text_to_table.py</b> — 통합: .txt 하나 → 표</summary>

- `preset_classifier`/`chunk_orchestrator`/`roundtrip_verify`의 함수를 직접 import해서 C→D→E를 한 번에 실행
- 출처 표가 없는 순수 텍스트가 입력이므로 `context_before/after`는 빈 문자열, E의 "정답과 비교"는 불가(환각/누락 체크만 유효)
- C가 폴백(`preset_id=None`)을 반환하면 자유 스키마 경로가 아직 없어 명확한 사유와 함께 종료
- 인코딩 자동 판별(UTF-8→UTF-8-BOM→CP949), HTTP 5xx 자동 재시도

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
