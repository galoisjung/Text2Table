# 텍스트 ↔ 표 변환 파이프라인

임의의 장문(기사·논문·소설·일기·보고서)을 구조화된 표로 변환하는 파이프라인.
표 → 장문 round-trip을 최종 산출물이 아니라 **추출 파이프라인을 디버깅하는 진단
도구**로 쓰는 것이 핵심 아이디어다.

---

## 빠른 실행

```bash
# 1) PDF → 표 (Markdown) + round-trip 진단용 long_text 생성
python pdf_table_extractor.py --input-dir samples --output-dir output_docs
python table_to_longtext.py --input output_docs --output-dir longtext_out --check-coverage

# 2) 장문 → 표 (진짜 목표)
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

---

## 왜 이렇게 만들었는가? 

- **초기 접근(임베딩+클러스터링)이 기대만큼 안 나옴.** round-trip을 파이프라인
  디버깅용 진단 도구로 사용하여 틀을 잡고 디테일하게 수정하는 방식을 채용.
- **전체를 한 번에 튜닝하지 않고 단계별로 GREEDY하게 검증**하는 방식을 채용. (C/A-1/A-2/D/E가
  독립 실행 가능한 CLI로 분리돼 있다).
- **LLM의 Self-Control에 기대지 않는다.** budget 적용, row_key 유일성, 완성도, 환각/누락
  체크는 전부 LLM 출력을 후처리하는 결정론적 코드로 한다.
- **표의 "모양"은 도메인과 무관하게 유한하다.** 기존 웹 테이블 연구(Lautert et al.,
  Crestan & Pantel)의 분류 체계를 채택해 preset을 5개로 수렴시킴.

---