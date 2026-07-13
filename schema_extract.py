"""
schema_extract.py

[A-2 단계] 스키마 확정 후 추출

A-1(schema_scan.py)에서 채택된 extension_columns(있다면)와 preset의
core_columns를 합쳐 최종 컬럼 목록을 프롬프트에 명시적으로 고정하고,
LLM이 그 컬럼만 사용해 Markdown 표를 직접 생성하게 한다.

설계 원칙
---------
- 컬럼 목록을 프롬프트에서 명시적으로 나열해 LLM이 컬럼을 임의로
  늘리거나 빼지 못하게 한다 (header proliferation을 A-2에서 다시
  방지하는 두 번째 저지선).
- preset.validation_rule은 사람이 읽는 설명일 뿐 아니라, 코드가 실제로
  검증하는 규칙이기도 하다: preset.key_uniqueness_columns로 row_key
  유일성을 프로그램적으로 체크하고, 컬럼별 채움 비율(completeness)도
  계산해 이상 징후를 잡는다. LLM 출력을 신뢰하지 않고 후처리로 검증한다는
  원칙을 A-1(recurrence)에 이어 여기서도 유지한다.
- vertical_entity/listing처럼 extension_columns가 없는 preset은 core만
  으로 바로 추출한다 (schema_scan을 아예 스킵했던 것과 대칭).

사용 예
-------
    python schema_extract.py --longtext tables_longtext.json \\
        --classification preset_classification.json \\
        --scan schema_scan_result.json \\
        --output schema_extract_result.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

from preset_library import PRESETS, Preset

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
LOW_COMPLETENESS_THRESHOLD = 0.5  # 이 비율 미만으로 채워진 컬럼은 경고 대상


# ─────────────────────────────────────────────────────────────
# 1. 최종 컬럼 목록 확정 (core + 채택된 extension)
# ─────────────────────────────────────────────────────────────
def resolve_final_columns(preset: Preset, accepted_extension: list[dict] | None) -> list[str]:
    columns = [c.name for c in preset.core_columns]
    if accepted_extension:
        columns += [c["name"] for c in accepted_extension if c["name"] not in columns]
    return columns


# ─────────────────────────────────────────────────────────────
# 2. 프롬프트 구성
# ─────────────────────────────────────────────────────────────
def _format_few_shot(preset: Preset) -> str:
    if not preset.few_shot:
        return "(참고 예시 없음)"
    lines = []
    for ex in preset.few_shot:
        lines.append(f"- 원문: \"{ex['text']}\"")
        lines.append(f"  -> 행: {ex['row']}")
    return "\n".join(lines)


def build_extraction_prompt(
    preset_id: str,
    text: str,
    accepted_extension: list[dict] | None = None,
    context_before: str = "",
    context_after: str = "",
) -> str:
    preset = PRESETS.get(preset_id)
    if preset is None:
        raise ValueError(f"알 수 없는 preset_id: {preset_id}")

    final_columns = resolve_final_columns(preset, accepted_extension)
    columns_str = " | ".join(final_columns)
    few_shot_block = _format_few_shot(preset)

    return f"""당신은 텍스트에서 표를 정확하게 추출하는 전문가입니다.

[표의 구조]
- 행의 의미: {preset.row_unit_desc}
- 컬럼(반드시 이 순서와 이름을 그대로 사용, 추가/삭제 금지): {columns_str}

[참고 예시 — 같은 유형의 다른 문서에서 실제로 이렇게 표로 변환되었습니다]
{few_shot_block}

[규칙]
1. 위에 나열된 컬럼만 사용하세요. 컬럼을 추가하거나 빼지 마세요.
2. 텍스트에 없는 정보를 추측하거나 지어내지 마세요. 해당 항목에 값이
   없으면 빈 문자열이 아니라 "-"로 표시하세요.
3. {preset.validation_rule}
4. 숫자는 원문 표기(단위 포함)를 그대로 유지하세요.
5. 출력은 파이프(|) 문법의 Markdown 표만 작성하세요. 헤더 행, 구분선(---),
   데이터 행만 포함하고 다른 설명이나 코드펜스는 쓰지 마세요.

[표 앞 문맥]
{context_before or "(없음)"}

[본문 — 이 내용을 표로 추출하세요]
{text}

[표 뒤 문맥]
{context_after or "(없음)"}
"""


# ─────────────────────────────────────────────────────────────
# 3. Ollama 호출 (일반 텍스트 출력, 재시도) -- table_to_longtext.py와 동일 패턴
# ─────────────────────────────────────────────────────────────
def call_ollama_extract(
    prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            text = re.sub(r"^```(?:markdown)?|```$", "", text, flags=re.MULTILINE).strip()
            if not text:
                raise ValueError("Ollama가 빈 응답을 반환했습니다.")
            return text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as e:
            last_error = e
            wait = 2 ** (attempt - 1)
            print(f"    [재시도 {attempt}/{max_retries}] {e} -> {wait}초 대기 후 재시도")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 500 <= status < 600:
                last_error = e
                wait = 2 ** (attempt - 1)
                print(f"    [재시도 {attempt}/{max_retries}] HTTP {status} -> {wait}초 대기 후 재시도")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Ollama 추출 호출이 {max_retries}회 모두 실패했습니다: {last_error}")


# ─────────────────────────────────────────────────────────────
# 4. Markdown 파싱 + 검증 (LLM 출력을 신뢰하지 않고 코드로 확인)
# ─────────────────────────────────────────────────────────────
def parse_markdown_table(md: str) -> tuple[list[str], list[dict]]:
    lines = [ln for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], []

    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(header):
            continue  # 형식이 깨진 행은 건너뜀 (validation에서 row_count로 드러남)
        rows.append(dict(zip(header, cells)))

    return header, rows


def validate_extraction(
    preset_id: str,
    header: list[str],
    rows: list[dict],
    expected_columns: list[str],
) -> dict:
    preset = PRESETS[preset_id]

    missing_columns = [c for c in expected_columns if c not in header]
    extra_columns = [c for c in header if c not in expected_columns]

    duplicate_keys: list[tuple] = []
    key_cols_present = [c for c in preset.key_uniqueness_columns if c in header]
    if key_cols_present:
        seen: dict[tuple, int] = {}
        for r in rows:
            key = tuple(r.get(c, "") for c in key_cols_present)
            if key in seen:
                duplicate_keys.append(key)
            else:
                seen[key] = 1

    completeness: dict[str, float] = {}
    for col in header:
        if not rows:
            completeness[col] = 0.0
            continue
        filled = sum(1 for r in rows if r.get(col, "").strip() not in ("", "-"))
        completeness[col] = round(filled / len(rows), 2)

    low_completeness_columns = [
        col for col, ratio in completeness.items()
        if ratio < LOW_COMPLETENESS_THRESHOLD and col not in key_cols_present
    ]

    return {
        "row_count": len(rows),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "duplicate_keys": duplicate_keys,
        "column_completeness": completeness,
        "low_completeness_columns": low_completeness_columns,
    }


# ─────────────────────────────────────────────────────────────
# 5. 오케스트레이션
# ─────────────────────────────────────────────────────────────
def extract_table(
    preset_id: str,
    text: str,
    accepted_extension: list[dict] | None = None,
    context_before: str = "",
    context_after: str = "",
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    preset = PRESETS.get(preset_id)
    if preset is None:
        return {"error": f"알 수 없는 preset_id: {preset_id}"}

    expected_columns = resolve_final_columns(preset, accepted_extension)
    prompt = build_extraction_prompt(
        preset_id, text, accepted_extension, context_before, context_after
    )
    table_markdown = call_ollama_extract(
        prompt, model=model, ollama_url=ollama_url, timeout=timeout, max_retries=max_retries
    )
    header, rows = parse_markdown_table(table_markdown)
    validation = validate_extraction(preset_id, header, rows, expected_columns)

    return {
        "preset_id": preset_id,
        "expected_columns": expected_columns,
        "table_markdown": table_markdown,
        "validation": validation,
    }


# ─────────────────────────────────────────────────────────────
# 6. CLI: 이전 두 단계(C, A-1) 결과와 조인해서 실행
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="preset_classifier.py + schema_scan.py 결과를 바탕으로 A-2 추출을 실행합니다."
    )
    parser.add_argument("--longtext", required=True, help="tables_longtext.json")
    parser.add_argument("--classification", required=True, help="preset_classifier.py 출력")
    parser.add_argument("--scan", default=None, help="schema_scan.py 출력 (없으면 core_columns만 사용)")
    parser.add_argument("--output", default="schema_extract_result.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    with open(args.longtext, encoding="utf-8") as f:
        longtext_entries = {e["table_id"]: e for e in json.load(f)}
    with open(args.classification, encoding="utf-8") as f:
        classifications = json.load(f)

    scan_by_id: dict[str, dict] = {}
    if args.scan:
        with open(args.scan, encoding="utf-8") as f:
            for s in json.load(f):
                scan_by_id[s["item_id"]] = s

    results = []
    for c in classifications:
        item_id = c["item_id"]
        preset_id = c.get("preset_id")
        entry = longtext_entries.get(item_id)

        if entry is None:
            print(f"[경고] {item_id}: long_text를 찾지 못해 건너뜁니다.")
            continue
        if not preset_id:
            print(f"[건너뜀] {item_id}: 폴백 항목 (자유 스키마 경로 필요)")
            continue

        scan_result = scan_by_id.get(item_id, {})
        accepted_extension = scan_result.get("accepted_columns")  # None이면 core만 사용

        print(f"{item_id} ({preset_id}) 추출 중...")
        result = extract_table(
            preset_id,
            entry.get("long_text", ""),
            accepted_extension=accepted_extension,
            context_before=entry.get("context_before", ""),
            context_after=entry.get("context_after", ""),
            model=args.model,
            ollama_url=args.ollama_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        result["item_id"] = item_id
        results.append(result)

        v = result.get("validation", {})
        print(f"    -> {v.get('row_count', 0)}행 추출")
        if v.get("missing_columns"):
            print(f"    [경고] 누락된 컬럼: {v['missing_columns']}")
        if v.get("extra_columns"):
            print(f"    [경고] 예상 밖 컬럼: {v['extra_columns']}")
        if v.get("duplicate_keys"):
            print(f"    [경고] row_key 중복: {v['duplicate_keys'][:3]}")
        if v.get("low_completeness_columns"):
            print(f"    [경고] 채움 비율 낮은 컬럼: {v['low_completeness_columns']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()