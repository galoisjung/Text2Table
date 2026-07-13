"""
chunk_orchestrator.py

[D 단계] 청크 처리

원칙: 스키마(A-1)는 문서당 한 번만 결정하고, 추출(A-2)만 청크마다 반복한다.
청크마다 스키마를 새로 제안하면 컬럼이 갈라져서 병합 단계가 예전에 실패했던
임베딩 클러스터링 문제를 통째로 다시 짊어지게 되므로, 이 원칙은 타협하지 않는다.

흐름
----
1. 문서가 모델 컨텍스트 한도를 넘는지 판단 (넘지 않으면 청크 없이 A-1/A-2 그대로)
2. 넘으면 문단/문장 경계로 청크 분할 (약간의 overlap 포함)
3. 대표 청크(앞부분 N개)만으로 A-1 스키마 스캔 1회 -> 스키마 확정
4. 모든 청크에 동일 스키마로 A-2 추출 반복
5. 청크별 표를 행 단위로 합치고, row_key를 느슨한 정규화(공백/기호 제거)로
   비교해 같은 개체로 보이는 행을 병합 (임베딩 유사도는 쓰지 않음 -- 실제로
   정확 일치로 안 잡히는 사례가 쌓이면 그때 도입 검토)
6. 병합된 전체 표에 schema_extract.validate_extraction()을 다시 실행
   (청크 단위 검증으로는 청크 간에 걸친 row_key 중복을 못 잡기 때문)

토큰 수는 tiktoken 등 정확한 토크나이저 없이 문자 수 기반 근사치로 추정한다
(--chars-per-token으로 보정 가능). gpt-oss:120b-cloud의 실측 컨텍스트는
128K(131072) 토큰이며, 프롬프트 템플릿/few_shot/출력 여유분을 위해
--reserved-tokens 만큼 미리 빼고 계산한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from preset_library import PRESETS
from schema_extract import extract_table, parse_markdown_table, resolve_final_columns, validate_extraction
from schema_scan import scan_schema

# gpt-oss:120b-cloud 실측 context_length (Ollama 모델 메타데이터 기준)
DEFAULT_MODEL_CONTEXT_TOKENS = 131072
DEFAULT_RESERVED_TOKENS = 6000  # 프롬프트 템플릿 + few_shot + 지시문 + 출력 여유분
DEFAULT_CHARS_PER_TOKEN = 2.0   # 한국어 텍스트 근사치 (정확한 토크나이저 없을 때)
DEFAULT_CHUNK_OVERLAP_RATIO = 0.1
DEFAULT_SCHEMA_SCAN_SAMPLE_CHUNKS = 2


# ─────────────────────────────────────────────────────────────
# 1. 토큰 수 추정 + 청크 필요 여부 판단
# ─────────────────────────────────────────────────────────────
def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    return int(len(text) / chars_per_token)


def available_token_budget(
    model_context_tokens: int = DEFAULT_MODEL_CONTEXT_TOKENS,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
) -> int:
    return max(model_context_tokens - reserved_tokens, 1)


def needs_chunking(
    text: str,
    model_context_tokens: int = DEFAULT_MODEL_CONTEXT_TOKENS,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> bool:
    budget = available_token_budget(model_context_tokens, reserved_tokens)
    return estimate_tokens(text, chars_per_token) > budget


# ─────────────────────────────────────────────────────────────
# 2. 청크 분할 (문단 -> 안 되면 문장 단위, overlap 포함)
# ─────────────────────────────────────────────────────────────
def _split_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    # 빈 줄로 구분된 단락이 없는 단일 블록 -> 문장 단위로 재시도
    sentences = re.split(r"(?<=[.!?다요음됨함])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_token_budget: int,
    overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> list[str]:
    units = _split_units(text)
    chunk_char_budget = int(chunk_token_budget * chars_per_token)
    overlap_char_budget = int(chunk_char_budget * overlap_ratio)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for u in units:
        u_len = len(u) + 1
        if current and current_len + u_len > chunk_char_budget:
            chunks.append(" ".join(current))
            # overlap: 마지막 청크의 뒤쪽 일부를 다음 청크 앞에 이어붙임
            overlap_units: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) > overlap_char_budget:
                    break
                overlap_units.insert(0, prev)
                overlap_len += len(prev) + 1
            current = overlap_units + [u]
            current_len = sum(len(x) + 1 for x in current)
        else:
            current.append(u)
            current_len += u_len

    if current:
        chunks.append(" ".join(current))

    if len(chunks) == 1 and estimate_tokens(chunks[0], chars_per_token) > chunk_token_budget:
        print("    [경고] 단일 문장/단락이 너무 길어 청크 예산을 초과합니다 (분할 불가, 그대로 진행)")

    return chunks


# ─────────────────────────────────────────────────────────────
# 3. 느슨한 정규화 기반 row 병합 (임베딩 없이 공백/기호 제거만)
# ─────────────────────────────────────────────────────────────
_NORMALIZE_RE = re.compile(r"[\s\(\)\[\]{}·,\.\-‑–—]")


def normalize_key_value(v: str) -> str:
    return _NORMALIZE_RE.sub("", (v or "").strip())


def merge_rows_with_alias(
    rows: list[dict], key_columns: list[str]
) -> tuple[list[dict], list[dict]]:
    """
    key_columns 기준으로 느슨하게 정규화한 키가 같으면 같은 행으로 보고 병합한다.
    - 한쪽이 비어있고("-" 포함) 다른 쪽이 채워져 있으면 채워진 값을 채택
    - 둘 다 채워져 있는데 값이 다르면 병합하지 않고 conflicts에 기록 (자동으로
      한쪽을 버리지 않음 -- 어느 쪽이 맞는지는 사람이 확인해야 함)
    key_columns가 비어있는 preset(event_timeline)은 병합하지 않고 그대로 이어붙인다.
    """
    if not key_columns:
        return rows, []

    merged: dict[tuple, dict] = {}
    conflicts: list[dict] = []
    unkeyed_counter = 0

    for row in rows:
        key = tuple(normalize_key_value(row.get(c, "")) for c in key_columns)
        if not any(key):
            # 키 컬럼이 전부 비어있으면 병합 대상으로 보지 않고 별도 보존
            unkeyed_counter += 1
            merged[("__unkeyed__", unkeyed_counter)] = dict(row)
            continue

        if key not in merged:
            merged[key] = dict(row)
            continue

        existing = merged[key]
        for col, val in row.items():
            if col in key_columns:
                continue  # 이미 key로 매칭됨 -- 원문 표기 차이(공백 등)는 충돌이 아님
            val = val.strip() if isinstance(val, str) else val
            existing_val = existing.get(col, "")
            if val and val != "-" and (not existing_val or existing_val == "-"):
                existing[col] = val
            elif val and val != "-" and existing_val and existing_val != "-" and val != existing_val:
                conflicts.append({"key": key, "column": col, "values": [existing_val, val]})

    return list(merged.values()), conflicts


def rows_to_markdown(header: list[str], rows: list[dict]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "-")) for h in header) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 4. 오케스트레이션
# ─────────────────────────────────────────────────────────────
def process_document(
    preset_id: str,
    full_text: str,
    context_before: str = "",
    context_after: str = "",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    model_context_tokens: int = DEFAULT_MODEL_CONTEXT_TOKENS,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    chunk_overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
    schema_scan_sample_chunks: int = DEFAULT_SCHEMA_SCAN_SAMPLE_CHUNKS,
) -> dict:
    preset = PRESETS.get(preset_id)
    if preset is None:
        return {"error": f"알 수 없는 preset_id: {preset_id}"}

    llm_kwargs = {
        k: v for k, v in {
            "model": model, "ollama_url": ollama_url,
            "timeout": timeout, "max_retries": max_retries,
        }.items() if v is not None
    }

    budget = available_token_budget(model_context_tokens, reserved_tokens)

    # ── 청크가 필요 없는 경우: 지금까지 만든 A-1/A-2를 그대로 사용 ──
    if not needs_chunking(full_text, model_context_tokens, reserved_tokens, chars_per_token):
        scan_result = (
            scan_schema(preset_id, full_text, **llm_kwargs)
            if preset.extension_budget_default is not None
            else {"skipped": True, "reason": "확장 열이 구조적으로 없는 preset"}
        )
        accepted = scan_result.get("accepted_columns") if not scan_result.get("skipped") else None
        extract_result = extract_table(
            preset_id, full_text, accepted_extension=accepted,
            context_before=context_before, context_after=context_after, **llm_kwargs,
        )
        return {"chunked": False, "num_chunks": 1, "scan": scan_result, **extract_result}

    # ── 청크 분할 ──
    chunks = chunk_text(full_text, budget, chunk_overlap_ratio, chars_per_token)
    print(f"    [D] 문서를 {len(chunks)}개 청크로 분할 (예산: 청크당 약 {budget} 토큰)")

    # ── 스키마는 대표 청크 샘플로 1회만 ──
    sample_text = "\n\n".join(chunks[:schema_scan_sample_chunks])
    scan_result = (
        scan_schema(preset_id, sample_text, **llm_kwargs)
        if preset.extension_budget_default is not None
        else {"skipped": True, "reason": "확장 열이 구조적으로 없는 preset"}
    )
    accepted = scan_result.get("accepted_columns") if not scan_result.get("skipped") else None
    expected_columns = resolve_final_columns(preset, accepted)

    # ── 청크마다 동일 스키마로 추출 반복 ──
    chunk_results = []
    all_rows: list[dict] = []
    header: list[str] = expected_columns

    for i, chunk in enumerate(chunks, 1):
        print(f"    [D] 청크 {i}/{len(chunks)} 추출 중...")
        res = extract_table(preset_id, chunk, accepted_extension=accepted, **llm_kwargs)
        chunk_results.append(res)
        h, rows = parse_markdown_table(res["table_markdown"])
        if h:
            header = h  # 마지막으로 관찰된 실제 헤더를 최종 병합 기준으로 사용
        all_rows.extend(rows)

    # ── 병합 (느슨한 정규화, 임베딩 없음) ──
    merged_rows, conflicts = merge_rows_with_alias(all_rows, preset.key_uniqueness_columns)
    merged_markdown = rows_to_markdown(header, merged_rows)

    # ── 병합된 전체 표에 대해 검증 재실행 (청크 단위 검증으로는 청크 간 중복을 못 잡음) ──
    final_validation = validate_extraction(preset_id, header, merged_rows, expected_columns)

    return {
        "chunked": True,
        "preset_id": preset_id,
        "num_chunks": len(chunks),
        "scan": scan_result,
        "expected_columns": expected_columns,
        "chunk_validations": [r.get("validation") for r in chunk_results],
        "merge_conflicts": conflicts,
        "table_markdown": merged_markdown,
        "validation": final_validation,
    }


# ─────────────────────────────────────────────────────────────
# 4.5. 중간 결과 저장/로드 (JSONL, 처리 즉시 1건씩 append)
#      -- roundtrip_verify.py와 동일한 패턴. D는 문서당 LLM 호출이
#      (스캔 1회 + 청크별 추출 N회) 여러 번이라 중간에 죽었을 때
#      손실이 더 크므로, 오히려 여기서 더 필요하다.
# ─────────────────────────────────────────────────────────────
def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # 일부 파일시스템/환경은 fsync를 지원하지 않음 -> 무시


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"    [경고] {path.name} {line_no}번째 줄 파싱 실패 -> 무시")
    return results


# ─────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D 단계: 긴 문서를 청크로 나눠 A-1/A-2를 실행합니다.")
    parser.add_argument("--longtext", required=True, help="tables_longtext.json")
    parser.add_argument("--classification", required=True, help="preset_classifier.py 출력")
    parser.add_argument("--output", default="chunk_extract_result.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--model-context-tokens", type=int, default=DEFAULT_MODEL_CONTEXT_TOKENS,
                         help="모델 컨텍스트 한도 (gpt-oss:120b-cloud 기본값: 131072)")
    parser.add_argument("--reserved-tokens", type=int, default=DEFAULT_RESERVED_TOKENS,
                         help="프롬프트 템플릿/출력을 위해 미리 빼둘 토큰 수")
    parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN,
                         help="토큰 수 근사 계산에 쓸 문자/토큰 비율")
    parser.add_argument("--chunk-overlap-ratio", type=float, default=DEFAULT_CHUNK_OVERLAP_RATIO)
    parser.add_argument("--schema-scan-sample-chunks", type=int, default=DEFAULT_SCHEMA_SCAN_SAMPLE_CHUNKS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="--output과 같은 폴더의 중간 저장 파일(.jsonl)을 읽어 이미 처리된 item_id는 건너뛰고 이어서 처리",
    )
    args = parser.parse_args()

    with open(args.longtext, encoding="utf-8") as f:
        longtext_entries = {e["table_id"]: e for e in json.load(f)}
    with open(args.classification, encoding="utf-8") as f:
        classifications = json.load(f)

    llm_kwargs = {
        k: v for k, v in {
            "model": args.model, "ollama_url": args.ollama_url,
            "timeout": args.timeout, "max_retries": args.max_retries,
        }.items() if v is not None
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")

    # --resume: 기존 jsonl에서 error 없이 끝난 item_id는 정상 완료로 보고 건너뛴다.
    # LLM 호출 도중 죽었던 항목(error 있음)만 다시 시도한다.
    existing_results: dict[str, dict] = {}
    if args.resume and jsonl_path.exists():
        for r in _read_jsonl(jsonl_path):
            iid = r.get("item_id")
            if iid is not None:
                existing_results[iid] = r  # 같은 item_id가 여러 번 있으면 마지막 것으로 덮어씀
        done_ids = {iid for iid, r in existing_results.items() if not r.get("error")}
        print(f"[재개 모드] 기존 결과 {len(existing_results)}건 로드, 완료된 {len(done_ids)}건은 건너뜁니다.")
    else:
        done_ids = set()
        if jsonl_path.exists():
            print(f"[주의] --resume 없이 실행되어 기존 {jsonl_path.name}을 새로 덮어씁니다.")
            jsonl_path.unlink()

    results: list[dict] = [r for iid, r in existing_results.items() if iid in done_ids]

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
        if args.resume and item_id in done_ids:
            print(f"{item_id} 이미 완료됨 -> 건너뜀")
            continue

        print(f"{item_id} ({preset_id}) 처리 중...")
        try:
            result = process_document(
                preset_id,
                entry.get("long_text", ""),
                context_before=entry.get("context_before", ""),
                context_after=entry.get("context_after", ""),
                model_context_tokens=args.model_context_tokens,
                reserved_tokens=args.reserved_tokens,
                chars_per_token=args.chars_per_token,
                chunk_overlap_ratio=args.chunk_overlap_ratio,
                schema_scan_sample_chunks=args.schema_scan_sample_chunks,
                **llm_kwargs,
            )
        except Exception as e:
            print(f"    [오류] {item_id} 처리 실패: {e}")
            result = {"table_markdown": "", "validation": {}, "error": str(e)}

        result["item_id"] = item_id
        results.append(result)
        _append_jsonl(jsonl_path, result)  # <- 항목 처리 즉시 디스크에 저장 (핵심)

        if result.get("error"):
            continue

        v = result.get("validation", {})
        print(f"    -> chunked={result.get('chunked')}, num_chunks={result.get('num_chunks')}, "
              f"{v.get('row_count', 0)}행")
        if result.get("merge_conflicts"):
            print(f"    [경고] 병합 충돌 {len(result['merge_conflicts'])}건: {result['merge_conflicts'][:2]}")

    # 재개로 쌓였을 수 있는 중복/실패 잔여 라인을 정리하기 위해 최종 결과 기준으로 재작성
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n중간 저장(재개용): {jsonl_path}")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()