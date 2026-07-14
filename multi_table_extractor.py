"""
multi_table_extractor.py

[F 단계] 하나의 장문에서 여러 개의 표를 반복 추출

지금까지의 파이프라인(C->D)은 "텍스트 1개 -> 표 1개"로 고정되어 있었다.
그런데 정보가 여러 겹인 문서(인물도 여러 명, 사건도 여러 개, 시간 흐름도
있는 소설/보고서 등)는 표 하나로 담을 수 없는 정보를 억지로 하나에
눌러 담게 되어 결과가 "뻔해" 보인다.

이 스크립트는 표를 하나 뽑을 때마다 E(round-trip)의 누락 체크를 정지
신호로 재활용한다: 원문의 salient token(숫자/날짜/고유명사 등)이 지금까지
뽑은 표들로 충분히 커버되지 않았다면, "아직 담기지 못한 정보가 있다"는
뜻이므로 다른 관점(preset)으로 표를 하나 더 시도한다. 새 검증 장치를
만들지 않고 이미 있는 E의 인프라를 반복 실행의 정지 조건으로 쓰는 것.

"몇 개의 표가 필요한지"를 LLM의 자기 판단에 맡기지 않고, 측정된 coverage
수치로 결정한다는 점에서 지금까지의 설계 원칙("LLM 자기절제를 믿지
않는다")을 그대로 잇는다.

max_tables는 비용 상한(기본 10)이고, coverage 기반 정지 조건과는 별개
레이어다 -- None으로 주면 상한을 풀지만, 무한루프 방지를 위한 내부
안전판(DEFAULT_HARD_SAFETY_LIMIT)은 항상 걸려 있다.

사용 예
-------
    python multi_table_extractor.py --input long_document.txt
    python multi_table_extractor.py --input long_document.txt --max-tables 0   # 무제한(안전판까지)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preset_classifier import classify_and_select_preset, DEFAULT_CLASSIFICATION_SAMPLE_CHARS
from chunk_orchestrator import process_document, chunk_text
from table_to_longtext import build_verbalization_prompt, call_ollama_generate, check_value_coverage
from roundtrip_verify import extract_salient_tokens
from schema_extract import parse_markdown_table

DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

DEFAULT_MAX_TABLES = 10          # None(또는 CLI에서 0)이면 무제한
DEFAULT_HARD_SAFETY_LIMIT = 50   # max_tables를 풀어도 항상 걸리는 안전판
DEFAULT_COVERAGE_STOP_THRESHOLD = 0.9


# ─────────────────────────────────────────────────────────────
# 1. "이미 이런 표를 뽑았다" 힌트 구성
# ─────────────────────────────────────────────────────────────
def _extract_row_keys(table_markdown: str) -> set[str]:
    """표의 첫 번째 컬럼(행 키) 값들을 집합으로 추출 (내용 중복 판정용)."""
    header, rows = parse_markdown_table(table_markdown)
    if not header:
        return set()
    key_col = header[0]
    return {row.get(key_col, "").strip() for row in rows if row.get(key_col, "").strip()}


# vertical_entity/listing/event_timeline은 core_columns가 문서 내용과 무관하게
# 항상 고정(예: 속성명/값, 항목/값, 시점/사건)이다. 이런 preset은 컬럼명으로
# 중복을 판단하면 내용이 완전히 달라도 항상 "같은 구조"로 잡혀 즉시 중단된다
# -- 실제로 해리포터 텍스트에서 이 문제로 표가 1개에서 멈췄다. 컬럼명 대신
# 행 라벨(내용)의 겹치는 비율로 판단해야 한다.
_FIXED_SCHEMA_PRESETS = {"vertical_entity", "listing", "event_timeline"}
_DUPLICATE_ROW_OVERLAP_THRESHOLD = 0.7
_MIN_ROW_KEY_NOVELTY_RATIO = 0.3  # 이 표의 행 라벨 중 최소 이 비율은 "새로운 내용"이어야
                                   # salient token 진전이 없어도 정체로 안 본다


def build_avoidance_note(previous_tables: list[dict]) -> str:
    if not previous_tables:
        return ""
    lines = [
        "[참고] 이 텍스트에서는 이미 다음과 같은 관점으로 표를 추출했습니다. "
        "가능하면 같은 개체/사건을 다른 관점(다른 행 기준)에서 보거나, "
        "아직 표로 안 옮겨진 다른 정보를 찾아보세요:"
    ]
    for i, t in enumerate(previous_tables, 1):
        cols = ", ".join(t.get("expected_columns", []))
        line = f"{i}. preset={t['preset_id']}, 컬럼=[{cols}]"
        if t["preset_id"] in _FIXED_SCHEMA_PRESETS:
            # 컬럼명이 항상 같아서 그것만으론 구분이 안 되니, 실제로 어떤 내용을
            # 다뤘는지(행 라벨 일부)까지 보여줘서 회피 방향을 구체적으로 잡아준다.
            sample_keys = list(_extract_row_keys(t.get("table_markdown", "")))[:5]
            if sample_keys:
                line += f", 다룬 내용 예시=[{', '.join(sample_keys)}]"
        lines.append(line)
    return "\n".join(lines)


def _is_duplicate_structure(
    preset_id: str, expected_columns: list[str], table_markdown: str, previous_tables: list[dict]
) -> bool:
    if preset_id in _FIXED_SCHEMA_PRESETS:
        new_keys = _extract_row_keys(table_markdown)
        if not new_keys:
            return False
        for t in previous_tables:
            if t["preset_id"] != preset_id:
                continue
            prev_keys = _extract_row_keys(t.get("table_markdown", ""))
            if not prev_keys:
                continue
            overlap = len(new_keys & prev_keys) / len(new_keys | prev_keys)
            if overlap >= _DUPLICATE_ROW_OVERLAP_THRESHOLD:
                return True
        return False

    cols = set(expected_columns)
    for t in previous_tables:
        if t["preset_id"] == preset_id and set(t.get("expected_columns", [])) == cols:
            return True
    return False


def _row_key_novelty_ratio(table_markdown: str, all_row_keys_seen: set[str]) -> float:
    """
    이 표의 행 라벨 중 지금까지 어떤 표에도 안 나왔던 비율. preset과 무관하게
    누적된 all_row_keys_seen과 비교한다 -- salient token(숫자/날짜) 진전이
    없어도, 표가 실제로 새로운 개체/속성을 다뤘다면 "정체"로 보지 않기 위한
    보조 신호다.
    """
    keys = _extract_row_keys(table_markdown)
    if not keys:
        return 0.0
    novel = keys - all_row_keys_seen
    return len(novel) / len(keys)


# ─────────────────────────────────────────────────────────────
# 2. 반복 추출 오케스트레이션
# ─────────────────────────────────────────────────────────────
def extract_multiple_tables(
    text: str,
    max_tables: int | None = DEFAULT_MAX_TABLES,
    coverage_stop_threshold: float = DEFAULT_COVERAGE_STOP_THRESHOLD,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    genre: str = "informative",
    classification_sample_chars: int = DEFAULT_CLASSIFICATION_SAMPLE_CHARS,
    include_entities: bool = True,
    **chunk_kwargs,
) -> dict:
    llm_kwargs = {"model": model, "ollama_url": ollama_url, "timeout": timeout, "max_retries": max_retries}

    salient_tokens = extract_salient_tokens(text, include_entities=include_entities)
    covered: set[str] = set()
    all_row_keys_seen: set[str] = set()
    tables: list[dict] = []
    stop_reason = ""

    # 분류용 텍스트를 문서 전체에서 구간별로 나눠, 회차마다 다른 구간을 보여준다.
    # (안 그러면 preset_classifier의 sample_chars 자체 제한 때문에 매 회차 똑같은
    # 앞부분만 반복해서 보게 되고, 뒷부분에 있는 서로 다른 정보를 전혀 못 봄)
    classification_segments = chunk_text(
        text,
        chunk_token_budget=max(classification_sample_chars // 2, 500),
        overlap_ratio=0.0,
        chars_per_token=2.0,
    )
    print(f"    [F] 분류용으로 문서를 {len(classification_segments)}개 구간으로 나눠 순환 사용")

    hard_limit = DEFAULT_HARD_SAFETY_LIMIT if max_tables is None else min(max_tables, DEFAULT_HARD_SAFETY_LIMIT)

    for iteration in range(1, hard_limit + 1):
        segment_idx = (iteration - 1) % len(classification_segments)
        classification_text = classification_segments[segment_idx]
        print(f"[F] {iteration}번째 표 시도 중... (분류 구간 {segment_idx + 1}/{len(classification_segments)})")
        avoid_note = build_avoidance_note(tables)
        classification = classify_and_select_preset(
            classification_text, avoid_note=avoid_note,
            sample_chars=len(classification_text) + 1, **llm_kwargs,
        )
        preset_id = classification.get("preset_id")

        if not preset_id:
            stop_reason = f"{iteration}번째 시도에서 분류 실패: {classification.get('fallback_reason')}"
            print(f"    -> 중단: {stop_reason}")
            break

        extraction = process_document(preset_id, text, context_before="", context_after="", **llm_kwargs, **chunk_kwargs)
        table_markdown = extraction.get("table_markdown", "")
        expected_columns = extraction.get("expected_columns", [])

        if not table_markdown.strip():
            stop_reason = f"{iteration}번째 시도에서 빈 표 추출"
            print(f"    -> 중단: {stop_reason}")
            break

        if _is_duplicate_structure(preset_id, expected_columns, table_markdown, tables):
            stop_reason = f"{iteration}번째 시도가 이전 표와 구조(preset+컬럼)가 동일해 중단"
            print(f"    -> 중단: {stop_reason}")
            break

        # 이 표를 다시 장문으로 복원해서, 원문 salient token이 얼마나 더 커버됐는지 측정
        verbalization_prompt = build_verbalization_prompt(table_markdown, "", "", genre=genre)
        regenerated = call_ollama_generate(verbalization_prompt, **llm_kwargs)

        remaining = [t for t in salient_tokens if t not in covered]
        cov = check_value_coverage(regenerated, remaining)
        newly_covered = [t for t in remaining if t not in cov["missing_values"]]
        covered.update(newly_covered)

        # salient token(숫자/날짜/개체명)과는 별개로, 이 표가 실제로 새로운
        # 행 라벨(개체/속성)을 다뤘는지도 측정 -- coverage만으로는 "정체"를
        # 오판할 수 있어서(예: 숫자가 적은 서사 텍스트) 보조 신호로 같이 본다.
        novelty_ratio = _row_key_novelty_ratio(table_markdown, all_row_keys_seen)
        all_row_keys_seen |= _extract_row_keys(table_markdown)

        coverage_ratio = len(covered) / len(salient_tokens) if salient_tokens else 1.0
        tables.append({
            "preset_id": preset_id,
            "classification": classification,
            "table_markdown": table_markdown,
            "expected_columns": expected_columns,
            "chunked": extraction.get("chunked"),
            "newly_covered_count": len(newly_covered),
            "cumulative_coverage": round(coverage_ratio, 3),
            "row_key_novelty_ratio": round(novelty_ratio, 3),
        })
        print(f"    -> preset={preset_id}, 새로 커버된 토큰={len(newly_covered)}, "
              f"누적 coverage={coverage_ratio:.2f}, 행 라벨 신선도={novelty_ratio:.2f}")

        if coverage_ratio >= coverage_stop_threshold:
            stop_reason = f"누적 coverage {coverage_ratio:.2f} >= 목표({coverage_stop_threshold})"
            break
        if not newly_covered and novelty_ratio < _MIN_ROW_KEY_NOVELTY_RATIO and iteration > 1:
            stop_reason = (
                f"추가 표가 새로운 정보를 담지 못함 "
                f"(salient token 진전 없음, 행 라벨 신선도={novelty_ratio:.2f} < {_MIN_ROW_KEY_NOVELTY_RATIO})"
            )
            break
    else:
        if max_tables is not None and hard_limit == max_tables:
            stop_reason = f"표 개수 상한({max_tables}) 도달"
        else:
            stop_reason = f"안전판({hard_limit}개) 도달 -- max_tables=None이거나 매우 크게 설정해도 항상 걸리는 상한"

    final_coverage = len(covered) / len(salient_tokens) if salient_tokens else 1.0
    return {
        "num_tables": len(tables),
        "tables": tables,
        "final_coverage": round(final_coverage, 3),
        "stop_reason": stop_reason,
    }


# ─────────────────────────────────────────────────────────────
# 3. 사람이 바로 읽는 Markdown 리포트
# ─────────────────────────────────────────────────────────────
def build_report_markdown(input_name: str, text: str, result: dict) -> str:
    lines = [f"# {input_name} → 표 {result['num_tables']}개 추출 결과\n"]
    lines.append(f"**최종 누적 coverage**: {result['final_coverage']}")
    lines.append(f"**정지 사유**: {result['stop_reason']}\n")

    for i, t in enumerate(result["tables"], 1):
        lines.append(f"## 표 {i} — preset: {t['preset_id']}\n")
        lines.append(f"- 컬럼: {t['expected_columns']}")
        lines.append(f"- 이 표로 새로 커버된 원문 토큰 수: {t['newly_covered_count']}")
        lines.append(f"- 누적 coverage: {t['cumulative_coverage']}\n")
        lines.append(t["table_markdown"])
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 0. 인코딩에 안전한 텍스트 파일 읽기 (text_to_table.py와 동일 패턴)
# ─────────────────────────────────────────────────────────────
def read_text_file(path: Path) -> str:
    """
    UTF-8 -> UTF-8(BOM) -> CP949 순으로 시도한다. 한국어 Windows에서
    메모장 등으로 "ANSI" 저장한 파일은 실제로는 CP949(EUC-KR 확장)인
    경우가 많아, UTF-8로 바로 읽으면 UnicodeDecodeError가 난다.
    """
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1,
        f"'{path}'를 utf-8/utf-8-sig/cp949로 모두 읽지 못했습니다. "
        "메모장 등에서 '다른 이름으로 저장' 시 인코딩을 UTF-8로 지정해 다시 저장해보세요."
    )


# ─────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="하나의 장문에서 여러 개의 표를 반복 추출합니다.")
    parser.add_argument("--input", required=True, help="입력 텍스트 파일(.txt)")
    parser.add_argument("--output", default=None, help="결과 .md 경로 (기본: 입력 파일명 기반)")
    parser.add_argument(
        "--max-tables", type=int, default=DEFAULT_MAX_TABLES,
        help="표 개수 상한 (기본 10). 0을 주면 무제한으로 처리하되, "
             f"내부 안전판({DEFAULT_HARD_SAFETY_LIMIT}개)은 항상 적용된다."
    )
    parser.add_argument("--coverage-stop-threshold", type=float, default=DEFAULT_COVERAGE_STOP_THRESHOLD)
    parser.add_argument(
        "--classification-sample-chars", type=int, default=DEFAULT_CLASSIFICATION_SAMPLE_CHARS,
        help="분류 구간 하나의 문자 수 상한. 문서를 이 크기로 나눠 회차마다 다른 "
             "구간을 분류에 사용한다."
    )
    parser.add_argument("--genre", choices=["informative", "narrative"], default="informative")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--model-context-tokens", type=int, default=None,
                         help="D 단계(청크 처리)로 그대로 전달됨. 모델 컨텍스트 한도.")
    parser.add_argument("--reserved-tokens", type=int, default=None,
                         help="D 단계로 그대로 전달됨. 프롬프트 템플릿/출력을 위해 미리 빼둘 토큰 수.")
    parser.add_argument("--chars-per-token", type=float, default=None,
                         help="D 단계로 그대로 전달됨. 토큰 수 근사 계산에 쓸 문자/토큰 비율.")
    parser.add_argument("--chunk-overlap-ratio", type=float, default=None,
                         help="D 단계로 그대로 전달됨. 청크 간 겹침 비율.")
    parser.add_argument(
        "--schema-scan-sample-chunks", type=int, default=None,
        help="D 단계로 그대로 전달됨. A-1 스캔에 쓸 앞부분 청크 개수. 0 이하면 모든 "
             "청크를 개별 스캔 후 합산(문서 전체 대상 스캔)."
    )
    parser.add_argument(
        "--quality-chunk-tokens", type=int, default=None,
        help="D 단계로 그대로 전달됨. 지정하면 컨텍스트 오버플로 여부와 무관하게 이 "
             "크기로 청크를 강제 분할한다 (lost-in-the-middle/context rot 완화용)."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    text = read_text_file(input_path).strip()
    print(f"입력: {input_path} ({len(text)}자)\n")

    max_tables = None if args.max_tables == 0 else args.max_tables

    chunk_kwargs = {
        k: v for k, v in {
            "model_context_tokens": args.model_context_tokens,
            "reserved_tokens": args.reserved_tokens,
            "chars_per_token": args.chars_per_token,
            "chunk_overlap_ratio": args.chunk_overlap_ratio,
            "schema_scan_sample_chunks": args.schema_scan_sample_chunks,
            "quality_chunk_tokens": args.quality_chunk_tokens,
        }.items() if v is not None
    }

    result = extract_multiple_tables(
        text,
        max_tables=max_tables,
        coverage_stop_threshold=args.coverage_stop_threshold,
        classification_sample_chars=args.classification_sample_chars,
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        genre=args.genre,
        **chunk_kwargs,
    )

    output_md = Path(args.output) if args.output else input_path.with_suffix(".multitable.md")
    output_json = output_md.with_suffix(".json")

    output_md.write_text(build_report_markdown(input_path.name, text, result), encoding="utf-8")
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 완료 ===")
    print(f"표 {result['num_tables']}개, 최종 coverage {result['final_coverage']}, 정지 사유: {result['stop_reason']}")
    print(f"리포트: {output_md}")
    print(f"전체 결과(JSON): {output_json}")


if __name__ == "__main__":
    main()