"""
text_to_table.py

임의의 장문(.txt) 하나를 표로 변환하는 엔드투엔드 스크립트.

C(preset_classifier) -> D(chunk_orchestrator, 내부에서 A-1/A-2 자동 호출)
까지만 실행한다. 표 -> 장문 round-trip 자기검증(E)은 여기서 다루지 않는다
(필요하면 별도로 roundtrip_verify.py를 직접 돌리면 된다).

지금까지의 다른 CLI들과 다른 점
-------------------------------
- 입력이 tables_longtext.json이 아니라 순수 .txt 파일이다. 즉
  context_before/after는 표 주변 문맥이라는 개념 자체가 없으므로 빈 문자열이다.
- C가 preset_id=None(폴백)을 반환하면 D를 실행할 스키마 기준이 없다.
  자유 스키마(A 단독) 경로는 아직 구현돼 있지 않으므로, 여기서는 명확한
  오류 메시지와 함께 중단한다 (README의 "알려진 한계"에 기록된 항목).

사용 예
-------
    python text_to_table.py --input essay.txt
    python text_to_table.py --input report.txt --output report_table.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from preset_classifier import classify_and_select_preset
from chunk_orchestrator import process_document, needs_chunking, maybe_split_by_subject
from schema_extract import parse_markdown_table
import preset_selection

DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────
# 0. 인코딩에 안전한 텍스트 파일 읽기
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
# 1. 오케스트레이션 (C -> D)
# ─────────────────────────────────────────────────────────────
def convert_text_to_table(
    text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    model_context_tokens: int | None = None,
    reserved_tokens: int | None = None,
    chars_per_token: float | None = None,
    chunk_overlap_ratio: float | None = None,
    schema_scan_sample_chunks: int | None = None,
    quality_chunk_tokens: int | None = None,
    check_subject_split: bool = True,
    verified_preset_selection: bool = False,
    genre: str = "informative",
    min_accept_score: float | None = None,
    selection_sample_chars: int | None = None,
) -> dict:
    llm_kwargs = {"model": model, "ollama_url": ollama_url, "timeout": timeout, "max_retries": max_retries}
    selection_kwargs = {} if min_accept_score is None else {"min_accept_score": min_accept_score}

    chunk_kwargs = {
        k: v for k, v in {
            "model_context_tokens": model_context_tokens,
            "reserved_tokens": reserved_tokens,
            "chars_per_token": chars_per_token,
            "chunk_overlap_ratio": chunk_overlap_ratio,
            "schema_scan_sample_chunks": schema_scan_sample_chunks,
            "quality_chunk_tokens": quality_chunk_tokens,
        }.items() if v is not None
    }

    # ── C+ (선택적): 후보 preset 스크리닝 + round-trip 검증 기반 선택 ──
    if verified_preset_selection:
        chunk_kwargs_for_check = {
            k: v for k, v in chunk_kwargs.items()
            if k in ("model_context_tokens", "reserved_tokens", "chars_per_token")
        }
        chunked_needed = needs_chunking(text, **chunk_kwargs_for_check)

        if chunked_needed:
            # ── 표본으로 preset만 결정 (A안): 청크마다/후보마다 전체를 다시
            # 추출하는 건 비용이 너무 커서, C(preset_classifier)가 이미 안고
            # 있는 "표본 기반 판단" 리스크 수준을 C+도 그대로 유지한다.
            # preset이 결정되면 실제 청크 추출은 그대로 D가 전담한다.
            sample_chars = selection_sample_chars or preset_selection.DEFAULT_SELECTION_SAMPLE_CHARS
            sample_text = text if len(text) <= sample_chars else text[:sample_chars]
            print(f"[C+] 텍스트가 길어 표본({len(sample_text)}자)으로 preset을 먼저 결정합니다...")
            selection = preset_selection.select_preset_with_verification(
                sample_text, genre=genre, **llm_kwargs, **selection_kwargs,
            )
            if not selection.get("success"):
                return {
                    "success": False,
                    "stage_failed": "C+",
                    "selection": selection,
                    "reason": selection.get("reason", "C+ 선택 실패 (표본 기반)"),
                }

            preset_id = selection["selected_preset_id"]
            print(f"    -> 표본 기반 채택: {preset_id} (점수={selection.get('selection_score')}, "
                  f"임계치 통과={selection.get('selection_passed_threshold')}) -- 전체 문서는 D가 청크 추출합니다")

            print("[D] 표를 추출하는 중...")
            extraction = process_document(
                preset_id, text, context_before="", context_after="",
                check_subject_split=check_subject_split, **llm_kwargs, **chunk_kwargs,
            )
            table_markdown = extraction.get("table_markdown", "")
            row_count = extraction.get("validation", {}).get("row_count", 0)
            print(f"    -> chunked={extraction.get('chunked')}, num_chunks={extraction.get('num_chunks')}, {row_count}행 추출")

            result = {
                "success": True,
                "preset_id": preset_id,
                "selection": selection,
                "selection_used_sample": True,
                "extraction": extraction,
                "table_markdown": table_markdown,
            }
            if not table_markdown.strip():
                result["success"] = False
                result["stage_failed"] = "D"
                result["reason"] = "표가 비어있게 추출됨"
            return result

        # ── 청크가 필요 없는 경우: 문서 전체로 C+ 전체 절차(스크리닝+표생성+E검증)를 그대로 수행 ──
        print("[C+] 후보 preset 스크리닝 + round-trip 검증 중...")
        selection = preset_selection.select_preset_with_verification(
            text, genre=genre, **llm_kwargs, **selection_kwargs,
        )
        if not selection.get("success"):
            return {
                "success": False,
                "stage_failed": "C+",
                "selection": selection,
                "reason": selection.get("reason", "C+ 선택 실패"),
            }

        preset_id = selection["selected_preset_id"]
        table_markdown = selection.get("table_markdown", "")
        print(f"    -> 채택된 preset: {preset_id} (점수={selection.get('selection_score')}, "
              f"임계치 통과={selection.get('selection_passed_threshold')})")

        subject_split = None
        if check_subject_split and table_markdown.strip():
            header, rows = parse_markdown_table(table_markdown)
            subject_split = maybe_split_by_subject(preset_id, header, rows, llm_kwargs)

        result = {
            "success": bool(table_markdown.strip()),
            "preset_id": preset_id,
            "selection": selection,
            "selection_used_sample": False,
            "extraction": {**selection["extraction"], "subject_split": subject_split},
            "table_markdown": table_markdown,
        }
        if not table_markdown.strip():
            result["success"] = False
            result["stage_failed"] = "C+"
            result["reason"] = "표가 비어있게 추출됨"
        return result

    # ── C: 축 분류 + preset 선택 ──
    print("[C] 텍스트를 분류하는 중...")
    classification = classify_and_select_preset(text, **llm_kwargs)
    preset_id = classification.get("preset_id")

    if not preset_id:
        return {
            "success": False,
            "stage_failed": "C",
            "classification": classification,
            "reason": classification.get("fallback_reason", "알 수 없는 분류 실패"),
        }
    print(f"    -> preset: {preset_id}  (축: {classification['axes']})")

    # ── D: 스키마 제안(A-1) + 추출(A-2), 필요하면 청크 처리까지 ──
    print("[D] 표를 추출하는 중...")
    extraction = process_document(
        preset_id, text, context_before="", context_after="",
        check_subject_split=check_subject_split, **llm_kwargs, **chunk_kwargs,
    )
    table_markdown = extraction.get("table_markdown", "")
    row_count = extraction.get("validation", {}).get("row_count", 0)
    print(f"    -> chunked={extraction.get('chunked')}, num_chunks={extraction.get('num_chunks')}, {row_count}행 추출")

    result = {
        "success": True,
        "preset_id": preset_id,
        "classification": classification,
        "extraction": extraction,
        "table_markdown": table_markdown,
    }

    if not table_markdown.strip():
        result["success"] = False
        result["stage_failed"] = "D"
        result["reason"] = "표가 비어있게 추출됨"

    return result


# ─────────────────────────────────────────────────────────────
# 2. 사람이 바로 읽는 Markdown 리포트
# ─────────────────────────────────────────────────────────────
def _quote_block(text: str) -> str:
    """여러 줄 텍스트를 markdown blockquote로 안전하게 감싼다.
    줄마다 '> '를 붙이지 않으면 빈 줄에서 인용구가 끊겨, 뒷부분이 마치
    잘린 것처럼 보인다 (pdf_table_extractor.py에서 겪었던 것과 같은 문제)."""
    return "\n".join(f"> {line}" for line in text.splitlines()) or "> (내용 없음)"


def build_report_markdown(input_name: str, text: str, result: dict) -> str:
    lines = [f"# {input_name} → 표 변환 결과\n"]

    if not result["success"]:
        lines.append(f"**실패** (단계: {result.get('stage_failed')}) — {result.get('reason')}\n")
        lines.append("## 원문\n")
        lines.append(_quote_block(text) + "\n")
        return "\n".join(lines)

    extraction = result["extraction"]

    lines.append(f"**preset**: {result['preset_id']}")

    selection = result.get("selection")
    if selection is not None:
        used_sample = result.get("selection_used_sample", False)
        mode_label = "C+ (표본 기반 preset 선택 -> D가 전체 문서 청크 추출)" if used_sample else "C+ (후보 스크리닝 + round-trip 검증, 청크 불필요)"
        lines.append(f"**선택 방식**: {mode_label}")
        if selection.get("fallback_used"):
            lines.append(f"**폴백 사용됨**: {selection.get('fallback_reason')} -> 기존 C(단일 분류)로 대체")
        else:
            lines.append(
                f"**선택 점수**: {selection.get('selection_score')} "
                f"(임계치 통과: {selection.get('selection_passed_threshold')})"
            )
            tried = selection.get("candidates_tried", [])
            evaluated = selection.get("candidates_evaluated", [])
            if len(tried) > 1:
                score_str = ", ".join(f"{e['preset_id']}={e['score']}" for e in evaluated)
                note = " (표본 기준 점수 -- 실제 표는 D가 전체 문서로 재추출함)" if used_sample else ""
                lines.append(f"**시도된 후보**: {tried} (점수: {score_str or '평가 실패'}){note}")
        if used_sample:
            lines.append(f"**청크 처리**: {extraction.get('chunked')} (총 {extraction.get('num_chunks')}개 청크)")
        lines.append("")
    else:
        lines.append(f"**청크 처리**: {extraction.get('chunked')} (총 {extraction.get('num_chunks')}개 청크)\n")

    lines.append("## 원문\n")
    lines.append(_quote_block(text) + "\n")

    scan = extraction.get("scan", {})
    if not scan.get("skipped") and scan.get("accepted_columns"):
        cols = [c["name"] for c in scan["accepted_columns"]]
        lines.append(f"## 스캔된 확장 컬럼 (A-1)\n{cols}\n")

    if extraction.get("merge_conflicts"):
        lines.append(f"## 청크 병합 충돌 (D)\n{extraction['merge_conflicts']}\n")

    subject_split = extraction.get("subject_split")
    if subject_split and subject_split.get("mixed"):
        labels = [t["label"] for t in subject_split["tables"]]
        lines.append(
            f"## ⚠️ 주제 섞임 감지 -- {len(labels)}개 표로 분리됨: {labels}\n"
            f"(아래 '추출된 표'는 분리 전 병합본입니다. 분리된 표는 이 아래에 각각 표시됩니다.)\n"
        )

    lines.append("## 추출된 표\n")
    lines.append(result["table_markdown"] or "(표 없음)")

    if subject_split and subject_split.get("mixed"):
        for t in subject_split["tables"]:
            lines.append(f"\n### 분리된 표 — {t['label']} ({t['row_count']}행)\n")
            lines.append(t["table_markdown"])

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 3. CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="임의의 장문(.txt)을 표로 변환합니다.")
    parser.add_argument("--input", required=True, help="입력 텍스트 파일(.txt)")
    parser.add_argument("--output", default=None, help="결과 .md 경로 (기본: 입력 파일명 기반)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--model-context-tokens", type=int, default=None)
    parser.add_argument("--reserved-tokens", type=int, default=None)
    parser.add_argument("--chars-per-token", type=float, default=None)
    parser.add_argument("--chunk-overlap-ratio", type=float, default=None)
    parser.add_argument(
        "--schema-scan-sample-chunks", type=int, default=None,
        help="A-1 스캔에 쓸 앞부분 청크 개수. 0 이하로 주면 모든 청크를 개별 스캔 후 "
             "합산한다 (문서 전체 대상 스캔, 호출 수는 늘어남). 기본값은 D의 기본값(2)."
    )
    parser.add_argument(
        "--quality-chunk-tokens", type=int, default=None,
        help="지정하면 컨텍스트 오버플로 여부와 무관하게 이 크기로 청크를 강제 분할한다 "
             "(lost-in-the-middle/context rot 완화용). 미지정 시 기존처럼 오버플로 위험이 "
             "있을 때만 청크한다."
    )
    parser.add_argument(
        "--no-subject-split", action="store_true",
        help="최종 표(vertical_entity/listing)에 대해 주제 섞임 감지+분리를 돌리지 않는다. "
             "기본은 켜져 있음."
    )
    parser.add_argument(
        "--verified-preset-selection", action="store_true",
        help="C(단일 분류) 대신 C+(후보 preset 스크리닝 + round-trip 검증 기반 선택)를 사용한다. "
             "텍스트가 청크가 필요할 만큼 길면 자동으로 기존 C→D로 폴백한다."
    )
    parser.add_argument("--genre", choices=["informative", "narrative"], default="informative",
                         help="C+ 사용 시 round-trip 재-verbalize에 쓰이는 문체.")
    parser.add_argument(
        "--min-accept-score", type=float, default=None,
        help="C+ 사용 시, 후보를 '채택 가능'으로 보는 최소 round-trip 점수 (기본 0.75)."
    )
    parser.add_argument(
        "--selection-sample-chars", type=int, default=None,
        help="C+ 사용 + 텍스트가 청크가 필요할 만큼 길 때, preset 결정에 쓸 앞부분 표본 "
             "문자 수 상한 (기본 16000). 표본으로 preset만 정하고, 실제 청크 추출은 D가 "
             "전체 문서로 수행한다."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    text = read_text_file(input_path).strip()
    print(f"입력: {input_path} ({len(text)}자)\n")

    if not text:
        print("[오류] 입력 파일이 비어있습니다.")
        sys.exit(1)

    result = convert_text_to_table(
        text,
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        model_context_tokens=args.model_context_tokens,
        reserved_tokens=args.reserved_tokens,
        chars_per_token=args.chars_per_token,
        chunk_overlap_ratio=args.chunk_overlap_ratio,
        schema_scan_sample_chunks=args.schema_scan_sample_chunks,
        quality_chunk_tokens=args.quality_chunk_tokens,
        check_subject_split=not args.no_subject_split,
        verified_preset_selection=args.verified_preset_selection,
        genre=args.genre,
        min_accept_score=args.min_accept_score,
        selection_sample_chars=args.selection_sample_chars,
    )

    output_md = Path(args.output) if args.output else input_path.with_suffix(".table.md")
    output_json = output_md.with_suffix(".json")

    output_md.write_text(build_report_markdown(input_path.name, text, result), encoding="utf-8")
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if not result["success"]:
        print(f"\n[실패] {result['stage_failed']} 단계에서 중단됨: {result['reason']}")
        print(f"(preset 폴백에 대한 자유 스키마 경로는 아직 구현되어 있지 않습니다.)")
        print(f"자세한 내용: {output_md}")
        sys.exit(1)

    print(f"\n=== 최종 표 ===\n{result['table_markdown']}\n")
    print(f"리포트: {output_md}")
    print(f"전체 결과(JSON): {output_json}")


if __name__ == "__main__":
    main()