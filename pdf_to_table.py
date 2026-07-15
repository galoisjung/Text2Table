"""
pdf_to_table.py

텍스트 레이어가 있는("글자를 긁을 수 있는") PDF를 받아
1) 텍스트를 추출하고
2) text_to_table.py(단일 표) 또는 multi_table_extractor.py(여러 표)의
   파이프라인으로 표를 뽑아내는 통합 스크립트.

새로 만든 건 "PDF -> 텍스트" 부분뿐이다. C(preset_classifier)/D(chunk_orchestrator)
오케스트레이션은 text_to_table.py의 convert_text_to_table()과
multi_table_extractor.py의 extract_multiple_tables()를 그대로 가져다 쓴다 --
같은 로직을 두 번 짜지 않는다.

스캔본(이미지 PDF, 텍스트 레이어 없음)은 지원 범위 밖이다. docling 같은 무거운
OCR 파이프라인은 pdf_table_extractor.py(표 추출 전용)가 이미 담당하고 있고,
이 스크립트는 "이미 글자를 긁을 수 있는 PDF"를 텍스트로 펼치는 가벼운 경로만
맡는다. 추출된 텍스트가 페이지당 평균 길이 기준으로 너무 짧으면 스캔본일
가능성이 크다고 보고 경고와 함께 중단한다.

사용 예
-------
    python pdf_to_table.py --input novel.pdf                  # 표 1개 (text_to_table.py)
    python pdf_to_table.py --input novel.pdf --multi           # 여러 표 (multi_table_extractor.py)
    python pdf_to_table.py --input novel.pdf --multi --max-tables 0   # 무제한
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

from text_to_table import convert_text_to_table
from text_to_table import build_report_markdown as build_single_report
from multi_table_extractor import extract_multiple_tables
from multi_table_extractor import build_report_markdown as build_multi_report

DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

MIN_CHARS_PER_PAGE_WARNING = 20  # 이보다 적으면 텍스트 레이어가 없는 스캔본일 가능성


# ─────────────────────────────────────────────────────────────
# 1. PDF -> 텍스트 (가벼운 텍스트 레이어 추출, OCR 없음)
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> tuple[str, dict]:
    """
    pypdf로 텍스트 레이어를 페이지 순서대로 추출해 이어붙인다.
    docling 같은 무거운 OCR 파이프라인은 쓰지 않는다 -- 이미 텍스트를
    긁을 수 있는 PDF를 대상으로 하기 때문에 그럴 필요가 없다.
    """
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)

    page_texts = []
    for page in reader.pages:
        try:
            page_texts.append((page.extract_text() or "").strip())
        except Exception as e:
            print(f"    [경고] 페이지 추출 실패, 건너뜀: {e}")
            page_texts.append("")

    full_text = "\n\n".join(t for t in page_texts if t)
    avg_chars_per_page = len(full_text) / num_pages if num_pages else 0.0

    stats = {
        "num_pages": num_pages,
        "extracted_chars": len(full_text),
        "avg_chars_per_page": round(avg_chars_per_page, 1),
    }
    return full_text, stats


# ─────────────────────────────────────────────────────────────
# 2. CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="텍스트 레이어가 있는 PDF를 표로 변환합니다.")
    parser.add_argument("--input", required=True, help="입력 PDF 파일")
    parser.add_argument("--output", default=None, help="결과 .md 경로 (기본: 입력 파일명 기반)")
    parser.add_argument(
        "--force", action="store_true",
        help="추출된 텍스트가 너무 적어도(스캔본 의심) 경고만 하고 강행한다."
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="여러 표를 반복 추출한다 (multi_table_extractor.py). 기본은 표 1개."
    )
    parser.add_argument("--max-tables", type=int, default=None, help="[--multi] 표 개수 상한 (0=무제한)")
    parser.add_argument("--coverage-stop-threshold", type=float, default=None, help="[--multi]")
    parser.add_argument("--classification-sample-chars", type=int, default=None)
    parser.add_argument("--genre", choices=["informative", "narrative"], default="informative")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--model-context-tokens", type=int, default=None)
    parser.add_argument("--reserved-tokens", type=int, default=None)
    parser.add_argument("--chars-per-token", type=float, default=None)
    parser.add_argument("--chunk-overlap-ratio", type=float, default=None)
    parser.add_argument("--schema-scan-sample-chunks", type=int, default=None)
    parser.add_argument("--quality-chunk-tokens", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)

    print(f"[PDF] '{input_path}' 텍스트 추출 중...")
    text, stats = extract_text_from_pdf(input_path)
    print(f"    -> {stats['num_pages']}페이지, {stats['extracted_chars']}자 "
          f"(페이지당 평균 {stats['avg_chars_per_page']}자)")

    if stats["avg_chars_per_page"] < MIN_CHARS_PER_PAGE_WARNING and not args.force:
        print(
            f"\n[중단] 페이지당 평균 {stats['avg_chars_per_page']}자로 텍스트가 거의 안 뽑혔습니다.\n"
            f"이 PDF는 텍스트 레이어가 없는 스캔본(이미지 기반)일 가능성이 큽니다.\n"
            f"스캔본은 이 스크립트의 지원 범위 밖입니다 -- OCR이 필요하면 pdf_table_extractor.py "
            f"(docling 기반) 쪽을 검토하세요.\n"
            f"그래도 강행하려면 --force 를 붙이세요."
        )
        return

    # 추출된 텍스트 자체도 중간 산출물로 남겨서, 필요하면 text_to_table.py /
    # multi_table_extractor.py를 이 파일에 직접 돌려 더 세밀한 옵션을 쓸 수 있게 한다.
    extracted_txt_path = input_path.with_suffix(".extracted.txt")
    extracted_txt_path.write_text(text, encoding="utf-8")
    print(f"    추출된 텍스트 저장: {extracted_txt_path}")

    if not text.strip():
        print("\n[오류] 추출된 텍스트가 비어있습니다.")
        return

    common_kwargs = {
        "model": args.model,
        "ollama_url": args.ollama_url,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
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

    if args.multi:
        print("\n[표 추출] multi_table_extractor.py 로 여러 표 추출...")
        max_tables = None if args.max_tables == 0 else args.max_tables
        multi_kwargs = {
            k: v for k, v in {
                "max_tables": max_tables,
                "coverage_stop_threshold": args.coverage_stop_threshold,
                "classification_sample_chars": args.classification_sample_chars,
                "genre": args.genre,
            }.items() if v is not None
        }
        result = extract_multiple_tables(text, **common_kwargs, **multi_kwargs, **chunk_kwargs)
        report = build_multi_report(input_path.name, text, result)
        default_suffix = ".multitable.md"
        summary = f"표 {result['num_tables']}개, 최종 coverage {result['final_coverage']}, 정지 사유: {result['stop_reason']}"
    else:
        print("\n[표 추출] text_to_table.py 로 표 1개 추출...")
        result = convert_text_to_table(text, **common_kwargs, **chunk_kwargs)
        report = build_single_report(input_path.name, text, result)
        default_suffix = ".table.md"
        if result["success"]:
            summary = f"표 추출 성공 (preset: {result['preset_id']})"
        else:
            summary = f"실패 ({result['stage_failed']} 단계): {result['reason']}"

    output_md = Path(args.output) if args.output else input_path.with_suffix(default_suffix)
    output_json = output_md.with_suffix(".json")
    output_md.write_text(report, encoding="utf-8")
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 완료 ===")
    print(summary)
    print(f"리포트: {output_md}")
    print(f"전체 결과(JSON): {output_json}")


if __name__ == "__main__":
    main()
