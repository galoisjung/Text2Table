"""
roundtrip_verify.py

[E 단계] round-trip 재활용한 경량 자기검증

D(chunk_orchestrator.py)가 만든 table_markdown이 원문(long_text)의 내용을
얼마나 정확하고 빠짐없이 담았는지, 이미 만들어둔 표->장문 verbalization
인프라(table_to_longtext.py)를 그대로 재사용해 검증한다. 새 LLM 호출 방식을
따로 만들지 않고, 있는 부품을 다른 방향으로 조립하는 단계다.

두 가지 독립적인 체크를 한다 (하나로 뭉치지 않는 이유: 원인이 다르면
디버깅 방향도 달라야 하므로 -- 그리디 검증 철학을 여기서도 유지):

1. 환각(hallucination) 체크
   표에 있는 값들이 실제로 원문(long_text)에 있었는가?
   -> table_to_longtext.check_value_coverage(원문, 표의_값들) 그대로 재사용.
      missing_values로 나온 것들은 "표가 원문에 없는 걸 지어냈을 가능성"이다.

2. 누락(omission) 체크 -- 진짜 라운드트립
   표를 다시 장문으로 복원(table_to_longtext.build_verbalization_prompt +
   call_ollama_generate)한 뒤, 원문에서 뽑은 숫자/날짜/퍼센트 같은 "사실성이
   강한 토큰"들이 그 복원문에도 남아있는가?
   -> 복원문에서 사라졌다면,애초에 표가 그 정보를 담지 못했다는 뜻이다
      (표에 없는 걸 verbalization이 만들어낼 리 없으므로).
   이 체크는 원본 표(ground truth)가 없어도 동작한다 -- 최종 목표인
   "임의의 장문 -> 표"에서는 애초에 비교할 원본 표가 없기 때문에, 이 방식이
   유일하게 일반화 가능한 자기검증이다.

사용 예
-------
    python roundtrip_verify.py --longtext tables_longtext.json \\
        --extracted chunk_extract_result.json \\
        --output roundtrip_verify_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from table_to_longtext import (
    build_verbalization_prompt,
    call_ollama_generate,
    check_value_coverage,
    extract_cell_values_from_markdown,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

DEFAULT_HALLUCINATION_THRESHOLD = 0.9  # 표 값이 원문에 있어야 하는 비율 (엄격)
DEFAULT_OMISSION_THRESHOLD = 0.7       # 원문 사실이 라운드트립 후 남아있어야 하는 비율


# ─────────────────────────────────────────────────────────────
# 1. 원문에서 "사실성이 강한" 토큰 추출 (숫자/날짜/퍼센트)
#    -- 표가 없어도 원문만으로 뽑을 수 있어야 일반화된 검증이 됨
# ─────────────────────────────────────────────────────────────
_SALIENT_PATTERNS = [
    r"\d{4}[-‑]\d{2}[-‑]\d{2}",     # 날짜 (YYYY-MM-DD)
    r"-?\d[\d,]*\.?\d*%",            # 퍼센트
    r"\(?-?\d[\d,]*\.?\d*\)?원",     # 금액 (음수 괄호 표기 포함)
    r"(?<![\d.%])\d{4,}(?![\d.%])",  # 그 외 4자리 이상 숫자 (일련번호성 노이즈 방지)
]


def extract_salient_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for pattern in _SALIENT_PATTERNS:
        tokens.extend(re.findall(pattern, text))
    # 중복 제거하되 순서는 유지 (등장 빈도보다 "이런 사실이 있었다"가 중요)
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    return unique_tokens


# ─────────────────────────────────────────────────────────────
# 2. 검증 오케스트레이션
# ─────────────────────────────────────────────────────────────
def verify_extraction(
    original_long_text: str,
    candidate_table_markdown: str,
    context_before: str = "",
    context_after: str = "",
    genre: str = "informative",
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    omission_threshold: float = DEFAULT_OMISSION_THRESHOLD,
) -> dict:
    if not candidate_table_markdown.strip():
        return {
            "table_markdown": candidate_table_markdown,
            "hallucination_check": None,
            "omission_check": None,
            "regenerated_long_text": "",
            "passed": False,
            "fail_reason": "표가 비어있음",
        }

    # ── 1) 환각 체크: 표의 값이 원문에 실제로 있었는가 ──
    table_values = extract_cell_values_from_markdown(candidate_table_markdown)
    hallucination_check = check_value_coverage(original_long_text, table_values)

    # ── 2) 누락 체크: 표 -> 장문 복원 -> 원문의 사실성 토큰이 남아있는가 ──
    verbalization_prompt = build_verbalization_prompt(
        candidate_table_markdown, context_before, context_after, genre=genre
    )
    regenerated_long_text = call_ollama_generate(
        verbalization_prompt, model=model, ollama_url=ollama_url,
        timeout=timeout, max_retries=max_retries,
    )
    salient_tokens = extract_salient_tokens(original_long_text)
    omission_check = check_value_coverage(regenerated_long_text, salient_tokens)

    passed = (
        hallucination_check["coverage_ratio"] >= hallucination_threshold
        and omission_check["coverage_ratio"] >= omission_threshold
    )

    return {
        "table_markdown": candidate_table_markdown,
        "hallucination_check": hallucination_check,
        "omission_check": omission_check,
        "regenerated_long_text": regenerated_long_text,
        "passed": passed,
        "fail_reason": None if passed else _build_fail_reason(
            hallucination_check, omission_check, hallucination_threshold, omission_threshold
        ),
    }


def _build_fail_reason(hall: dict, omit: dict, hall_th: float, omit_th: float) -> str:
    reasons = []
    if hall["coverage_ratio"] < hall_th:
        reasons.append(
            f"환각 의심 {len(hall['missing_values'])}건 "
            f"(표에는 있는데 원문엔 없는 값, coverage={hall['coverage_ratio']})"
        )
    if omit["coverage_ratio"] < omit_th:
        reasons.append(
            f"누락 의심 {len(omit['missing_values'])}건 "
            f"(원문에는 있는데 라운드트립 후 사라진 값, coverage={omit['coverage_ratio']})"
        )
    return "; ".join(reasons)


# ─────────────────────────────────────────────────────────────
# 2.5. 중간 결과 저장/로드 (JSONL, 처리 즉시 1건씩 append)
# ─────────────────────────────────────────────────────────────
def _append_jsonl(path: Path, obj: dict) -> None:
    """
    결과 1건을 즉시 파일에 append하고 flush+fsync한다.
    round-trip 검증은 항목마다 LLM 호출이 있어 도중에 죽을 수 있으므로
    (네트워크 오류, Ctrl+C 등) 이미 append된 항목들은 디스크에 남아 있다.
    """
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
# 2.6. 사람이 바로 읽는 Markdown 리포트
# ─────────────────────────────────────────────────────────────
def build_markdown_report(
    results: list[dict],
    longtext_entries: dict[str, dict],
    extracted_entries_by_id: dict[str, dict],
) -> str:
    lines = ["# Round-trip 검증 결과\n"]
    passed_count = len([r for r in results if r.get("passed")])
    lines.append(f"총 {len(results)}건 중 통과 {passed_count}건, 실패 {len(results) - passed_count}건\n")
    lines.append("---\n")

    for r in results:
        item_id = r.get("item_id", "(알수없음)")
        entry = longtext_entries.get(item_id, {})
        ex = extracted_entries_by_id.get(item_id, {})
        preset_id = ex.get("preset_id", "-")
        page_number = entry.get("page_number", "-")

        if r.get("error"):
            status = f"⚠️ 오류: {r['error']}"
        elif r.get("passed"):
            status = "✅ 통과"
        else:
            status = f"❌ 실패 — {r.get('fail_reason', '')}"

        lines.append(f"## {item_id}  (preset: {preset_id}, page: {page_number}) — {status}\n")

        lines.append("**원문 (long_text)**\n")
        lines.append(f"> {entry.get('long_text', '(원문을 찾지 못함)')}\n")

        original_table = entry.get("table_markdown", "")
        lines.append("**정답 표 (원본 table_markdown — long_text로 변환되기 전 원래 표, 참고용)**\n")
        lines.append(original_table if original_table.strip() else "(원본 표 없음 -- long_text가 원본 표 없이 생성된 경우)")
        lines.append("")

        lines.append("**추출된 표 (이번 파이프라인이 long_text만 보고 재구성한 결과)**\n")
        table_md = r.get("table_markdown", "")
        lines.append(table_md if table_md.strip() else "(표 없음)")
        lines.append("")

        lines.append("**복원된 장문 (round-trip 결과)**\n")
        lines.append(f"> {r.get('regenerated_long_text') or '(복원 실패)'}\n")

        hall = r.get("hallucination_check")
        omit = r.get("omission_check")
        lines.append("**검증 수치**")
        if hall:
            lines.append(
                f"- 환각 체크: coverage={hall['coverage_ratio']} "
                f"({hall['covered_values']}/{hall['total_values']}), "
                f"의심값={hall['missing_values']}"
            )
        if omit:
            lines.append(
                f"- 누락 체크: coverage={omit['coverage_ratio']} "
                f"({omit['covered_values']}/{omit['total_values']}), "
                f"사라진값={omit['missing_values']}"
            )
        lines.append("\n---\n")

    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser(
        description="chunk_orchestrator.py의 추출 결과를 round-trip으로 자기검증합니다."
    )
    parser.add_argument("--longtext", required=True, help="tables_longtext.json (원본 long_text, context)")
    parser.add_argument("--extracted", required=True, help="chunk_orchestrator.py 출력 (item_id, table_markdown)")
    parser.add_argument("--output", default="roundtrip_verify_result.json")
    parser.add_argument("--genre", choices=["informative", "narrative"], default="informative")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--hallucination-threshold", type=float, default=DEFAULT_HALLUCINATION_THRESHOLD)
    parser.add_argument("--omission-threshold", type=float, default=DEFAULT_OMISSION_THRESHOLD)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="--output과 같은 폴더의 중간 저장 파일(.jsonl)을 읽어 이미 처리된 item_id는 건너뛰고 이어서 처리",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="LLM을 새로 호출하지 않고, 기존 .jsonl만 읽어 .json/.md 리포트를 재생성한다. "
             "실행이 중간에 끊겨 최종 .json/.md가 안 만들어졌을 때 사용.",
    )
    args = parser.parse_args()

    with open(args.longtext, encoding="utf-8") as f:
        longtext_entries = {e["table_id"]: e for e in json.load(f)}
    with open(args.extracted, encoding="utf-8") as f:
        extracted_entries = json.load(f)
    extracted_entries_by_id = {e["item_id"]: e for e in extracted_entries}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")

    if args.report_only:
        if not jsonl_path.exists():
            print(f"[오류] {jsonl_path}가 없어 리포트를 재생성할 수 없습니다.")
            return
        results = _read_jsonl(jsonl_path)
        fail_count = len([r for r in results if not r.get("passed")])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        md_path = output_path.with_suffix(".md")
        md_path.write_text(
            build_markdown_report(results, longtext_entries, extracted_entries_by_id),
            encoding="utf-8",
        )
        print(f"[리포트 재생성] {jsonl_path}에서 {len(results)}건 로드 (통과 {len(results) - fail_count}, 실패 {fail_count})")
        print(f"결과 저장: {output_path}")
        print(f"미리보기용 리포트: {md_path}")
        return

    # --resume: 기존 jsonl에서 "error" 없이 끝난 item_id는 (통과/실패 여부와 무관하게)
    # 정상적으로 검증이 완료된 것으로 보고 건너뛴다. LLM 호출 자체가 죽었던
    # 항목(error 있음)만 다시 시도한다.
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

    # 재개 모드에서 이미 완료된 결과는 최종 집계에 그대로 포함시킨다.
    results: list[dict] = [r for iid, r in existing_results.items() if iid in done_ids]
    fail_count = len([r for r in results if not r.get("passed")])

    interrupted = False
    try:
        for ex in extracted_entries:
            item_id = ex["item_id"]
            entry = longtext_entries.get(item_id)
            table_markdown = ex.get("table_markdown", "")

            if entry is None:
                print(f"[경고] {item_id}: 원본 long_text를 찾지 못해 건너뜁니다.")
                continue
            if args.resume and item_id in done_ids:
                print(f"{item_id} 이미 완료됨 -> 건너뜀")
                continue

            print(f"{item_id} 검증 중...")
            try:
                result = verify_extraction(
                    entry.get("long_text", ""),
                    table_markdown,
                    context_before=entry.get("context_before", ""),
                    context_after=entry.get("context_after", ""),
                    genre=args.genre,
                    model=args.model,
                    ollama_url=args.ollama_url,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    hallucination_threshold=args.hallucination_threshold,
                    omission_threshold=args.omission_threshold,
                )
            except Exception as e:
                print(f"    [오류] {item_id} 검증 실패: {e}")
                result = {
                    "table_markdown": table_markdown,
                    "hallucination_check": None,
                    "omission_check": None,
                    "regenerated_long_text": "",
                    "passed": False,
                    "fail_reason": None,
                    "error": str(e),
                }

            result["item_id"] = item_id
            results.append(result)
            _append_jsonl(jsonl_path, result)  # <- 항목 처리 즉시 디스크에 저장 (핵심)

            if result.get("error"):
                continue

            if result["passed"]:
                print("    -> 통과")
            else:
                fail_count += 1
                print(f"    -> 실패: {result['fail_reason']}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[중단됨] Ctrl+C 감지 -- 지금까지 처리된 {len(results)}건으로 리포트를 저장합니다. "
              f"이어서 하려면 --resume으로 다시 실행하세요.")

    # 재개로 쌓였을 수 있는 중복/실패 잔여 라인을 정리하기 위해
    # 최종 결과 기준으로 jsonl을 한 번 깔끔하게 재작성한다.
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = output_path.with_suffix(".md")
    md_report = build_markdown_report(results, longtext_entries, extracted_entries_by_id)
    md_path.write_text(md_report, encoding="utf-8")

    print(f"\n=== {'검증 중단됨' if interrupted else '검증 완료'} ===")
    print(f"통과: {len(results) - fail_count}/{len(results)}, 실패: {fail_count}")
    print(f"중간 저장(재개용): {jsonl_path}")
    print(f"결과 저장: {output_path}")
    print(f"미리보기용 리포트: {md_path}")


if __name__ == "__main__":
    main()