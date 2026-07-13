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
# 3. CLI: D의 출력과 원본 long_text를 조인해서 실행
# ─────────────────────────────────────────────────────────────
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
    args = parser.parse_args()

    with open(args.longtext, encoding="utf-8") as f:
        longtext_entries = {e["table_id"]: e for e in json.load(f)}
    with open(args.extracted, encoding="utf-8") as f:
        extracted_entries = json.load(f)

    results = []
    fail_count = 0

    for ex in extracted_entries:
        item_id = ex["item_id"]
        entry = longtext_entries.get(item_id)
        table_markdown = ex.get("table_markdown", "")

        if entry is None:
            print(f"[경고] {item_id}: 원본 long_text를 찾지 못해 건너뜁니다.")
            continue

        print(f"{item_id} 검증 중...")
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
        result["item_id"] = item_id
        results.append(result)

        if result["passed"]:
            print("    -> 통과")
        else:
            fail_count += 1
            print(f"    -> 실패: {result['fail_reason']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 검증 완료 ===")
    print(f"통과: {len(results) - fail_count}/{len(results)}, 실패: {fail_count}")
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
