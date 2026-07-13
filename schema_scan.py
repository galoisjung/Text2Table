"""
schema_scan.py

[A-1 단계] 스키마 제안 (extension_columns 1차 스캔)

preset_library.py 가 정의한 core_columns/few_shot/row_unit_desc를 그대로
프롬프트에 엮어서, 이 문서에서만 등장하는 extension_columns 후보를 스캔한다.

설계 원칙
---------
- LLM에게 "N개만 골라라"처럼 budget을 강제하지 않는다. 대신 각 후보가
  실제로 몇 개의 서로 다른 개체/사건에서 반복 등장했는지(recurrence)를
  정직하게 보고하게 하고, budget 적용은 코드에서 결정론적으로 한다
  (LLM의 자기절제에 기대지 않는다는, coverage check와 같은 철학).
- extension_budget_default가 None인 preset(vertical_entity, listing)은
  구조적으로 확장 열이 없으므로 LLM 호출 자체를 스킵한다 (비용 절감).
- recurrence < 2 인 후보는 "이 문서 전체에서 한 번만 언급된 디테일"로 보고
  기본적으로 채택하지 않는다 (header proliferation 방지).
- 채택된 후보가 budget의 2배를 넘으면 강제로 자르지 않고 대신
  header_proliferation_risk 플래그만 남긴다 -- 자동 판단보다 로그를 쌓아
  나중에 preset budget 자체를 조정할 근거로 쓰는 쪽을 우선한다.

사용 예
-------
    python schema_scan.py --longtext tables_longtext.json \\
        --classification preset_classification.json \\
        --output schema_scan_result.json
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
MIN_RECURRENCE_TO_ACCEPT = 2
OVERFLOW_MULTIPLIER = 2  # budget의 몇 배까지를 "위험 신호"로 볼지

# preset_id -> 프롬프트에서 사용할 "행이 가리키는 대상"의 자연스러운 명사
_ROW_UNIT_NOUN = {
    "horizontal_relational_static": "개체",
    "horizontal_relational_timeseries": "개체(각 시점의 기록)",
    "event_timeline": "사건",
}


# ─────────────────────────────────────────────────────────────
# 1. 프롬프트 구성 (few_shot을 preset에서 그대로 가져옴)
# ─────────────────────────────────────────────────────────────
def _format_few_shot(preset: Preset) -> str:
    if not preset.few_shot:
        return "(참고 예시 없음)"
    lines = []
    for ex in preset.few_shot:
        lines.append(f"- 원문: \"{ex['text']}\"")
        lines.append(f"  -> 행: {ex['row']}")
    return "\n".join(lines)


def build_schema_scan_prompt(preset_id: str, text: str) -> str:
    preset = PRESETS.get(preset_id)
    if preset is None:
        raise ValueError(f"알 수 없는 preset_id: {preset_id}")
    if preset.extension_budget_default is None:
        raise ValueError(
            f"'{preset_id}'는 확장 열(extension_columns)이 구조적으로 없는 preset입니다. "
            "스키마 스캔이 필요 없으니 scan_schema()가 아니라 바로 추출 단계로 넘어가세요."
        )

    core_names = ", ".join(c.name for c in preset.core_columns)
    unit_noun = _ROW_UNIT_NOUN.get(preset_id, "개체")
    few_shot_block = _format_few_shot(preset)

    return f"""당신은 텍스트를 표로 정리하기 전에, 표에 추가로 필요한 "속성 컬럼" 후보를
찾아내는 분석가입니다. 아직 표를 만들지 마세요 — 컬럼 후보만 찾으면 됩니다.

[이 텍스트가 정리될 표의 구조]
- 행의 의미: {preset.row_unit_desc}
- 이미 확정된 기본 컬럼(다시 제안하지 마세요): {core_names}

[참고 예시 — 같은 유형의 다른 문서가 실제로 이렇게 표로 변환된 사례입니다]
{few_shot_block}

[규칙]
1. 여러 {unit_noun}에 걸쳐 "공통적으로, 반복적으로" 언급되는 속성만 후보로 제안하세요.
   특정 {unit_noun} 하나에서만 등장한 디테일은 컬럼이 아니라 그 항목의 부가 서술일
   뿐이므로 후보에서 제외하세요.
2. 각 후보마다, 서로 다른 {unit_noun} 몇 개에서 실제로 등장했는지를 recurrence
   정수값으로 정직하게 세어 보고하세요. 과대 보고하면 표에 빈 칸이 늘어나고,
   과소 보고하면 정보가 누락됩니다 — 실제로 센 값만 보고하세요.
3. 표현이 다르지만 같은 의미인 속성(예: "본사주소"와 "본점소재지")은 하나의
   후보로 통합하고, 더 자연스러운 쪽을 대표 이름으로 선택하세요.
4. 컬럼 이름은 짧은 명사형으로, 원문 표현을 과도하게 의역하지 마세요.
5. example_values에는 실제로 텍스트에 등장한 값을 1~3개 그대로 인용하세요.

아래 JSON 형식으로만 답하세요. 다른 설명이나 markdown 코드펜스는 쓰지 마세요.
{{"candidates": [{{"name": "...", "recurrence": 정수, "example_values": ["...", "..."]}}]}}

[텍스트]
{text}
"""


# ─────────────────────────────────────────────────────────────
# 2. Ollama 호출 (JSON 강제 + 폴백, 재시도) -- 이전 단계들과 동일한 패턴
# ─────────────────────────────────────────────────────────────
def _extract_json_object(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"JSON으로 파싱할 수 없는 응답입니다: {raw[:200]}...")


def call_ollama_scan(
    prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "").strip()
            if not raw_text:
                raise ValueError("Ollama가 빈 응답을 반환했습니다.")
            return _extract_json_object(raw_text)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as e:
            last_error = e
            wait = 2 ** (attempt - 1)
            print(f"    [재시도 {attempt}/{max_retries}] {e} -> {wait}초 대기 후 재시도")
            time.sleep(wait)

    raise RuntimeError(f"Ollama 스캔 호출이 {max_retries}회 모두 실패했습니다: {last_error}")


# ─────────────────────────────────────────────────────────────
# 3. budget 정책 적용 (결정론적 후처리, LLM에 맡기지 않음)
# ─────────────────────────────────────────────────────────────
def apply_budget_policy(candidates: list[dict], extension_budget_default: int) -> dict:
    valid = [
        c for c in candidates
        if isinstance(c.get("name"), str) and c["name"].strip()
        and isinstance(c.get("recurrence"), (int, float))
    ]

    accepted = sorted(
        (c for c in valid if c["recurrence"] >= MIN_RECURRENCE_TO_ACCEPT),
        key=lambda c: -c["recurrence"],
    )
    rejected_low_recurrence = [c for c in valid if c["recurrence"] < MIN_RECURRENCE_TO_ACCEPT]

    overflow_threshold = extension_budget_default * OVERFLOW_MULTIPLIER
    header_proliferation_risk = len(accepted) > overflow_threshold

    return {
        "accepted_columns": accepted,
        "rejected_low_recurrence": rejected_low_recurrence,
        "budget_default": extension_budget_default,
        "overflow_threshold": overflow_threshold,
        "header_proliferation_risk": header_proliferation_risk,
    }


# ─────────────────────────────────────────────────────────────
# 4. 오케스트레이션
# ─────────────────────────────────────────────────────────────
def scan_schema(
    preset_id: str,
    text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    preset = PRESETS.get(preset_id)
    if preset is None:
        return {"skipped": True, "reason": f"알 수 없는 preset_id: {preset_id}"}

    if preset.extension_budget_default is None:
        return {
            "skipped": True,
            "reason": f"'{preset_id}'는 확장 열이 구조적으로 없어 스캔이 불필요함",
        }

    prompt = build_schema_scan_prompt(preset_id, text)
    raw_result = call_ollama_scan(
        prompt, model=model, ollama_url=ollama_url, timeout=timeout, max_retries=max_retries
    )
    candidates = raw_result.get("candidates", [])
    if not isinstance(candidates, list):
        return {"skipped": False, "error": "candidates 필드가 리스트가 아님", "raw": raw_result}

    policy_result = apply_budget_policy(candidates, preset.extension_budget_default)
    return {"skipped": False, "preset_id": preset_id, **policy_result}


# ─────────────────────────────────────────────────────────────
# 5. CLI: preset_classifier.py 결과와 조인해서 실행
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="preset_classifier.py의 분류 결과를 바탕으로 A-1 스키마 스캔을 실행합니다."
    )
    parser.add_argument("--longtext", required=True, help="tables_longtext.json (table_id, long_text 포함)")
    parser.add_argument("--classification", required=True, help="preset_classifier.py의 출력 (item_id, preset_id)")
    parser.add_argument("--output", default="schema_scan_result.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    with open(args.longtext, encoding="utf-8") as f:
        longtext_entries = {e["table_id"]: e for e in json.load(f)}
    with open(args.classification, encoding="utf-8") as f:
        classifications = json.load(f)

    results = []
    for c in classifications:
        item_id = c["item_id"]
        preset_id = c.get("preset_id")
        entry = longtext_entries.get(item_id)

        if entry is None:
            print(f"[경고] {item_id}: long_text를 찾지 못해 건너뜁니다.")
            continue
        if not preset_id:
            print(f"[건너뜀] {item_id}: 폴백 항목이라 preset이 없음 (자유 스키마 경로 필요)")
            continue

        print(f"{item_id} ({preset_id}) 스캔 중...")
        result = scan_schema(
            preset_id,
            entry.get("long_text", ""),
            model=args.model,
            ollama_url=args.ollama_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        result["item_id"] = item_id
        results.append(result)

        if result.get("skipped"):
            print(f"    -> 스킵 ({result['reason']})")
        elif "accepted_columns" in result:
            names = [c["name"] for c in result["accepted_columns"]]
            print(f"    -> 채택: {names}")
            if result["header_proliferation_risk"]:
                print(f"    [경고] header_proliferation_risk: 채택 {len(names)}개 > 임계 {result['overflow_threshold']}개")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
