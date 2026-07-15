"""
subject_split.py

[A-2.5 단계] 단일 주제(single_subject) preset 내부의 "주제 섞임" 감지 + 분리

vertical_entity/listing처럼 "행 = 속성-값 쌍 1개"로 하나의 주제를 가정하는
preset은, 실제로는 서로 다른 두 개 이상의 주제(예: "책 정보"와 "등장인물 정보")가
한 표에 억지로 눌려 담기는 경우가 있다. preset(표의 "모양")은 맞게 골랐어도,
"이 표가 정말 하나의 주제만 다루는가"는 별개의 문제다.

기존 validate_extraction()의 row_unit_mismatch_warning은 정반대 방향의 실패
(row_key 값이 전부 똑같음)만 잡는다. 값이 다양한데 그 다양성 자체가 사실은
두 주제가 섞여서 생긴 것인 경우는 잡지 못한다 -- 이 모듈이 그 빈틈을 메운다.

설계 원칙 (schema_scan.py의 recurrence 패턴과 동일):
- LLM에게 "분리해야 하나?"를 직접 묻지 않는다. 대신 row_key(속성명/항목)
  목록을 "같은 대상을 설명하는 것끼리" 그룹으로만 묶어 보고하게 하고,
  실제 분리 여부(그룹 크기 임계치 등)는 코드가 결정론적으로 판단한다.
- event_timeline/horizontal_relational_*는 대상이 아니다. single_subject
  전제 위에 서 있고, row_key가 "이 표가 다루는 대상의 속성 이름"인
  preset(vertical_entity, listing)에서만 의미 있는 체크다. event_timeline은
  row_key가 "시점"이라 여러 행이 있는 게 애초에 정상 동작이므로 대상이 아니다.
"""

from __future__ import annotations

import json
import re
import time

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

# row_key 종류가 이보다 적으면 "주제가 섞였다"고 판단할 근거가 부족하다고 보고 스킵
MIN_ROWS_FOR_SPLIT_CHECK = 4
# 이보다 작은 그룹은 노이즈(진짜 별개 주제가 아니라 그냥 속성 몇 개)로 보고
# 분리 여부 판단에서 제외한다
MIN_GROUP_SIZE = 2

# 이 preset들만 "하나의 주제"라는 전제 위에 서 있다 (single_subject + 고정 라벨행).
APPLICABLE_PRESETS = {"vertical_entity", "listing"}

_NORMALIZE_RE = re.compile(r"[\s\(\)\[\]{}·,\.\-‑–—]")


def normalize_item(v: str) -> str:
    return _NORMALIZE_RE.sub("", (v or "").strip())


# ─────────────────────────────────────────────────────────────
# 1. 프롬프트 구성
# ─────────────────────────────────────────────────────────────
def build_grouping_prompt(row_key_items: list[str]) -> str:
    items_block = "\n".join(f"- {item}" for item in row_key_items)
    return f"""당신은 표의 행 라벨들을 분석하는 분석가입니다. 아래는 하나의 표 안에 있는
행 라벨(속성명 또는 항목명) 목록입니다. 이 표는 원래 "하나의 대상"을 설명한다고
가정하고 만들어졌지만, 실제로는 서로 다른 대상을 설명하는 라벨들이 섞여 있을 수
있습니다.

[규칙]
1. 라벨들을 "같은 대상/주제를 설명하는 것끼리" 그룹으로 묶으세요.
2. 정말로 하나의 대상만 다루고 있다면 그룹은 1개여야 합니다. 억지로 쪼개지 마세요.
3. 서로 다른 대상(예: 책 자체의 정보 vs 등장인물 개별 정보)을 설명하는 라벨이
   섞여 있다면, 그 기준으로 그룹을 나누세요.
4. 각 그룹에 그 대상을 짧게 설명하는 label을 붙이세요.
5. 모든 라벨은 정확히 하나의 그룹에만 속해야 합니다 (누락/중복 금지).

[행 라벨 목록]
{items_block}

아래 JSON 형식으로만 답하세요. 다른 설명이나 markdown 코드펜스는 쓰지 마세요.
{{"groups": [{{"label": "...", "items": ["...", "..."]}}]}}
"""


# ─────────────────────────────────────────────────────────────
# 2. Ollama 호출 (다른 A-1/A-2 단계와 동일한 재시도 패턴)
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


def call_ollama_group(
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
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 500 <= status < 600:
                last_error = e
                wait = 2 ** (attempt - 1)
                print(f"    [재시도 {attempt}/{max_retries}] HTTP {status} -> {wait}초 대기 후 재시도")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Ollama 그룹핑 호출이 {max_retries}회 모두 실패했습니다: {last_error}")


# ─────────────────────────────────────────────────────────────
# 3. 결정론적 분리 정책 (LLM 자기절제를 믿지 않는다 원칙)
# ─────────────────────────────────────────────────────────────
def apply_split_policy(groups: list[dict], min_group_size: int = MIN_GROUP_SIZE) -> dict:
    valid = [
        g for g in groups
        if isinstance(g.get("items"), list) and len(g["items"]) >= min_group_size
        and isinstance(g.get("label"), str) and g["label"].strip()
    ]
    small = [g for g in groups if g not in valid]

    return {
        "mixed": len(valid) >= 2,
        "accepted_groups": valid,
        "small_groups": small,
    }


# ─────────────────────────────────────────────────────────────
# 4. 감지 오케스트레이션
# ─────────────────────────────────────────────────────────────
def detect_subject_split(
    preset_id: str,
    row_key_items: list[str],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """
    row_key_items: 표의 row_key 컬럼(vertical_entity의 "속성명", listing의 "항목")
    값 목록. 내부에서 중복 제거 후 판단한다.
    """
    if preset_id not in APPLICABLE_PRESETS:
        return {
            "skipped": True,
            "reason": f"'{preset_id}'는 주제 섞임 체크 대상이 아님 (single_subject 고정 라벨 전제가 없음)",
        }

    unique_items = list(dict.fromkeys(i for i in row_key_items if i and i.strip()))
    if len(unique_items) < MIN_ROWS_FOR_SPLIT_CHECK:
        return {
            "skipped": True,
            "reason": f"row_key 종류가 {len(unique_items)}개뿐이라 판단 근거 부족 "
                      f"(최소 {MIN_ROWS_FOR_SPLIT_CHECK}개 필요)",
        }

    prompt = build_grouping_prompt(unique_items)
    raw_result = call_ollama_group(
        prompt, model=model, ollama_url=ollama_url, timeout=timeout, max_retries=max_retries
    )
    groups = raw_result.get("groups", [])
    if not isinstance(groups, list) or not groups:
        return {"skipped": False, "error": "groups 필드가 비어있거나 리스트가 아님", "raw": raw_result}

    policy_result = apply_split_policy(groups)
    return {"skipped": False, "preset_id": preset_id, **policy_result}


# ─────────────────────────────────────────────────────────────
# 5. 실제 표 분리 (row 단위)
# ─────────────────────────────────────────────────────────────
def split_rows_by_group(
    header: list[str],
    rows: list[dict],
    row_key_column: str,
    accepted_groups: list[dict],
) -> list[dict]:
    """
    accepted_groups 기준으로 rows를 나눠
    [{"label": ..., "header": ..., "rows": [...]}, ...] 형태로 반환한다.

    어느 그룹에도 안 걸리는 행은 데이터를 조용히 버리지 않고 "미분류" 그룹으로
    모아서 남긴다 (chunk_orchestrator.merge_rows_with_alias의 conflicts 기록과
    같은 철학 -- 애매한 건 자동으로 버리지 않고 사람이 보이게 남긴다).
    """
    item_to_group: dict[str, int] = {}
    for gi, g in enumerate(accepted_groups):
        for item in g["items"]:
            item_to_group[normalize_item(item)] = gi

    buckets: list[list[dict]] = [[] for _ in accepted_groups]
    unclassified: list[dict] = []

    for row in rows:
        key = normalize_item(row.get(row_key_column, ""))
        gi = item_to_group.get(key)
        if gi is None:
            unclassified.append(row)
        else:
            buckets[gi].append(row)

    result = [
        {"label": g["label"], "header": header, "rows": bucket}
        for g, bucket in zip(accepted_groups, buckets)
        if bucket
    ]
    if unclassified:
        result.append({"label": "미분류", "header": header, "rows": unclassified})

    return result
