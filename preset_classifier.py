"""
preset_classifier.py

[C 단계] 장르(축) 분류 + 프리셋 선택

임의의 텍스트(long_text 등)를 받아 로컬 Ollama LLM으로 세 가지 축
(focus, time_structure, attribute_count)을 분류하고, preset_library의
결정 트리로 preset_id를 확정한다.

여기서 나오는 결과는 "이 텍스트는 어떤 모양의 표가 되어야 하는가"까지이며,
실제 컬럼명/행 데이터를 채우는 건 다음 단계(A: 스키마 제안)의 몫이다.

신뢰도(confidence)가 낮거나 축이 명확하지 않으면 preset_id=None으로
반환하고 fallback_reason을 채운다 -- 이 경우 자유 스키마(A 단독) 경로로
넘기라는 신호다. 프리셋이 억지로 맞지 않는 문서를 강제 분류하지 않는 것이
프리셋 오분류로 인한 연쇄 오류보다 낫다는 설계 원칙을 그대로 반영한다.

사용 예
-------
    python preset_classifier.py --input tables_longtext.json --field long_text
    python preset_classifier.py --input some_free_text.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

from preset_library import PRESETS, select_preset

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

_VALID_FOCUS = {"single_subject", "multi_entity"}
_VALID_TIME = {"timeseries", "non_timeseries"}
_VALID_ATTR = {"single_attribute", "multi_attribute", "not_applicable"}


# ─────────────────────────────────────────────────────────────
# 1. 분류 프롬프트
# ─────────────────────────────────────────────────────────────
def build_classification_prompt(text: str) -> str:
    return f"""당신은 임의의 한국어 텍스트를 표로 정리하기 위한 사전 분석가입니다.
아래 텍스트를 읽고, 이 텍스트를 표로 만든다면 표의 "모양"이 어떻게 되어야
하는지 판단하는 세 가지 질문에 답하세요. 텍스트에 실제 표가 있었는지는
중요하지 않습니다 — 순수한 서술문이라도 판단하세요.

[질문 1] focus: 이 텍스트가 여러 개체(사람/회사/항목 등)를 나열하는가,
아니면 하나의 주제나 개체에 대해서만 이야기하는가?
- "multi_entity": 서로 구분되는 여러 개체가 나열됨
- "single_subject": 하나의 주제/개체에 대한 이야기뿐

[질문 2] time_structure: 내용이 날짜나 사건 순서 등 시간 흐름을 따라 전개되는가?
- "timeseries": 예, 시간순으로 의미 있게 정렬됨
- "non_timeseries": 아니오, 시간 순서가 중요하지 않음

[질문 3] attribute_count: (focus가 multi_entity인 경우만 판단)
각 개체에 대해 언급되는 속성이 실질적으로 하나뿐인가, 여러 개인가?
- "single_attribute": 개체마다 속성이 사실상 1개 (예: 항목별 결과 하나)
- "multi_attribute": 개체마다 여러 속성이 함께 언급됨
- focus가 single_subject라면 "not_applicable"

아래 JSON 형식으로만 답하세요. 다른 설명이나 markdown 코드펜스는 쓰지 마세요.
{{"focus": "...", "time_structure": "...", "attribute_count": "...", "confidence": 0.0에서 1.0 사이 숫자, "reasoning": "판단 근거 한 문장"}}

[텍스트]
{text}
"""


# ─────────────────────────────────────────────────────────────
# 2. Ollama 호출 (JSON 강제 + 다단계 폴백, 재시도)
#    -> text_to_table_pipeline.py 에서 이미 검증된 패턴 재사용
# ─────────────────────────────────────────────────────────────
def _extract_json_object(raw: str) -> dict:
    """
    1) 그대로 json.loads 시도
    2) 실패하면 가장 바깥쪽 {...} 블록을 정규식으로 추출해 재시도
    """
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


def call_ollama_classify(
    prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",  # 가능한 백엔드에서는 JSON 출력을 강제
        "stream": False,
    }
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "").strip()
            if not raw_text:
                raise ValueError("Ollama가 빈 응답을 반환했습니다.")
            return _extract_json_object(raw_text)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status < 500:
                # 4xx(요청 자체가 잘못됨)는 재시도해도 결과가 같으므로 즉시 실패시킨다.
                raise
            # 5xx(502/503/504 등)는 업스트림(예: -cloud 모델의 원격 추론 백엔드)의
            # 일시적 문제인 경우가 많으므로 재시도 대상에 포함한다.
            last_error = e
            wait = 2 ** (attempt - 1)
            print(f"    [재시도 {attempt}/{max_retries}] HTTP {status} 오류 -> {wait}초 대기 후 재시도")
            time.sleep(wait)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as e:
            last_error = e
            wait = 2 ** (attempt - 1)
            print(f"    [재시도 {attempt}/{max_retries}] {e} -> {wait}초 대기 후 재시도")
            time.sleep(wait)

    raise RuntimeError(f"Ollama 분류 호출이 {max_retries}회 모두 실패했습니다: {last_error}")


# ─────────────────────────────────────────────────────────────
# 3. 축 검증 + preset 결정
# ─────────────────────────────────────────────────────────────
def _validate_axes(axes: dict) -> tuple[bool, str]:
    focus = axes.get("focus")
    time_structure = axes.get("time_structure")
    attribute_count = axes.get("attribute_count")

    if focus not in _VALID_FOCUS:
        return False, f"focus 값이 유효하지 않음: {focus!r}"
    if time_structure not in _VALID_TIME:
        return False, f"time_structure 값이 유효하지 않음: {time_structure!r}"
    if attribute_count not in _VALID_ATTR:
        return False, f"attribute_count 값이 유효하지 않음: {attribute_count!r}"
    return True, ""


def classify_and_select_preset(
    text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    텍스트 -> (축 분류 -> preset 결정) 전체 파이프라인.
    반환값에는 항상 axes(원본 분류 결과)와 preset_id(또는 None)가 담긴다.
    preset_id가 None이면 fallback_reason에 사유가 남는다.
    """
    if not text or not text.strip():
        return {
            "axes": None,
            "preset_id": None,
            "fallback_reason": "빈 텍스트",
        }

    prompt = build_classification_prompt(text)
    axes = call_ollama_classify(
        prompt, model=model, ollama_url=ollama_url, timeout=timeout, max_retries=max_retries
    )

    valid, reason = _validate_axes(axes)
    if not valid:
        return {"axes": axes, "preset_id": None, "fallback_reason": reason}

    confidence = axes.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < confidence_threshold:
        return {
            "axes": axes,
            "preset_id": None,
            "fallback_reason": f"신뢰도 부족 ({confidence:.2f} < {confidence_threshold})",
        }

    preset_id = select_preset(
        axes["focus"], axes["time_structure"], axes.get("attribute_count", "not_applicable")
    )

    if preset_id is None:
        return {
            "axes": axes,
            "preset_id": None,
            "fallback_reason": "결정 트리에서 preset을 특정하지 못함 (축 조합 확인 필요)",
        }

    return {"axes": axes, "preset_id": preset_id, "fallback_reason": None}


# ─────────────────────────────────────────────────────────────
# 3.5. 중간 결과 저장/로드 (JSONL, 처리 즉시 1건씩 append)
# ─────────────────────────────────────────────────────────────
def _append_jsonl(path: Path, obj: dict) -> None:
    """
    결과 1건을 즉시 파일에 append하고 flush+fsync한다.
    루프 도중 프로세스가 죽거나(502 반복, Ctrl+C, OOM 등) 해도
    이미 append된 항목들은 디스크에 남아 있다.
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
# 4. 메인 (CLI)
# ─────────────────────────────────────────────────────────────
_TEXT_INPUT_SUFFIXES = {".json", ".txt", ".md"}


def _load_texts_from_file(input_path: Path, field: str) -> list[dict]:
    """
    .json이면 [{field: "..."} ...] 구조로 보고 각 항목에서 field를 뽑는다.
    .txt/.md 등이면 파일 전체를 텍스트 하나로 취급한다.
    """
    if input_path.suffix.lower() == ".json":
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for i, entry in enumerate(data):
            text = entry.get(field, "")
            items.append({"item_id": entry.get("table_id", f"item_{i}"), "text": text})
        return items
    else:
        text = input_path.read_text(encoding="utf-8")
        return [{"item_id": input_path.stem, "text": text}]


def _load_texts(input_path: Path, field: str, recursive: bool = True) -> list[dict]:
    """
    input_path가 파일이면 기존과 동일하게 처리한다.
    input_path가 폴더면 그 안의 .json/.txt/.md 파일들을 모두 찾아 각각
    _load_texts_from_file로 읽은 뒤 하나의 리스트로 합친다.
    폴더 입력일 때는 파일명 충돌을 막기 위해 item_id 앞에 "파일stem__"을 붙인다.
    """
    if input_path.is_file():
        return _load_texts_from_file(input_path, field)

    if not input_path.is_dir():
        raise FileNotFoundError(f"입력 경로가 존재하지 않습니다: {input_path}")

    glob_fn = input_path.rglob if recursive else input_path.glob
    files = sorted(
        p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in _TEXT_INPUT_SUFFIXES
    )
    if not files:
        raise ValueError(
            f"입력 폴더에 처리 가능한 파일(.json/.txt/.md)이 없습니다: {input_path}"
        )

    items: list[dict] = []
    for fp in files:
        try:
            file_items = _load_texts_from_file(fp, field)
        except Exception as e:
            print(f"    [건너뜀] {fp}: 읽기 실패 ({e})")
            continue
        for item in file_items:
            item["item_id"] = f"{fp.stem}__{item['item_id']}"
            items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser(description="텍스트를 축 분류하여 preset을 선택합니다.")
    parser.add_argument(
        "--input",
        required=True,
        help="입력 파일(.json 배열 또는 .txt/.md 단일 텍스트) 또는 그런 파일들이 들어있는 폴더 경로",
    )
    parser.add_argument("--field", default="long_text", help=".json 입력일 때 텍스트가 담긴 필드명")
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        default=True,
        help="폴더 입력일 때 하위 폴더까지 탐색하지 않음 (기본값: 하위 폴더까지 탐색)",
    )
    parser.add_argument("--output", default="preset_classification.json", help="결과 저장 경로")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="--output과 같은 폴더의 중간 저장 파일(.jsonl)을 읽어 이미 처리된 item_id는 건너뛰고 이어서 처리",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        print(f"폴더 입력 감지: {input_path} (recursive={args.recursive})")
    items = _load_texts(input_path, args.field, recursive=args.recursive)
    print(f"총 {len(items)}개 항목을 분류합니다.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")

    # --resume: 기존 jsonl에서 "오류(예외)"로 끝나지 않은 item_id는 이미 완료된 것으로
    # 보고 건너뛴다. 신뢰도 부족 등으로 preset_id=None인 정상 폴백 결과도 완료로 취급한다.
    existing_results: dict[str, dict] = {}
    if args.resume and jsonl_path.exists():
        for r in _read_jsonl(jsonl_path):
            iid = r.get("item_id")
            if iid is not None:
                existing_results[iid] = r  # 같은 item_id가 여러 번 있으면 마지막 것으로 덮어씀
        done_ids = {
            iid
            for iid, r in existing_results.items()
            if not str(r.get("fallback_reason", "")).startswith("오류:")
        }
        print(f"[재개 모드] 기존 결과 {len(existing_results)}건 로드, 완료된 {len(done_ids)}건은 건너뜁니다.")
    else:
        done_ids = set()
        if jsonl_path.exists():
            print(f"[주의] --resume 없이 실행되어 기존 {jsonl_path.name}을 새로 덮어씁니다.")
            jsonl_path.unlink()

    # 재개 모드에서 이미 완료된 결과는 최종 집계에 그대로 포함시킨다.
    results: list[dict] = [r for iid, r in existing_results.items() if iid in done_ids]
    preset_counts: dict[str, int] = {}
    fallback_count = 0
    for r in results:
        if r.get("preset_id"):
            preset_counts[r["preset_id"]] = preset_counts.get(r["preset_id"], 0) + 1
        else:
            fallback_count += 1

    for i, item in enumerate(items, 1):
        if args.resume and item["item_id"] in done_ids:
            print(f"[{i}/{len(items)}] {item['item_id']} 이미 완료됨 -> 건너뜀")
            continue

        print(f"[{i}/{len(items)}] {item['item_id']} 분류 중...")
        try:
            result = classify_and_select_preset(
                item["text"],
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                confidence_threshold=args.confidence_threshold,
            )
        except Exception as e:
            result = {"axes": None, "preset_id": None, "fallback_reason": f"오류: {e}"}

        result["item_id"] = item["item_id"]
        results.append(result)
        _append_jsonl(jsonl_path, result)  # <- 항목 처리 즉시 디스크에 저장 (핵심)

        if result["preset_id"]:
            preset_counts[result["preset_id"]] = preset_counts.get(result["preset_id"], 0) + 1
            print(f"    -> {result['preset_id']}")
        else:
            fallback_count += 1
            print(f"    -> 폴백 (사유: {result['fallback_reason']})")

    # 재개로 쌓였을 수 있는 중복/실패 잔여 라인을 정리하기 위해
    # 최종 결과 기준으로 jsonl을 한 번 깔끔하게 재작성한다.
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 분류 완료 ===")
    for preset_id, count in sorted(preset_counts.items(), key=lambda x: -x[1]):
        print(f"  {preset_id}: {count}건")
    print(f"  폴백(자유 스키마 필요): {fallback_count}건")
    print(f"중간 저장(재개용): {jsonl_path}")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()