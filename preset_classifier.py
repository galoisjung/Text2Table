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
DEFAULT_CLASSIFICATION_SAMPLE_CHARS = 16000  # 대략 8천 토큰 남짓. 축 분류는 구조/장르
                                              # 판단만 하면 되므로 전체 문서를 다 넣을
                                              # 필요가 없다 -- 텍스트가 길면 400(payload
                                              # 초과) 오류가 나므로 앞부분 샘플만 사용.

_VALID_FOCUS = {"single_subject", "multi_entity"}
_VALID_TIME = {"timeseries", "non_timeseries"}
_VALID_ATTR = {"single_attribute", "multi_attribute", "not_applicable"}


# ─────────────────────────────────────────────────────────────
# 1. 분류 프롬프트
# ─────────────────────────────────────────────────────────────
def build_classification_prompt(text: str, avoid_note: str = "") -> str:
    avoid_block = f"\n{avoid_note}\n" if avoid_note else ""
    return f"""당신은 임의의 한국어 텍스트를 표로 정리하기 위한 사전 분석가입니다.
아래 텍스트를 읽고, 이 텍스트를 표로 만든다면 표의 "모양"이 어떻게 되어야
하는지 판단하는 세 가지 질문에 답하세요. 텍스트에 실제 표가 있었는지는
중요하지 않습니다 — 순수한 서술문이라도 판단하세요. 아래 텍스트는 원문
전체가 아니라 앞부분 발췌일 수 있습니다 — 그래도 구조적 패턴(개체 나열
여부, 시간 흐름 여부)은 충분히 판단 가능하니 발췌라는 이유로 판단을
주저하지 마세요.

[질문 1] focus: 이 텍스트가 여러 개체(사람/회사/항목 등)를 나열하는가,
아니면 하나의 주제나 개체에 대해서만 이야기하는가?
- "multi_entity": 서로 구분되는 여러 개체가 나열됨 (예: 회사 A, B, C 각각의 정보)
- "single_subject": 하나의 주제/개체에 대한 이야기뿐

주의: 속성(레이블)이 여러 개 나열된다고 해서 자동으로 multi_entity는 아닙니다.
예를 들어 "계약일자는 X, 담당자는 Y, 위치는 Z이다"처럼 여러 속성이 나열돼도
그 속성들이 전부 하나의 대상(이 계약 자체, 이 문서 자체)을 설명하고 있다면
여전히 single_subject입니다. "계약일자"나 "담당자" 같은 속성 이름(레이블) 자체를
개체로 착각하지 마세요 — 실제로 이름이 다른 대상(서로 다른 회사명, 사람 이름,
날짜별로 구분되는 사건 등)이 여러 개 나열될 때만 multi_entity입니다.

[질문 2] time_structure: 내용이 날짜나 사건 순서 등 시간 흐름을 따라 전개되는가?
- "timeseries": 예, 시간순으로 의미 있게 정렬됨
- "non_timeseries": 아니오, 시간 순서가 중요하지 않음

주의: 텍스트에 날짜가 여러 번 언급된다고 자동으로 timeseries는 아닙니다.
"계약일은 2024년 6월 19일이고, 제출일은 2024년 6월 21일이다"처럼 하나의
대상에 관한 서로 다른 종류의 날짜 속성(계약일/기간/제출일 등)이 나열된
것이라면, 그 날짜들은 서로 다른 사건을 시간순으로 나타내는 게 아니라 그
대상의 부속 정보일 뿐이므로 non_timeseries입니다. timeseries는 "행"에
해당하는 여러 사건·거래·기록이 날짜마다 반복되며 나열될 때만 해당합니다.

[질문 3] attribute_count: (focus가 multi_entity인 경우만 판단)
각 개체에 대해 언급되는 속성이 실질적으로 하나뿐인가, 여러 개인가?
- "single_attribute": 개체마다 속성이 사실상 1개 (예: 항목별 결과 하나)
- "multi_attribute": 개체마다 여러 속성이 함께 언급됨
- focus가 single_subject라면 "not_applicable"

[참고 예시 — 헷갈리기 쉬운 두 경우를 대조]
- "평가계약일자는 2024년 6월 19일이며, 평가기간은 6월 19일~20일로 설정되었고,
  제출일자는 6월 21일이다. 담당자는 홍길동이다."
  -> focus: single_subject (계약일/기간/제출일/담당자 전부 이 계약 하나를
  설명하는 속성일 뿐, 서로 다른 개체가 아님), time_structure: non_timeseries
  (날짜들은 사건의 시간순 나열이 아니라 한 계약의 부속 정보)
- "2024년 6월 17일에 A사는 100주를 거래했고, 2024년 6월 11일에 B사는
  200주를 거래했다."
  -> focus: multi_entity (A사 거래, B사 거래라는 서로 다른 사건이 나열됨),
  time_structure: timeseries (날짜마다 다른 사건이 반복되며 나열됨)
- "해리는 생일 아침 눈을 떴다. 잠시 후 도비가 나타나 경고를 전했고, 저녁이
  되자 위즐리 가족이 그를 데리러 왔다."
  -> focus: single_subject (해리라는 한 이야기의 흐름), time_structure:
  timeseries (사건이 "잠시 후", "저녁이 되자"처럼 시간 순서로 전개됨 --
  날짜가 명시적 숫자로 안 나와도 사건이 순서대로 이어지면 timeseries다).
  이런 경우는 책 제목·저자 같은 "책 메타데이터"(vertical_entity)와 구분해야
  한다 -- 실제 이야기 전개 자체를 표로 만들 때는 event_timeline이 맞다.
{avoid_block}
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
    sample_chars: int = DEFAULT_CLASSIFICATION_SAMPLE_CHARS,
    avoid_note: str = "",
) -> dict:
    """
    텍스트 -> (축 분류 -> preset 결정) 전체 파이프라인.
    반환값에는 항상 axes(원본 분류 결과)와 preset_id(또는 None)가 담긴다.
    preset_id가 None이면 fallback_reason에 사유가 남는다.

    텍스트가 sample_chars보다 길면 앞부분만 잘라서 분류 프롬프트에 쓴다.
    구조/장르 판단은 문서 전체를 다 볼 필요가 없고, 다 넣으면 -cloud 모델
    게이트웨이가 요청 자체를 400으로 거부하는 경우가 있어 이를 방지한다.

    avoid_note가 있으면(여러 표를 반복 추출하는 상황) 프롬프트에 그대로
    삽입되어, 이미 사용한 관점과 다른 구조를 찾도록 유도한다.
    """
    if not text or not text.strip():
        return {
            "axes": None,
            "preset_id": None,
            "fallback_reason": "빈 텍스트",
        }

    sample_text = text if len(text) <= sample_chars else text[:sample_chars]
    prompt = build_classification_prompt(sample_text, avoid_note=avoid_note)
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
# 4. 메인 (CLI)
# ─────────────────────────────────────────────────────────────
def _load_texts(input_path: Path, field: str) -> list[dict]:
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


def main():
    parser = argparse.ArgumentParser(description="텍스트를 축 분류하여 preset을 선택합니다.")
    parser.add_argument("--input", required=True, help="입력 파일 (.json 배열 또는 .txt/.md 단일 텍스트)")
    parser.add_argument("--field", default="long_text", help=".json 입력일 때 텍스트가 담긴 필드명")
    parser.add_argument("--output", default="preset_classification.json", help="결과 저장 경로")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--classification-sample-chars", type=int, default=DEFAULT_CLASSIFICATION_SAMPLE_CHARS,
        help="분류 프롬프트에 쓸 앞부분 문자 수 상한. 이보다 길면 잘라서 사용 "
             "(전체를 다 넣으면 -cloud 모델에서 400 오류가 날 수 있음)."
    )
    args = parser.parse_args()

    items = _load_texts(Path(args.input), args.field)
    print(f"총 {len(items)}개 항목을 분류합니다.")

    results = []
    preset_counts: dict[str, int] = {}
    fallback_count = 0

    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['item_id']} 분류 중...")
        try:
            result = classify_and_select_preset(
                item["text"],
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                confidence_threshold=args.confidence_threshold,
                sample_chars=args.classification_sample_chars,
            )
        except Exception as e:
            result = {"axes": None, "preset_id": None, "fallback_reason": f"오류: {e}"}

        result["item_id"] = item["item_id"]
        results.append(result)

        if result["preset_id"]:
            preset_counts[result["preset_id"]] = preset_counts.get(result["preset_id"], 0) + 1
            print(f"    -> {result['preset_id']}")
        else:
            fallback_count += 1
            print(f"    -> 폴백 (사유: {result['fallback_reason']})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 분류 완료 ===")
    for preset_id, count in sorted(preset_counts.items(), key=lambda x: -x[1]):
        print(f"  {preset_id}: {count}건")
    print(f"  폴백(자유 스키마 필요): {fallback_count}건")
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()