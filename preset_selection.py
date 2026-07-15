"""
preset_selection.py

[C+ 단계] 저렴한 후보 스크리닝 + E(round-trip) 기반 실측 검증으로 preset 선택

기존 C(preset_classifier.py)는 "이 텍스트는 어떤 모양이어야 하는가"를 LLM
자기 보고(단일 분류 + confidence)에 의존해 한 번에 확정한다. 문제는 텍스트가
여러 preset으로 동시에 그럴듯하게 보일 수 있다는 점이다 -- 단일 분류는 그중
하나만 채택하고 나머지는 아예 후보에도 못 올린다.

이 모듈은 절충안을 구현한다:
1. (저렴) 후보 스크리닝 -- "이 텍스트에 5개 preset 중 어떤 게 구조적으로
   그럴듯한가"를 다중 판정으로 한 번만 물어본다. 여기서는 정밀한 판단을
   기대하지 않는다 -- 말이 안 되는 것만 걸러내는 용도.
2. (비쌈) 후보로 남은 preset들만 실제로 스캔(A-1)+추출(A-2)을 돌려 표를 만든다.
3. 만들어진 표들을 E(round-trip, roundtrip_verify.verify_extraction 재사용)로
   평가해서, "LLM이 이 preset이 맞다고 말했다"가 아니라 "실제로 만들어보니
   원문을 잘 담았다"는 측정치로 최종 채택한다 -- LLM 자기절제를 믿지 않는다는
   원칙을 preset 선택 자체에도 적용한 것.

의도적으로 다루지 않는 것
-------------------------
- 청크 분할: 이 모듈은 단일 텍스트 덩어리를 대상으로 한다. 긴 문서에 적용하려면
  청크마다 이 선택을 반복하거나 D(chunk_orchestrator)와 결합하는 별도 설계가
  필요하다 -- 지금은 호출 전에 chunk_orchestrator.needs_chunking()으로 판단해
  청크가 필요하면 기존 C(단일 분류)로 폴백하는 쪽을 택한다.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

from preset_library import PRESETS
from schema_scan import scan_schema
from schema_extract import extract_table
from roundtrip_verify import verify_extraction

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

# 청크가 필요한 긴 문서에서 C+를 쓸 때, 후보 표를 만들고 검증하는 데 쓸 표본
# 크기 상한 -- preset_classifier.DEFAULT_CLASSIFICATION_SAMPLE_CHARS와 같은 값을
# 쓴다 (C가 이미 표본 기반 판단이라는 리스크를 안고 있으므로, C+도 같은 수준의
# 리스크를 유지하는 선택 -- 문서 전체를 후보마다 통째로 재추출하는 건 비용이
# 너무 크다).
DEFAULT_SELECTION_SAMPLE_CHARS = 16000

# E 평가 점수(환각/누락 coverage 평균)가 이 미만이면 "채택할 만큼 좋지 않다"로 판단
DEFAULT_MIN_ACCEPT_SCORE = 0.75


# ─────────────────────────────────────────────────────────────
# 1. 저렴한 후보 스크리닝 (다중 판정, 구조만 봄 -- 표는 아직 안 만듦)
# ─────────────────────────────────────────────────────────────
def build_screening_prompt(text: str) -> str:
    preset_descs = "\n".join(f"- {pid}: {p.row_unit_desc}" for pid, p in PRESETS.items())
    return f"""당신은 텍스트를 표로 만들기 전에, 어떤 "표 모양(preset)"이 이 텍스트에
구조적으로 성립할 수 있는지 넓게 스크리닝하는 분석가입니다. 아직 표를 만들지
마세요 -- 어떤 모양들이 "말이 되는지"만 판단하면 됩니다. 하나의 텍스트가 여러
모양으로 동시에 성립할 수 있습니다 (예: 거래 기록 나열은 시점별 거래 표로도,
회사별 최종 요약 표로도 만들 수 있습니다). 확실하지 않으면 넓게 포함하고,
명백히 말이 안 되는 것만 제외하세요.

[preset 후보 목록]
{preset_descs}

[텍스트]
{text}

아래 JSON 형식으로만 답하세요. 다른 설명이나 markdown 코드펜스는 쓰지 마세요.
{{"candidates": ["preset_id", "preset_id", ...]}}
"""


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


def call_ollama_screen(
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

    raise RuntimeError(f"Ollama 스크리닝 호출이 {max_retries}회 모두 실패했습니다: {last_error}")


def screen_preset_candidates(
    text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[str]:
    """
    LLM이 보고한 후보 중, 실제로 존재하는 preset_id만 남긴다 (LLM 출력을
    신뢰하지 않고 코드로 검증하는 원칙 -- 존재하지 않는 preset_id를 그대로
    믿고 넘기면 이후 단계에서 KeyError로 죽는다).
    """
    prompt = build_screening_prompt(text)
    raw = call_ollama_screen(prompt, model=model, ollama_url=ollama_url, timeout=timeout, max_retries=max_retries)
    candidates = raw.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    valid = [c for c in candidates if isinstance(c, str) and c in PRESETS]
    return list(dict.fromkeys(valid))  # 중복 제거, 순서 유지


# ─────────────────────────────────────────────────────────────
# 2. 후보별로 실제 표 생성 (A-1 + A-2, 청크 없음)
# ─────────────────────────────────────────────────────────────
def build_candidate_table(preset_id: str, text: str, llm_kwargs: dict) -> dict:
    preset = PRESETS[preset_id]
    scan_result = (
        scan_schema(preset_id, text, **llm_kwargs)
        if preset.extension_budget_default is not None
        else {"skipped": True, "reason": "확장 열이 구조적으로 없는 preset"}
    )
    accepted = scan_result.get("accepted_columns") if not scan_result.get("skipped") else None
    extraction = extract_table(preset_id, text, accepted_extension=accepted, **llm_kwargs)
    return {"preset_id": preset_id, "scan": scan_result, **extraction}


# ─────────────────────────────────────────────────────────────
# 3. 후보 평가 (E 재사용) + 결정론적 채택 정책
# ─────────────────────────────────────────────────────────────
def _combined_score(verify_result: dict) -> float:
    hall = verify_result.get("hallucination_check") or {}
    omit = verify_result.get("omission_check") or {}
    return round((hall.get("coverage_ratio", 0.0) + omit.get("coverage_ratio", 0.0)) / 2, 3)


def select_preset_with_verification(
    text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    genre: str = "informative",
    min_accept_score: float = DEFAULT_MIN_ACCEPT_SCORE,
    fallback_single_classify: bool = True,
) -> dict:
    """
    1) 스크리닝으로 후보 preset을 넓게 뽑고
    2) 후보마다 실제 표를 만들고
    3) E로 평가해서, 임계치(min_accept_score)를 넘는 것 중 점수가 가장 높은
       것을 채택한다. 아무도 임계치를 못 넘으면, 그래도 그 중 최고점을 채택하되
       selection_passed_threshold=False로 표시해 "품질이 보장되지 않았다"는
       신호를 남긴다.

    후보가 하나도 안 나오거나(스크리닝 실패/전부 표 생성 실패)
    fallback_single_classify=True면 기존 C(preset_classifier)의 단일 분류로
    폴백한다 -- "아무 표도 안 나오는 것"보다는 낫다는 판단.
    """
    llm_kwargs = {"model": model, "ollama_url": ollama_url, "timeout": timeout, "max_retries": max_retries}

    candidates = screen_preset_candidates(text, **llm_kwargs)
    if not candidates:
        return _fallback_or_empty(text, llm_kwargs, fallback_single_classify, reason="스크리닝에서 후보를 하나도 찾지 못함")

    print(f"    [C+] 스크리닝 후보: {candidates}")

    evaluated = []
    for preset_id in candidates:
        try:
            candidate_result = build_candidate_table(preset_id, text, llm_kwargs)
        except Exception as e:
            print(f"    [경고] {preset_id} 후보 표 생성 실패, 건너뜀: {e}")
            continue

        table_markdown = candidate_result.get("table_markdown", "")
        if not table_markdown.strip():
            print(f"    [경고] {preset_id} 후보가 빈 표를 반환함, 건너뜀")
            continue

        try:
            verify_result = verify_extraction(
                text, table_markdown, genre=genre, preset_id=preset_id, **llm_kwargs,
            )
        except Exception as e:
            print(f"    [경고] {preset_id} round-trip 평가 실패, 건너뜀: {e}")
            continue

        score = _combined_score(verify_result)
        evaluated.append({
            "preset_id": preset_id,
            "extraction": candidate_result,
            "verification": verify_result,
            "score": score,
        })
        hall_r = (verify_result.get("hallucination_check") or {}).get("coverage_ratio")
        omit_r = (verify_result.get("omission_check") or {}).get("coverage_ratio")
        print(f"    [C+] 후보 {preset_id}: round-trip 점수={score} (환각coverage={hall_r}, 누락coverage={omit_r})")

    if not evaluated:
        return _fallback_or_empty(text, llm_kwargs, fallback_single_classify, reason="후보들이 전부 표 생성/평가에 실패함")

    passing = [e for e in evaluated if e["score"] >= min_accept_score]
    pool = passing if passing else evaluated
    best = max(pool, key=lambda e: e["score"])

    if not passing:
        print(f"    [경고] 임계치({min_accept_score})를 넘는 후보가 없음 -- 최고점({best['preset_id']}, {best['score']})을 그대로 채택")

    return {
        "success": True,
        "selected_preset_id": best["preset_id"],
        "selection_score": best["score"],
        "selection_passed_threshold": bool(passing),
        "candidates_tried": candidates,
        "candidates_evaluated": evaluated,
        "extraction": best["extraction"],
        "table_markdown": best["extraction"].get("table_markdown", ""),
    }


def _fallback_or_empty(text: str, llm_kwargs: dict, fallback_single_classify: bool, reason: str) -> dict:
    if not fallback_single_classify:
        return {"success": False, "reason": reason, "candidates_evaluated": []}

    from preset_classifier import classify_and_select_preset  # 지연 import (순환 참조 방지용은 아니고, 폴백 시에만 필요)

    print(f"    [C+] {reason} -> 기존 C(단일 분류)로 폴백")
    classification = classify_and_select_preset(text, **llm_kwargs)
    preset_id = classification.get("preset_id")
    if not preset_id:
        return {
            "success": False,
            "reason": f"{reason}; 폴백 분류도 실패 ({classification.get('fallback_reason')})",
            "candidates_evaluated": [],
        }

    try:
        extraction = build_candidate_table(preset_id, text, llm_kwargs)
    except Exception as e:
        return {
            "success": False,
            "reason": f"폴백 preset({preset_id})으로도 표 생성 실패: {e}",
            "candidates_evaluated": [],
        }

    return {
        "success": True,
        "selected_preset_id": preset_id,
        "selection_score": None,
        "selection_passed_threshold": None,
        "fallback_used": True,
        "fallback_reason": reason,
        "candidates_tried": [preset_id],
        "candidates_evaluated": [],
        "extraction": extraction,
        "table_markdown": extraction.get("table_markdown", ""),
    }


# ─────────────────────────────────────────────────────────────
# 4. CLI (다른 A-1/A-2/E와 동일하게, 단독 디버깅용 격리 도구)
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="C+ 단계: 후보 preset 스크리닝 + round-trip 검증 기반 선택을 실행합니다."
    )
    parser.add_argument("--input", required=True, help="입력 텍스트 파일(.txt)")
    parser.add_argument("--output", default=None, help="결과 .json 경로")
    parser.add_argument("--genre", choices=["informative", "narrative"], default="informative")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--min-accept-score", type=float, default=DEFAULT_MIN_ACCEPT_SCORE)
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="후보가 전부 실패했을 때 기존 C(단일 분류)로 폴백하지 않고 그냥 실패로 남긴다."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8").strip()
    print(f"입력: {input_path} ({len(text)}자)\n")

    result = select_preset_with_verification(
        text,
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        genre=args.genre,
        min_accept_score=args.min_accept_score,
        fallback_single_classify=not args.no_fallback,
    )

    output_path = Path(args.output) if args.output else input_path.with_suffix(".preset_selection.json")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 결과 ===")
    if result["success"]:
        print(f"채택된 preset: {result['selected_preset_id']} (점수={result.get('selection_score')}, "
              f"임계치 통과={result.get('selection_passed_threshold')})")
    else:
        print(f"실패: {result.get('reason')}")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()