"""
table_to_longtext.py

[1단계] 표 -> 장문 서술 (Verbalization)

pdf_table_extractor.py 가 생성한 `{stem}_tables.json` (table_markdown 필드 포함)을
입력으로 받아, 각 표를 로컬 Ollama LLM을 통해 자연스러운 한국어 장문 서술로 변환한다.

설계 의도
---------
- PDF -> 표 추출은 pdf_table_extractor.py 가 담당하므로 여기서 재구현하지 않는다.
- 표가 이미 Markdown 문자열로 와 있으므로, 별도 그룹핑/렌더링 없이 그대로 프롬프트에 삽입한다.
- 표 -> 장문 변환 시 "표의 모든 셀 값이 최소 1회 이상 언급되었는가"를 자동 체크하여,
  이후 라운드트립(장문 -> 표) 검증 단계에서 어느 쪽이 문제인지 구분할 수 있게 한다.

사용 예
-------
    python table_to_longtext.py --input output_docs --output-dir longtext_out --check-coverage
    python table_to_longtext.py --input output_docs/report_tables.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────
# 기본 설정 (text_to_table_pipeline.py 와 동일한 로컬 Ollama 규약)
# ─────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────
# 1. tables.json 로딩
# ─────────────────────────────────────────────────────────────
def load_table_entries(input_path: Path) -> list[dict]:
    """
    input_path 가 단일 *_tables.json 파일이면 그 파일만, 디렉터리면 하위의
    모든 *_tables.json 파일을 읽어 표 엔트리 리스트로 합쳐 반환한다.
    각 엔트리에 source_file(원본 tables.json 경로)을 부여한다.
    """
    files: list[Path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*_tables.json"))
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {input_path}")

    if not files:
        raise FileNotFoundError(
            f"'{input_path}' 에서 *_tables.json 파일을 찾지 못했습니다. "
            "pdf_table_extractor.py 를 먼저 실행했는지 확인하세요."
        )

    entries: list[dict] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            entry = dict(entry)
            entry["source_file"] = str(f)
            entries.append(entry)

    return entries


# ─────────────────────────────────────────────────────────────
# 2. Markdown 표 파싱 (커버리지 체크용 셀 값 추출)
# ─────────────────────────────────────────────────────────────
def extract_cell_values_from_markdown(table_markdown: str) -> list[str]:
    """
    파이프 문법 Markdown 표에서 헤더/구분선을 제외한 데이터 셀 값만 평탄화하여 반환.
    완벽한 마크다운 파서는 아니며, 커버리지 체크용 근사치 추출이 목적이다.
    """
    if not table_markdown.strip():
        return []

    lines = [ln for ln in table_markdown.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    values: list[str] = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        values.extend(c for c in cells if c)

    return values


# ─────────────────────────────────────────────────────────────
# 3. Ollama 호출 (재시도 + 지수 백오프)
# ─────────────────────────────────────────────────────────────
def call_ollama_generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if not text:
                raise ValueError("Ollama가 빈 응답을 반환했습니다.")
            return text
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

    raise RuntimeError(f"Ollama 호출이 {max_retries}회 모두 실패했습니다: {last_error}")


# ─────────────────────────────────────────────────────────────
# 4. 프롬프트 구성 & 검증
# ─────────────────────────────────────────────────────────────
def build_verbalization_prompt(
    table_markdown: str,
    context_before: str,
    context_after: str,
    genre: str = "informative",
) -> str:
    if genre == "narrative":
        style_instruction = (
            "이 표는 서사(소설/일기 등) 문맥에서 등장하는 정보입니다. "
            "시간 순서나 인과관계가 드러나도록 자연스러운 이야기체 문장으로 서술하세요."
        )
    else:
        style_instruction = (
            "이 표는 보고서/논문 등 정보 전달 문서에서 등장합니다. "
            "정확하고 간결한 설명체 문장으로, 문단 형태로 서술하세요."
        )

    return f"""당신은 Markdown 표를 빠짐없이 자연스러운 한국어 문장으로 풀어 쓰는 전문가입니다.

[규칙]
1. 아래 "표 원자료"에 있는 모든 셀 값을 반드시 한 번 이상 문장에 포함하세요. 하나라도 빠뜨리면 안 됩니다.
2. 표에 없는 정보를 추측하거나 지어내지 마세요.
3. 숫자는 원본 표기(단위 포함)를 그대로 유지하세요.
4. {style_instruction}
5. 출력은 서술 문단만 작성하고, 표 형식이나 목록(불릿) 형식은 사용하지 마세요.
6. 표 앞뒤 문맥이 주어지면 자연스럽게 이어지도록 참고하되, 문맥 자체를 반복 서술하지는 마세요.

[표 앞 문맥]
{context_before or "(없음)"}

[표 원자료 (Markdown)]
{table_markdown}

[표 뒤 문맥]
{context_after or "(없음)"}

위 규칙을 지켜 표 내용을 빠짐없이 서술한 문단을 작성하세요.
"""


def check_value_coverage(long_text: str, values: list[str]) -> dict:
    """
    표의 각 값이 생성된 장문에 실제로 등장하는지 확인.
    쉼표/공백 차이는 무시하고 느슨하게 비교한다(완벽한 검증은 2단계 라운드트립에서 수행).
    """
    normalized_text = re.sub(r"[,\s]", "", long_text)
    missing = []
    for v in values:
        v_norm = re.sub(r"[,\s]", "", str(v))
        if not v_norm:
            continue
        if v_norm not in normalized_text:
            missing.append(v)

    total = len([v for v in values if str(v).strip()])
    covered = total - len(missing)
    return {
        "total_values": total,
        "covered_values": covered,
        "coverage_ratio": round(covered / total, 3) if total else 1.0,
        "missing_values": missing,
    }


# ─────────────────────────────────────────────────────────────
# 4.5. 중간 결과 저장/로드 (JSONL, 처리 즉시 1건씩 append)
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
# 5. 메인 처리
# ─────────────────────────────────────────────────────────────
def process_table_entry(
    entry: dict,
    model: str,
    ollama_url: str,
    timeout: int,
    max_retries: int,
    genre: str,
    check_coverage: bool,
) -> dict:
    table_markdown = entry.get("table_markdown", "")

    if not table_markdown.strip():
        return {
            "table_id": entry.get("table_id"),
            "source_file": entry.get("source_file"),
            "page_number": entry.get("page_number"),
            "long_text": "",
            "skipped_reason": "빈 표 (table_markdown 없음)",
        }

    prompt = build_verbalization_prompt(
        table_markdown,
        entry.get("context_before_table", ""),
        entry.get("context_after_table", ""),
        genre=genre,
    )

    long_text = call_ollama_generate(
        prompt, model=model, ollama_url=ollama_url, timeout=timeout, max_retries=max_retries
    )

    result = {
        "table_id": entry.get("table_id"),
        "source_file": entry.get("source_file"),
        "page_number": entry.get("page_number"),
        "context_before": entry.get("context_before_table", ""),
        "context_after": entry.get("context_after_table", ""),
        "table_markdown": table_markdown,
        "long_text": long_text,
    }

    if check_coverage:
        values = extract_cell_values_from_markdown(table_markdown)
        result["coverage"] = check_value_coverage(long_text, values)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="pdf_table_extractor.py의 *_tables.json(table_markdown)을 읽어 표를 장문 서술로 변환합니다."
    )
    parser.add_argument("--input", required=True, help="*_tables.json 파일 또는 이를 포함한 디렉터리")
    parser.add_argument("--output-dir", default="longtext_output", help="결과 저장 디렉터리")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 모델명")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama 서버 주소")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="요청 타임아웃(초)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="재시도 횟수")
    parser.add_argument(
        "--genre",
        choices=["informative", "narrative"],
        default="informative",
        help="서술 스타일: informative(보고서/논문) 또는 narrative(소설/일기)",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="생성된 장문에 표의 모든 값이 포함되었는지 느슨하게 검증",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="output-dir의 기존 tables_longtext.jsonl을 읽어 이미 성공한 table_id는 건너뛰고 이어서 처리",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_table_entries(input_path)
    print(f"총 {len(entries)}개의 표 엔트리를 로드했습니다.")

    jsonl_path = output_dir / "tables_longtext.jsonl"

    # --resume: 기존 jsonl에서 이미 "성공"한 table_id는 건너뛴다.
    # (실패했던 항목은 done_ids에 포함시키지 않아 이번 실행에서 다시 시도된다.)
    existing_results: dict[str, dict] = {}
    if args.resume and jsonl_path.exists():
        for r in _read_jsonl(jsonl_path):
            tid = r.get("table_id")
            if tid is not None:
                existing_results[tid] = r  # 같은 table_id가 여러 번 있으면 마지막 것으로 덮어씀
        done_ids = {
            tid for tid, r in existing_results.items() if r.get("long_text") and not r.get("error")
        }
        print(f"[재개 모드] 기존 결과 {len(existing_results)}건 로드, 성공한 {len(done_ids)}건은 건너뜁니다.")
    else:
        done_ids = set()
        if jsonl_path.exists():
            print(f"[주의] --resume 없이 실행되어 기존 {jsonl_path.name}을 새로 덮어씁니다.")
            jsonl_path.unlink()

    # 재개 모드에서 이미 성공한 결과는 최종 집계에 그대로 포함시킨다.
    results: list[dict] = [r for tid, r in existing_results.items() if tid in done_ids]

    for idx, entry in enumerate(entries, 1):
        table_id = entry.get("table_id")
        if args.resume and table_id in done_ids:
            print(f"[{idx}/{len(entries)}] table_id={table_id} 이미 완료됨 -> 건너뜀")
            continue

        print(f"[{idx}/{len(entries)}] table_id={table_id} 처리 중...")
        try:
            result = process_table_entry(
                entry,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                genre=args.genre,
                check_coverage=args.check_coverage,
            )

            if args.check_coverage and "coverage" in result:
                cov = result["coverage"]
                if cov["coverage_ratio"] < 1.0:
                    print(
                        f"    [커버리지 경고] {cov['covered_values']}/{cov['total_values']} "
                        f"({cov['coverage_ratio']*100:.1f}%) - 누락: {cov['missing_values'][:5]}"
                        f"{' 외 다수' if len(cov['missing_values']) > 5 else ''}"
                    )
        except Exception as e:
            print(f"    [오류] table_id={table_id} 처리 실패: {e}")
            result = {
                "table_id": table_id,
                "source_file": entry.get("source_file"),
                "error": str(e),
            }

        results.append(result)
        _append_jsonl(jsonl_path, result)  # <- 항목 처리 즉시 디스크에 저장 (핵심)

    # 재개로 쌓였을 수 있는 중복/실패 잔여 라인을 정리하기 위해
    # 최종 결과 기준으로 jsonl을 한 번 깔끔하게 재작성한다.
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_json_path = output_dir / "tables_longtext.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_txt_path = output_dir / "tables_longtext.txt"
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"=== table_id: {r.get('table_id')} (page {r.get('page_number')}) ===\n")
            f.write(r.get("long_text", "") or f"[생성 실패: {r.get('error', r.get('skipped_reason', ''))}]")
            f.write("\n\n")

    success_count = len([r for r in results if r.get("long_text")])
    print(f"\n완료: {success_count}/{len(results)}개 표 서술 생성")
    print(f"중간 저장(재개용): {jsonl_path}")
    print(f"결과 저장: {out_json_path}")
    print(f"미리보기용 텍스트: {out_txt_path}")


if __name__ == "__main__":
    main()