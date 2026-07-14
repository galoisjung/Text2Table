"""
table_to_longtext.py

[1단계] 표 -> 장문 서술 (Verbalization)

pdf_table_extractor.py 가 생성한 `{stem}_tables.md` (사람이 읽는 통합 마크다운,
표마다 <!-- table_id: ... --> 주석 포함)를 입력으로 받아, 각 표를 로컬 Ollama LLM을
통해 자연스러운 한국어 장문 서술로 변환한다.

설계 의도
---------
- PDF -> 표 추출은 pdf_table_extractor.py 가 담당하므로 여기서 재구현하지 않는다.
- LLM에게는 애초부터 순수 Markdown 표 문자열만 전달된다(이전 JSON 기반 버전도 마찬가지였음).
  이번 변경의 실질적 의미는 "우리 스크립트가 무엇을 파싱해서 구조화하는가"이며,
  .md 파일 하나를 사람이 검수하는 산출물이자 프로그램 입력으로 동시에 쓸 수 있게 한다.
- .md 는 사람이 읽기 좋게 만든 포맷이라 표를 식별할 손잡이가 없으므로,
  pdf_table_extractor.py 가 각 표 앞에 <!-- table_id: ... --> 숨김 주석을 심어 둔다.
- 표 -> 장문 변환 시 "표의 모든 셀 값이 최소 1회 이상 언급되었는가"를 자동 체크하여,
  이후 라운드트립(장문 -> 표) 검증 단계에서 어느 쪽이 문제인지 구분할 수 있게 한다.

사용 예
-------
    python table_to_longtext.py --input output_docs --output-dir longtext_out --check-coverage
    python table_to_longtext.py --input output_docs/report_tables.md
"""

from __future__ import annotations

import argparse
import json
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
# 1. tables.md 로딩 & 파싱
# ─────────────────────────────────────────────────────────────
_HEADER_RE = re.compile(r'^## Table (\d+) \(page (.+?)\)\s*$', re.MULTILINE)
_ID_COMMENT_RE = re.compile(r'<!--\s*table_id:\s*(.+?)\s*-->')
_TABLE_BLOCK_RE = re.compile(r'(^\|.+\|\s*$\n(?:^\|.+\|\s*$\n?)*)', re.MULTILINE)


def _dequote(text: str) -> str:
    """블록인용(>) 표시를 제거하고 빈 줄/구분선을 걸러내며 원래 줄바꿈 구조를 복원."""
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            line = line[1:].strip()
        if line and line != "---":
            cleaned.append(line)
    return "\n".join(cleaned)


def parse_tables_md(md_text: str, source_file: str) -> list[dict]:
    """
    pdf_table_extractor.py 가 생성한 `{stem}_tables.md` 텍스트를 파싱하여
    표 엔트리 리스트로 변환한다. '## Table N (page P)' 헤더로 표를 구분하고,
    <!-- table_id: ... --> 숨김 주석에서 식별자를, 연속된 '|...|' 줄에서
    표 본문을, 나머지 텍스트에서 앞/뒤 문맥을 복원한다.
    """
    matches = list(_HEADER_RE.finditer(md_text))
    entries: list[dict] = []

    for i, hm in enumerate(matches):
        seq, page = hm.group(1), hm.group(2)
        start = hm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        block = md_text[start:end]
        block = re.sub(r'\n---\s*$', '', block.strip())

        id_m = _ID_COMMENT_RE.search(block)
        table_id = id_m.group(1) if id_m else None
        block_wo_id = block[id_m.end():] if id_m else block

        tbl_m = _TABLE_BLOCK_RE.search(block_wo_id)
        table_markdown = tbl_m.group(1).strip() if tbl_m else ""
        before_text = block_wo_id[: tbl_m.start()] if tbl_m else block_wo_id
        after_text = block_wo_id[tbl_m.end():] if tbl_m else ""

        page_number: int | str | None = int(page) if page.isdigit() else (page or None)

        entries.append(
            {
                "table_id": table_id,
                "table_sequence": int(seq),
                "page_number": page_number,
                "context_before_table": _dequote(before_text),
                "context_after_table": _dequote(after_text),
                "table_markdown": table_markdown,
                "source_file": source_file,
            }
        )

    return entries


def load_table_entries(input_path: Path) -> list[dict]:
    """
    input_path 가 단일 *_tables.md 파일이면 그 파일만, 디렉터리면 하위의
    모든 *_tables.md 파일을 읽어 표 엔트리 리스트로 합쳐 반환한다.
    """
    files: list[Path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*_tables.md"))
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {input_path}")

    if not files:
        raise FileNotFoundError(
            f"'{input_path}' 에서 *_tables.md 파일을 찾지 못했습니다. "
            "pdf_table_extractor.py 를 먼저 실행했는지 확인하세요."
        )

    entries: list[dict] = []
    for f in files:
        md_text = f.read_text(encoding="utf-8")
        parsed = parse_tables_md(md_text, source_file=str(f))
        if not parsed:
            print(f"    [경고] '{f}' 에서 표를 찾지 못했습니다 (형식이 다른 것 같습니다).")
        entries.extend(parsed)

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
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as e:
            last_error = e
            wait = 2 ** (attempt - 1)
            print(f"    [재시도 {attempt}/{max_retries}] {e} -> {wait}초 대기 후 재시도")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 500 <= status < 600:
                # 5xx: 게이트웨이/서버 쪽 일시적 오류 -> 재시도 대상
                # (특히 -cloud 모델은 로컬이 아니라 Ollama 클라우드로 요청이 넘어가므로
                # 502 Bad Gateway 같은 오류가 일시적으로 발생할 수 있다)
                last_error = e
                wait = 2 ** (attempt - 1)
                print(f"    [재시도 {attempt}/{max_retries}] HTTP {status} -> {wait}초 대기 후 재시도")
                time.sleep(wait)
            else:
                # 4xx 등 클라이언트 오류는 재시도해도 소용없으므로 즉시 중단
                raise

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


# 하이픈처럼 보이지만 코드포인트가 다른 문자들 (LLM이 날짜/음수 표기에 흔히 섞어 씀).
# 정규화 없이 비교하면 "2024-06-17"(U+002D)과 "2024‑06‑17"(U+2011, non-breaking hyphen)이
# 육안으론 똑같아 보여도 문자열 비교에서는 다른 값으로 취급되어 실제로 존재하는 값이
# "누락"으로 오탐된다.
_DASH_VARIANTS_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")


def _normalize_for_comparison(s: str) -> str:
    s = re.sub(r"[,\s]", "", s)
    s = _DASH_VARIANTS_RE.sub("-", s)
    return s


_DATE_LIKE_RE = re.compile(
    r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{4}년\s*\d{1,2}월\s*\d{1,2}일$"
)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _value_covered(value: str, normalized_text: str, text_digits: str) -> bool:
    v_norm = _normalize_for_comparison(value)
    if v_norm and v_norm in normalized_text:
        return True
    # 날짜는 "1996년 08월 29일"과 "1996-08-29"처럼 표기가 달라도 같은 날짜일 수 있다.
    # 일반 문자열 비교로 실패했을 때만, 숫자만 남긴 형태로 한 번 더 확인한다
    # (임의의 숫자값까지 이 fallback을 타면 우연한 부분 일치가 생길 수 있으므로
    # 날짜처럼 보이는 값에만 적용한다).
    if _DATE_LIKE_RE.match(value.strip()):
        v_digits = _digits_only(value)
        if len(v_digits) >= 6 and v_digits in text_digits:
            return True
    return False


def check_value_coverage(long_text: str, values: list[str]) -> dict:
    """
    표의 각 값이 생성된 장문에 실제로 등장하는지 확인.
    쉼표/공백 차이와 하이픈류 문자 변형은 무시하고 느슨하게 비교하며,
    날짜처럼 보이는 값은 표기 형식(하이픈 vs "년/월/일")이 달라도 숫자만
    같으면 동일한 값으로 인정한다 (완벽한 검증은 2단계 라운드트립에서 수행).
    """
    normalized_text = _normalize_for_comparison(long_text)
    text_digits = _digits_only(long_text)
    missing = []
    for v in values:
        v_str = str(v)
        if not v_str.strip():
            continue
        if not _value_covered(v_str, normalized_text, text_digits):
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
        description="pdf_table_extractor.py의 *_tables.md를 읽어 표를 장문 서술로 변환합니다."
    )
    parser.add_argument("--input", required=True, help="*_tables.md 파일 또는 이를 포함한 디렉터리")
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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_table_entries(input_path)
    print(f"총 {len(entries)}개의 표 엔트리를 로드했습니다.")

    results = []
    for idx, entry in enumerate(entries, 1):
        print(f"[{idx}/{len(entries)}] table_id={entry.get('table_id')} 처리 중...")
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
            results.append(result)

            if args.check_coverage and "coverage" in result:
                cov = result["coverage"]
                if cov["coverage_ratio"] < 1.0:
                    print(
                        f"    [커버리지 경고] {cov['covered_values']}/{cov['total_values']} "
                        f"({cov['coverage_ratio']*100:.1f}%) - 누락: {cov['missing_values'][:5]}"
                        f"{' 외 다수' if len(cov['missing_values']) > 5 else ''}"
                    )
        except Exception as e:
            print(f"    [오류] table_id={entry.get('table_id')} 처리 실패: {e}")
            results.append(
                {
                    "table_id": entry.get("table_id"),
                    "source_file": entry.get("source_file"),
                    "error": str(e),
                }
            )

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
    print(f"결과 저장: {out_json_path}")
    print(f"미리보기용 텍스트: {out_txt_path}")


if __name__ == "__main__":
    main()