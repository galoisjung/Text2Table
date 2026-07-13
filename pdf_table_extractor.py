"""
pdf_table_extractor.py

pdf2json.py (다른 프로젝트) 를 이식한 버전.
표를 tidy JSON 대신 Markdown 표 문자열로 추출한다는 점만 다르다.

산출물 (per PDF, stem 기준)
---------------------------
- {stem}_texts.json    : 페이지별 텍스트 (벡터DB 적재용, 원본과 동일)
- {stem}_tables.json   : 표 메타데이터 + table_markdown 필드
- {stem}_tables.md     : 사람이 바로 읽을 수 있는 통합 마크다운 (context + 표)
- {stem}_meta.txt      : 처리 요약
"""

from __future__ import annotations

import gc
import json
import re
import uuid
from pathlib import Path

import pandas as pd

from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.factories.ocr_factory import OcrFactory
from paddle_ocr_model import PaddleOcrModel, PaddleOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling_core.types.doc import TextItem, TableItem
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# ─────────────────────────────────────────────────────────────
# 기본 설정 (CLI 인자로 덮어쓸 수 있음)
# ─────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = Path("samples")
DEFAULT_OUTPUT_DIR = Path("output_docs")
DEFAULT_GPU = 0

AFTER_CONTEXT_LIMIT = 3

converter = None


class PaddlePdfPipeline(StandardPdfPipeline):
    def get_ocr_model(self, artifacts_path):
        factory = OcrFactory()
        factory.register(
            cls=PaddleOcrModel,
            plugin_name="paddle_ocr",
            plugin_module_name="paddle_ocr_model",
        )
        return factory.create_instance(
            options=self.pipeline_options.ocr_options,
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )


def init_converter(gpu_id: int = DEFAULT_GPU) -> DocumentConverter:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[초기화] GPU {gpu_id}번 할당")

    opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=False,
        generate_page_images=True,
        images_scale=2.0,
        ocr_options=PaddleOcrOptions(lang=["korean"], force_full_page_ocr=False),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts,
                pipeline_cls=PaddlePdfPipeline,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


# ─────────────────────────────────────────────────────────────
# 표 정규화 헬퍼 (원본 pdf2json.py 와 동일)
# ─────────────────────────────────────────────────────────────
_UNIT_RE = re.compile(r'\(?\s*단위\s*[:：]?\s*([가-힣a-zA-Z0-9%,\s]+)\)?')
_NOTE_RE = re.compile(r'\(\s*주\s*\d*\s*\)')
_ID_KWS = {"구분", "항목", "계정과목", "과목", "내용", "분류", "구성", "구성요소", "항목명", "계정"}


def _extract_unit(s: str) -> tuple[str, str]:
    m = _UNIT_RE.search(s)
    unit = m.group(1).strip() if m else ""
    cleaned = _UNIT_RE.sub("", s)
    cleaned = _NOTE_RE.sub("", cleaned).strip("_ ").strip()
    return cleaned, unit


def _normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    col_units: dict[str, str] = {}
    df = df.copy()

    def _col_key(col) -> str:
        return "_".join(str(c) for c in col) if isinstance(col, tuple) else str(col)

    if isinstance(df.columns, pd.MultiIndex):
        depth = df.columns.nlevels
        new_tuples: list[tuple] = []

        for col_tuple in df.columns:
            levels: list[str] = []
            tuple_unit = ""
            for lv in col_tuple:
                cleaned, unit = _extract_unit(str(lv))
                if unit:
                    tuple_unit = unit
                levels.append(cleaned)
            while len(levels) > 1 and not levels[-1]:
                levels.pop()
            levels += [""] * (depth - len(levels))
            if tuple_unit:
                # 정제된(최종) 레벨 튜플로 키를 저장해야 이후 조회 시 일치함
                col_units[_col_key(tuple(levels))] = tuple_unit
            new_tuples.append(tuple(levels))

        seen: dict[tuple, int] = {}
        final_tuples: list[tuple] = []
        for t in new_tuples:
            if t in seen:
                seen[t] += 1
                lst = list(t)
                lst[-1] = (f"{lst[-1]}__{seen[t]}" if lst[-1] else str(seen[t]))
                final_tuples.append(tuple(lst))
            else:
                seen[t] = 0
                final_tuples.append(t)

        df.columns = pd.MultiIndex.from_tuples(final_tuples)

    else:
        new_cols: list[str] = []
        seen_s: dict[str, int] = {}

        for col in df.columns:
            raw = " ".join(str(c) for c in col if str(c).strip()) if isinstance(col, tuple) else str(col)
            cleaned, unit = _extract_unit(raw)
            if not cleaned:
                cleaned = "Unnamed"
            if unit:
                # 정제된(최종) 컬럼명으로 키를 저장해야 dataframe_to_markdown 등에서 바로 조회 가능
                col_units[cleaned] = unit

            if cleaned in seen_s:
                seen_s[cleaned] += 1
                new_cols.append(f"{cleaned}__{seen_s[cleaned]}")
            else:
                seen_s[cleaned] = 0
                new_cols.append(cleaned)

        df.columns = new_cols

    return df, col_units


def _detect_id_columns(df: pd.DataFrame) -> list[int]:
    n = len(df.columns)
    id_idx: list[int] = []

    for i, col in enumerate(df.columns):
        col_str = "".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
        series = df.iloc[:, i].astype(str).str.strip()
        non_null = series[~series.isin(["", "nan", "None"])]

        if non_null.empty:
            continue

        is_kw = any(kw in col_str for kw in _ID_KWS) or "Unnamed" in col_str
        num_ratio = (
            non_null.str.replace(",", "", regex=False)
            .str.match(r"^-?\d+\.?\d*%?$")
            .mean()
        )
        is_leading = i < max(3, n // 3)

        if is_kw or (is_leading and num_ratio < 0.25):
            id_idx.append(i)
        else:
            break

    return id_idx if id_idx else [0]


def _forward_fill_id_columns(df: pd.DataFrame, id_col_indices: list[int]) -> pd.DataFrame:
    """rowspan 으로 인해 비어있는 행 기준 컬럼 셀을 위 행 값으로 채운다."""
    df = df.copy()
    for idx in id_col_indices:
        col = df.columns[idx]
        mask = df[col].astype(str).str.strip().isin(["", "nan", "None"])
        df.loc[mask, col] = None
        df[col] = df[col].ffill()
    return df


# ─────────────────────────────────────────────────────────────
# 신규: DataFrame -> Markdown 표 변환
# ─────────────────────────────────────────────────────────────
def dataframe_to_markdown(df: pd.DataFrame, col_units: dict | None = None) -> str:
    """
    (MultiIndex 컬럼 지원) DataFrame을 파이프 문법 Markdown 표로 변환.
    단위(col_units)가 있으면 헤더 셀에 '(단위)' 형태로 덧붙인다.
    """
    if df.empty:
        return ""

    col_units = col_units or {}

    def _col_key(col) -> str:
        return "_".join(str(c) for c in col) if isinstance(col, tuple) else str(col)

    headers: list[str] = []
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            label = " / ".join(str(c) for c in col if str(c).strip())
            unit = col_units.get(_col_key(col), "")
            headers.append(f"{label} ({unit})" if unit else label)
    else:
        for col in df.columns:
            unit = col_units.get(str(col), "")
            headers.append(f"{col} ({unit})" if unit else str(col))

    def _esc(v) -> str:
        s = "" if v is None else str(v)
        s = s.strip()
        if s in ("nan", "None"):
            return ""
        return s.replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(_esc(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row_idx in range(len(df)):
        row_vals = [_esc(df.iat[row_idx, i]) for i in range(len(df.columns))]
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────────────────────────
def process_single_pdf(pdf_path: Path, output_dir: Path):
    global converter

    stem = pdf_path.stem
    table_images_dir = output_dir / f"{stem}_table_images"
    table_images_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  [{pdf_path.name}] 파싱 시작...")
        result = converter.convert(pdf_path)
        doc = result.document

        texts_for_vector_db = []
        tables_with_context = []

        context_buffer = []
        table_img_count = 0

        pending_table = None
        after_context_count = 0

        table_counter = 0

        for element, _level in doc.iterate_items():

            if isinstance(element, TextItem):
                text_content = element.text.strip()
                if not text_content:
                    continue

                page_no = None
                if hasattr(element, "prov") and element.prov:
                    page_no = element.prov[0].page_no

                texts_for_vector_db.append({"text": text_content, "page_number": page_no})

                context_buffer.append(text_content)
                if len(context_buffer) > 3:
                    context_buffer.pop(0)

                if pending_table is not None:
                    if pending_table["context_after_table"]:
                        pending_table["context_after_table"] += "\n" + text_content
                    else:
                        pending_table["context_after_table"] = text_content
                    after_context_count += 1
                    if after_context_count >= AFTER_CONTEXT_LIMIT:
                        pending_table = None

            elif isinstance(element, TableItem):
                page_no = None
                if hasattr(element, "prov") and element.prov:
                    page_no = element.prov[0].page_no

                table_img_path = None
                col_units: dict = {}
                try:
                    df = element.export_to_dataframe(doc)
                    df, col_units = _normalize_columns(df)

                    for col in df.columns:
                        if df[col].dtype == "object":
                            df[col] = df[col].astype(str).str.replace(r'\(\s*주\s*\d*\s*\)', "", regex=True)
                            df[col] = df[col].str.replace(",", "", regex=False)

                    def _is_meaningless(c) -> bool:
                        s = "".join(str(lv) for lv in c) if isinstance(c, tuple) else str(c)
                        return s.isdigit() or s.startswith("Unnamed")

                    if all(_is_meaningless(c) for c in df.columns):
                        col_units = {}
                        if len(df.columns) == 2:
                            df.columns = ["구분", "내용"]
                        else:
                            df.columns = [f"항목_{i + 1}" for i in range(len(df.columns))]

                    current_columns = list(df.columns)
                    id_cols = _detect_id_columns(df)
                    df = _forward_fill_id_columns(df, id_cols)

                except Exception as e:
                    table_img_count += 1
                    try:
                        tbl_img = element.get_image(doc)
                        if tbl_img is not None:
                            tbl_img_filename = f"table_{table_img_count}.png"
                            tbl_img.save(table_images_dir / tbl_img_filename, format="PNG")
                            print(f"    [표 이미지 폴백] {stem} - {tbl_img_filename} (사유: {e})")
                    except Exception as img_e:
                        print(f"    [경고] {stem} - 표 이미지 폴백 저장 실패: {img_e}")
                    continue

                is_merged = False
                if tables_with_context:
                    prev_entry = tables_with_context[-1]
                    prev_columns = prev_entry.get("columns", [])
                    cleaned_buffer = [t for t in context_buffer if t.strip() and not t.isdigit()]

                    if len(current_columns) == len(prev_columns) and len(cleaned_buffer) == 0:
                        prev_df = prev_entry["df"]
                        if current_columns == prev_columns:
                            merged_df = pd.concat([prev_df, df], ignore_index=True)
                        else:
                            df.columns = prev_columns
                            merged_df = pd.concat([prev_df, df], ignore_index=True)

                        prev_entry["df"] = merged_df
                        is_merged = True
                        prev_entry["is_merged"] = True
                        if table_img_path:
                            prev_entry.setdefault("table_image_paths", []).append(table_img_path)
                        print(f"    [병합 완료] {stem} - 분할된 표 통합")

                if not is_merged:
                    table_counter += 1
                    new_table_entry = {
                        "table_id": str(uuid.uuid4()),
                        "page_number": page_no,
                        "table_sequence": table_counter,
                        "is_merged": False,
                        "context_before_table": "\n".join(context_buffer),
                        "context_after_table": "",
                        "table_image_paths": [table_img_path] if table_img_path else [],
                        "metadata": {"col_units": col_units} if col_units else {},
                        "df": df,
                        "columns": current_columns,
                    }
                    tables_with_context.append(new_table_entry)
                    context_buffer.clear()
                    pending_table = new_table_entry
                    after_context_count = 0

        # ── 표 엔트리 최종 직렬화 (DataFrame -> Markdown) ──
        final_table_outputs = []
        for entry in tables_with_context:
            final_df = entry.pop("df")
            _ = entry.pop("columns", None)
            col_units = entry.get("metadata", {}).get("col_units", {})
            entry["table_markdown"] = dataframe_to_markdown(final_df, col_units)
            final_table_outputs.append(entry)

        with open(output_dir / f"{stem}_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts_for_vector_db, f, ensure_ascii=False, indent=2)

        with open(output_dir / f"{stem}_tables.json", "w", encoding="utf-8") as f:
            json.dump(final_table_outputs, f, ensure_ascii=False, indent=2)

        # 사람이 바로 읽을 수 있는 통합 마크다운
        md_lines = [f"# {pdf_path.name}\n"]
        for entry in final_table_outputs:
            md_lines.append(f"## Table {entry['table_sequence']} (page {entry['page_number']})\n")
            if entry["context_before_table"]:
                md_lines.append(f"> {entry['context_before_table']}\n")
            md_lines.append(entry["table_markdown"] + "\n")
            if entry["context_after_table"]:
                md_lines.append(f"> {entry['context_after_table']}\n")
            md_lines.append("---\n")
        (output_dir / f"{stem}_tables.md").write_text("\n".join(md_lines), encoding="utf-8")

        (output_dir / f"{stem}_meta.txt").write_text(
            f"source={pdf_path.name}\n"
            f"extracted_texts={len(texts_for_vector_db)}\n"
            f"extracted_tables={len(final_table_outputs)}\n"
            f"table_image_fallbacks={table_img_count}\n",
            encoding="utf-8",
        )

        return (pdf_path.name, True, None)

    except Exception as e:
        return (pdf_path.name, False, str(e))

    finally:
        try:
            del result, doc
        except NameError:
            pass
        gc.collect()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF에서 표를 Markdown으로 추출합니다.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="PDF 파일이 있는 디렉터리")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="결과 저장 디렉터리")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU, help="사용할 GPU 번호")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pdf_files = list(input_dir.glob("*.pdf"))
    if not all_pdf_files:
        print(f"\n[안내] '{input_dir}' 폴더에 PDF 파일이 없습니다.")
        return

    pdf_files_to_process = []
    skipped_files = []
    for pdf in all_pdf_files:
        expected_meta_file = output_dir / f"{pdf.stem}_meta.txt"
        if expected_meta_file.exists():
            skipped_files.append(pdf.name)
        else:
            pdf_files_to_process.append(pdf)

    print(f"\n총 {len(all_pdf_files)}개의 PDF 중,")
    if skipped_files:
        print(f" ⏭️ [Save Point] 이미 완료된 {len(skipped_files)}개 파일은 건너뜁니다.")
    print(f" ▶️ 실제 변환을 진행할 파일: {len(pdf_files_to_process)}개\n")

    if not pdf_files_to_process:
        return

    global converter
    print("초기화 중...")
    converter = init_converter(args.gpu)

    success_count = 0
    fail_count = 0

    for idx, pdf in enumerate(pdf_files_to_process, 1):
        print(f"\n[{idx}/{len(pdf_files_to_process)}] '{pdf.name}' 파일 처리 시작...")
        file_name, success, error_msg = process_single_pdf(pdf, output_dir)
        if success:
            print(f"[{idx}/{len(pdf_files_to_process)}] ✓ '{file_name}' 처리 완료")
            success_count += 1
        else:
            print(f"[{idx}/{len(pdf_files_to_process)}] ✗ '{file_name}' 실패: {error_msg}")
            fail_count += 1

    print(f"\n=== 작업 완료 ===")
    print(f"새로 성공: {success_count}건 / 실패: {fail_count}건 (기존 완료: {len(skipped_files)}건)")
    print(f"결과물 저장 위치: {output_dir.resolve()}")


if __name__ == "__main__":
    main()