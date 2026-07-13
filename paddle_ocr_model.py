# paddle_ocr_model.py
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type

import cv2
import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.models.base_ocr_model import BaseOcrModel

_log = logging.getLogger(__name__)


class PaddleOcrOptions(OcrOptions):
    kind = "paddleocr"
    lang: list[str] = ["korean","english"]          # PaddleOCR 언어 코드
    use_angle_cls: bool = False    # 텍스트 방향 감지
    use_gpu: bool = True
    show_log: bool = False
    force_full_page_ocr: bool = True

    @classmethod
    def get_kind(cls) -> str:
        return "paddleocr"


class PaddleOcrModel(BaseOcrModel):

    options: PaddleOcrOptions

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: PaddleOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self._paddle = None

        # crop 저장 디렉토리 설정
        self._debug_dir = Path("debug_crops")
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._crop_count = 0  # 전체 crop 카운터

        if self.enabled:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                lang=options.lang[0]
            )
            
    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return PaddleOcrOptions

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:

        for page in page_batch:
            if not self.enabled or page.image is None:
                yield page
                continue

            ocr_rects = self.get_ocr_rects(page)
            if not ocr_rects:
                yield page
                continue

            all_ocr_cells: List[TextCell] = []

            for rect in ocr_rects:
                # 영역 좌표 추출 (TOPLEFT 기준)
                scale_x = page.image.width / page.size.width
                scale_y = page.image.height / page.size.height

                x0 = int(rect.l * scale_x)
                y0 = int(rect.t * scale_y)
                x1 = int(rect.r * scale_x)
                y1 = int(rect.b * scale_y)

                # 유효하지 않은 rect 스킵
                if x1 <= x0 or y1 <= y0:
                    continue

                # PIL 이미지 crop → numpy array
                crop = page.image.crop((x0, y0, x1, y1))
                img_array = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
                # PaddleOCR 실행
                try:
                    # 💡 자동 세팅된 객체에 표준 ocr() 메서드 호출
                    results = self._paddle.predict(img_array)
                except Exception as e:
                    _log.warning(f"PaddleOCR failed on page {page.page_no}: {e}")
                    continue

                if not results:
                    continue

                # 💡 [핵심 수정] PaddleOCR 버전에 따른 리스트 차원 자동 감지
                # 요소가 [ [[좌표..]], ('텍스트', 정확도) ] 구조인지 
                # 결과 파싱 → TextCell 변환
                for line in results:
                    # 안전장치: 구조가 [좌표배열, 텍스트정보] 가 아니면 스킵
                    quad_data = line['rec_polys']       # 좌표 데이터
                    text_data = line['rec_texts']  # 텍스트 및 신뢰도 데이터
                    for quad, text in zip(quad_data, text_data):
                    # text_data가 ('추출텍스트', 0.98) 같은 튜플 형태인지 체크 후 텍스트만 분리

                        if not text or not str(text).strip():
                            continue
    
                        # crop 내 좌표 → 페이지 원본 좌표로 역변환
                        xs = [p[0] for p in quad]
                        ys = [p[1] for p in quad]
    
                        cell_x0 = min(xs) + x0
                        cell_y0 = min(ys) + y0
                        cell_x1 = max(xs) + x0
                        cell_y1 = max(ys) + y0
    
                        bbox = BoundingBox(
                            l=cell_x0 / scale_x,
                            t=cell_y0 / scale_y,
                            r=cell_x1 / scale_x,
                            b=cell_y1 / scale_y,
                            coord_origin=CoordOrigin.TOPLEFT,
                        )
    
                        cell = TextCell(
                            index=0, 
                            text=str(text),
                            orig=str(text),
                            from_ocr=True,
                            rect=BoundingRectangle.from_bounding_box(bbox),
                        )
                        all_ocr_cells.append(cell)

            # 부모 클래스의 post_process_cells로 기존 셀과 합치기
            self.post_process_cells(all_ocr_cells, page)

            yield page



# 플러그인 등록 함수 (v2.27.0+ pluggy 방식)
def docling_register():
    return {
        "ocr_engines": [PaddleOcrModel],
    }