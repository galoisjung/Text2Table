"""
preset_library.py

[C 단계] 프리셋 라이브러리

지금까지의 설계 논의에서 도출한 결론:
- row_unit(행이 무엇을 의미하는가)은 '초점 단위'와 '시간구조' 두 축의 조합으로
  거의 결정되고, 이 두 축은 서로 독립이 아니라 하나가 다른 하나를 규정하는
  lookup 관계다 (곱해지는 축이 아니라 매핑 함수).
- 유일하게 추가 분기가 필요한 지점은 '다중개체 + 비시계열' 칸으로, 여기서
  속성이 1개뿐이면 listing, 여러 개면 horizontal_relational_static으로 갈린다.
- 이렇게 나온 base preset은 5개로 수렴한다. 이 개수는 문서 도메인이 늘어나도
  바뀌지 않는다 (도메인은 core_columns의 '내용'이 아니라 extension_columns의
  '후보'로만 영향을 준다 -- 그건 A 단계의 책임).
- 표 병합 셀/다단헤더 같은 구조적 복잡도(concise/nested/multivalued/split)는
  별도의 preset이 아니라 전처리 플래그로 다룬다 (여기서는 자리만 마련해둠).

이 파일은 순수 데이터 + 결정 로직만 담고, LLM 호출은 1_preset_classifier.py 에서 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    name: str
    role: str  # "row_key" | "row_key_entity" | "row_key_temporal" | "value"
    desc: str = ""


@dataclass
class Preset:
    preset_id: str
    shape_literature_term: str  # 기존 web-table 연구의 대응 용어
    focus: str  # "single_subject" | "multi_entity"
    time_structure: str  # "timeseries" | "non_timeseries"
    attribute_count: str  # "single_attribute" | "multi_attribute" | "not_applicable"
    row_unit_desc: str
    core_columns: list[ColumnSpec]
    extension_budget_default: int | None  # None = 확장 불가(구조상 의미 없음)
    validation_rule: str
    few_shot: list[dict] = field(default_factory=list)


PRESETS: dict[str, Preset] = {

    "vertical_entity": Preset(
        preset_id="vertical_entity",
        shape_literature_term="Vertical (Entity) Table — Lautert et al.",
        focus="single_subject",
        time_structure="non_timeseries",
        attribute_count="not_applicable",
        row_unit_desc="행 = 속성-값 쌍 1개 (하나의 주제/개체에 대한 속성 나열)",
        core_columns=[
            ColumnSpec("속성명", "row_key", "원문 레이블을 정규화한 이름"),
            ColumnSpec("값", "value"),
        ],
        extension_budget_default=None,  # 열이 늘어날 구조가 아님
        validation_rule="속성명은 표 내에서 서로 중복되지 않아야 함",
        few_shot=[{
            "text": "평가계약일자는 2024년 6월 19일이며",
            "row": "| 평가계약일자 | 2024년 6월 19일 |",
        }],
    ),

    "event_timeline": Preset(
        preset_id="event_timeline",
        shape_literature_term="(단일주제 + 시계열, 문헌에 직접 대응 용어 없음 — Relational의 특수형)",
        focus="single_subject",
        time_structure="timeseries",
        attribute_count="not_applicable",
        row_unit_desc="행 = 사건/시점 1개 (하나의 주제가 시간 순서로 전개)",
        core_columns=[
            ColumnSpec("시점", "row_key_temporal"),
            ColumnSpec("사건", "value"),
        ],
        extension_budget_default=4,
        validation_rule="시점은 시간순 정렬 가능해야 함 (날짜/순번 등)",
        few_shot=[{
            "text": "2024년 3월 첫째 주, 주인공은 회사를 그만두고 유럽으로 떠났다.",
            "row": "| 2024-03 | 퇴사 후 유럽으로 출국 |",
        }],
    ),

    "horizontal_relational_static": Preset(
        preset_id="horizontal_relational_static",
        shape_literature_term="Horizontal Relational Table — Lehmberg et al.",
        focus="multi_entity",
        time_structure="non_timeseries",
        attribute_count="multi_attribute",
        row_unit_desc="행 = 개체 1개, 열 = 그 개체의 여러 속성",
        core_columns=[
            ColumnSpec("개체명", "row_key"),
        ],
        extension_budget_default=6,  # 1차 스캔 결과에 따라 동적으로 조정 권장
        validation_rule="개체명은 문서 내에서 고유해야 함 (별칭 통합 필요)",
        few_shot=[{
            "text": "양도인인 주식회사 에이피에스(대표이사 정기로, 설립연월일 1996년 08월 29일)는...",
            "row": "| 주식회사 에이피에스 | 정기로 | 1996-08-29 | ... |",
        }],
    ),

    "horizontal_relational_timeseries": Preset(
        preset_id="horizontal_relational_timeseries",
        shape_literature_term="Horizontal Relational Table + temporal core column",
        focus="multi_entity",
        time_structure="timeseries",
        attribute_count="multi_attribute",
        row_unit_desc="행 = 개체×시점 1건 (거래/기록 1건), 시간순 정렬",
        core_columns=[
            ColumnSpec("시점", "row_key_temporal"),
            ColumnSpec("개체명", "row_key_entity"),
        ],
        extension_budget_default=6,
        validation_rule="(시점, 개체명) 복합키가 유일해야 함 (동일 시점에 여러 개체 가능)",
        few_shot=[{
            "text": "2024-06-17에 웨스트라이즈는 9000000주를 40500원에 양수도하였다.",
            "row": "| 2024-06-17 | 웨스트라이즈 | 9000000 | 40500 |",
        }],
    ),

    "listing": Preset(
        preset_id="listing",
        shape_literature_term="Vertical Listing — Crestan & Pantel",
        focus="multi_entity",
        time_structure="non_timeseries",
        attribute_count="single_attribute",
        row_unit_desc="행 = 개체 1개, 열은 [개체명, 단일 속성값] 2개로 고정",
        core_columns=[
            ColumnSpec("항목", "row_key"),
            ColumnSpec("값", "value"),
        ],
        extension_budget_default=None,  # 속성이 정의상 1개뿐이라 확장 불필요
        validation_rule="값 컬럼은 보통 통제 어휘(예: 이행함/미이행)에 가까움. "
                         "자유 서술이면 horizontal_relational_static 재검토",
        few_shot=[{
            "text": "점검항목 1 '정보의 원천'에 대해서는 점검결과는 이행함이다.",
            "row": "| 정보의 원천 | 이행함 |",
        }],
    ),
}


def select_preset(
    focus: str,
    time_structure: str,
    attribute_count: str = "not_applicable",
) -> str | None:
    """
    분류 축(focus, time_structure, attribute_count) -> preset_id 결정 트리.

    핵심 설계 원칙: focus와 time_structure는 서로 곱해지는 독립 축이 아니라
    row_unit을 정하는 하나의 lookup이다. attribute_count는 오직
    (multi_entity, non_timeseries) 칸에서만 추가로 분기하는 2차 축이다.
    """
    if focus == "single_subject":
        return "event_timeline" if time_structure == "timeseries" else "vertical_entity"

    if focus == "multi_entity":
        if time_structure == "timeseries":
            return "horizontal_relational_timeseries"
        # non_timeseries
        if attribute_count == "single_attribute":
            return "listing"
        if attribute_count == "multi_attribute":
            return "horizontal_relational_static"
        return None  # attribute_count 미상 -> 판단 불가, 폴백 필요

    return None  # focus 자체가 불명 -> 폴백 필요


def get_preset(preset_id: str) -> Preset | None:
    return PRESETS.get(preset_id)
