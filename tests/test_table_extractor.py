from __future__ import annotations

from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk


def _row(text: str, row_id: int = 3) -> Chunk:
    return Chunk(
        chunk_id=f"row-{row_id}",
        doc_id="doc-table",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "experiments.csv", "chunk_kind": "table_row", "row_id": row_id},
    )


def test_csv_like_row_becomes_experiment() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row(
            "material: ВТ6 | regime: annealing | property: tensile strength | "
            "value: 1120 | unit: MPa | equipment: печь SNOL"
        )
    )

    assert len(bundle.experiments) == 1
    experiment = bundle.experiments[0]
    assert experiment.materials[0].canonical_name == "ВТ6"
    assert experiment.regimes[0].canonical_name == "отжиг"
    assert experiment.measurements[0].property_canonical == "прочность"
    assert experiment.measurements[0].unit == "MPa"


def test_russian_column_aliases_work() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("Материал: 12Х18Н10Т | Режим: закалка | Свойство: твёрдость | Значение: 240 | Единица: HV")
    )

    experiment = bundle.experiments[0]
    assert experiment.materials[0].canonical_name == "12Х18Н10Т"
    assert experiment.regimes[0].canonical_name == "закалка"
    assert experiment.measurements[0].property_canonical == "твёрдость"


def test_empty_row_rejected() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(_row("material: | regime: | property: "))

    assert bundle.experiments == []
    assert any(item.reason in {"empty_row", "missing_material"} for item in bundle.rejected_items)


def test_row_index_is_preserved_in_evidence() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("Материал: ВТ6 | Режим: отжиг | Свойство: прочность | Значение: 1120 | Единица: MPa", row_id=7)
    )

    assert bundle.experiments[0].evidence[0].source.row_index == 7


def test_multivalue_cells_are_handled_safely() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("Материал: ВТ6; Ti-6Al-4V | Режим: отжиг | Свойство: прочность | Значение: 1120 | Единица: MPa")
    )

    materials = [item.canonical_name for item in bundle.experiments[0].materials]
    assert materials == ["ВТ6"]
