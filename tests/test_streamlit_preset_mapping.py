from __future__ import annotations

import app.ui as ui


def test_streamlit_preset_title_mapping_is_stable() -> None:
    assert ui.PRESET_TITLE_TO_ID == {
        "Лучший ответ": "expert_max",
        "Строгая проверка": "strict_audit",
        "Офлайн-режим": "offline_reliable",
    }


def test_streamlit_default_preset_is_expert_max() -> None:
    assert ui.DEFAULT_PRESET_TITLE == "Лучший ответ"
    assert ui.DEFAULT_PRESET_ID == "expert_max"
    assert ui.preset_id_for_title(None) == "expert_max"
    assert ui.preset_id_for_title("unknown") == "expert_max"


def test_streamlit_preset_labels_map_to_expected_ids() -> None:
    assert ui.preset_id_for_title("Лучший ответ") == "expert_max"
    assert ui.preset_id_for_title("Строгая проверка") == "strict_audit"
    assert ui.preset_id_for_title("Офлайн-режим") == "offline_reliable"


def test_streamlit_ask_payload_uses_selected_preset_id() -> None:
    payload = ui.build_ask_payload("Сравни ВТ6 и 7075-T6 по прочности.", top_k=12, preset_id="expert_max")

    assert payload == {
        "question": "Сравни ВТ6 и 7075-T6 по прочности.",
        "top_k": 12,
        "preset_id": "expert_max",
    }


def test_streamlit_ask_api_posts_json_body_with_selected_preset(monkeypatch) -> None:
    captured = {}

    def fake_api_post(path, *, json_body=None, timeout=90, params=None):
        captured.update({"path": path, "json_body": json_body, "timeout": timeout, "params": params})
        return {"status": "ok"}

    monkeypatch.setattr(ui, "api_post", fake_api_post)

    response = ui.ask_api("Что уже делали по ВТ6?", preset_id="strict_audit")

    assert response == {"status": "ok"}
    assert captured["path"] == "/ask"
    assert captured["json_body"]["preset_id"] == "strict_audit"
    assert captured["json_body"]["question"] == "Что уже делали по ВТ6?"


def test_streamlit_demo_questions_cover_demo_strengths() -> None:
    assert ui.EXAMPLE_QUESTIONS == [
        "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
        "Сравни ВТ6 и 7075-T6 по прочности.",
        "Какие есть противоречия или неоднородные данные по прочности?",
        "Какие пробелы в данных найдены?",
        "Найди evidence по прочности 7075-T6 после aging.",
    ]

    hints = " ".join(ui.DEMO_QUESTION_HINTS.values())
    for expected in ["exact graph query", "normalized units", "conflict detection", "DataGap", "hybrid retrieval"]:
        assert expected in hints
