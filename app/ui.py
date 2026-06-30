"""Streamlit product UI for the scientific knowledge graph demo."""

from __future__ import annotations

import os
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.ui_helpers import (  # noqa: E402
    active_document_changes,
    build_compact_metrics,
    diagnostics_to_safe_summary,
    documents_to_rows,
    evidence_to_rows,
    evidence_to_user_rows,
    facts_to_rows,
    facts_to_user_rows,
    format_answer_markdown,
    graph_to_display,
    graph_to_interactive_html,
    no_exact_match_warning,
    subgraph_to_tables,
)
from app.graph.answer_graph import answer_graph_to_html, build_answer_graph  # noqa: E402


API_BASE = os.getenv("API_BASE", "http://localhost:8000")

PRESET_TITLE_TO_ID = {
    "Лучший ответ": "expert_max",
    "Строгая проверка": "strict_audit",
    "Офлайн-режим": "offline_reliable",
}
DEFAULT_PRESET_TITLE = "Лучший ответ"
DEFAULT_PRESET_ID = PRESET_TITLE_TO_ID[DEFAULT_PRESET_TITLE]

EXAMPLE_QUESTIONS = [
    "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
    "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?",
    "Что уже делали по ВТ6?",
    "Сравни ВТ6 и 7075-T6 по прочности.",
    "Какие пробелы есть по коррозионной стойкости?",
]


def api_get(path: str, params: dict[str, Any] | None = None, *, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(f"{API_BASE}{path}", params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    response = requests.post(f"{API_BASE}{path}", params=params or {}, json=json_body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_patch(path: str, json_body: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    response = requests.patch(f"{API_BASE}{path}", json=json_body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def preset_id_for_title(title: str | None) -> str:
    return PRESET_TITLE_TO_ID.get(str(title or DEFAULT_PRESET_TITLE), DEFAULT_PRESET_ID)


def build_ask_payload(question: str, top_k: int = 12, preset_id: str = DEFAULT_PRESET_ID) -> dict[str, Any]:
    return {"question": question, "top_k": top_k, "preset_id": preset_id}


def ask_api(question: str, top_k: int = 12, preset_id: str = DEFAULT_PRESET_ID) -> dict[str, Any]:
    return api_post("/ask", json_body=build_ask_payload(question, top_k=top_k, preset_id=preset_id), timeout=90)


def _safe_get(path: str, params: dict[str, Any] | None = None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return api_get(path, params=params)
    except Exception as exc:
        return default or {"error": str(exc)}


def _selected_preset_id() -> str:
    return preset_id_for_title(st.session_state.get("preset_title", DEFAULT_PRESET_TITLE))


def _dataframe(rows: list[dict[str, Any]], *, empty: str) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info(empty)


def _answer_graph_key(payload: dict[str, Any]) -> str:
    identity = {
        "question": st.session_state.get("last_question", ""),
        "status": payload.get("status"),
        "answer_mode": payload.get("answer_mode"),
        "analytical_intent": payload.get("analytical_intent"),
        "constraints": payload.get("constraints"),
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _answer_graph_modal_state_key(answer_key: str) -> str:
    return f"answer_graph_modal_open_{answer_key}"


def _ensure_answer_graph_modal_state(answer_key: str) -> str:
    state_key = _answer_graph_modal_state_key(answer_key)
    st.session_state.setdefault("answer_graph_modal_open", False)
    st.session_state.setdefault(state_key, False)
    return state_key


def _open_answer_graph_modal(answer_key: str) -> None:
    st.session_state["answer_graph_modal_open"] = True
    st.session_state[_answer_graph_modal_state_key(answer_key)] = True


def _close_answer_graph_modal(answer_key: str) -> None:
    st.session_state["answer_graph_modal_open"] = False
    st.session_state[_answer_graph_modal_state_key(answer_key)] = False


def _render_graph_header(answer_key: str) -> None:
    _ensure_answer_graph_modal_state(answer_key)
    title_col, action_col = st.columns([0.58, 0.42], vertical_alignment="center")
    with title_col:
        st.subheader("Интерактивный связанный граф")
    with action_col:
        st.button(
            "Развернуть карту",
            key=f"open_answer_graph_modal_{answer_key}",
            on_click=_open_answer_graph_modal,
            args=(answer_key,),
            use_container_width=True,
        )


def _render_interactive_graph(payload: dict[str, Any], answer_graph: Any, answer_key: str) -> None:
    components.html(
        answer_graph_to_html(answer_graph, container_id=f"answerGraphCompact_{answer_key}"),
        height=560,
        scrolling=False,
    )
    if st.session_state.get(_answer_graph_modal_state_key(answer_key)):
        _render_large_answer_graph(answer_graph, answer_key)
    with st.expander("Технический подграф"):
        st.caption("Raw subgraph для аудита. Основная карта выше агрегирует только смысловую цепочку ответа.")
        components.html(graph_to_interactive_html(graph_to_display(payload), max_nodes=20, max_edges=30), height=420, scrolling=False)
        nodes, edges = subgraph_to_tables(graph_to_display(payload))
        _dataframe(nodes, empty="Nodes отсутствуют.")
        _dataframe(edges, empty="Edges отсутствуют.")


def _render_answer_graph_modal_css() -> None:
    st.markdown(
        """
<style>
div[data-testid="stModal"] div[role="dialog"],
div[data-testid="stDialog"] div[role="dialog"] {
  width: min(85vw, 1500px) !important;
  max-width: min(85vw, 1500px) !important;
}
div[data-testid="stModal"] div[role="dialog"] > div,
div[data-testid="stDialog"] div[role="dialog"] > div {
  max-height: 90vh;
}
div[data-testid="stModal"],
div[data-testid="stDialog"] {
  background: rgba(15, 23, 42, 0.34);
}
div[data-testid="stModal"] button[aria-label="Close"],
div[data-testid="stDialog"] button[aria-label="Close"] {
  display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_large_answer_graph(answer_graph: Any, answer_key: str) -> None:
    dialog = getattr(st, "dialog", None)
    if dialog is not None:
        try:
            decorator = dialog("Крупная карта ответа", width="large")
        except TypeError:
            decorator = dialog("Крупная карта ответа")

        @decorator
        def _large_graph_dialog() -> None:
            _render_answer_graph_modal_css()
            _, close_col = st.columns([0.9, 0.1])
            with close_col:
                if st.button("×", key=f"close_answer_graph_modal_{answer_key}", help="Закрыть"):
                    _close_answer_graph_modal(answer_key)
                    st.rerun()
            components.html(
                answer_graph_to_html(
                    answer_graph,
                    render_height=820,
                    render_width=1500,
                    container_id=f"answerGraphExpanded_{answer_key}",
                ),
                height=860,
                scrolling=False,
            )

        _large_graph_dialog()
        return

    with st.container(border=True):
        close_col, _ = st.columns([0.2, 0.8])
        with close_col:
            if st.button("Закрыть", key=f"close_answer_graph_modal_inline_{answer_key}"):
                _close_answer_graph_modal(answer_key)
                st.rerun()
        components.html(
            answer_graph_to_html(
                answer_graph,
                render_height=820,
                render_width=1500,
                container_id=f"answerGraphExpanded_{answer_key}",
            ),
            height=860,
            scrolling=False,
        )


def _render_answer(payload: dict[str, Any]) -> None:
    warning = no_exact_match_warning(payload)
    if warning:
        st.warning(warning)
    st.subheader("Ответ")
    st.markdown(format_answer_markdown(payload))
    metrics = build_compact_metrics(payload)
    if metrics:
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.metric(label, value)


def _render_details(payload: dict[str, Any]) -> None:
    with st.expander("Проверенные факты"):
        _dataframe(facts_to_user_rows(payload), empty="Проверенные facts для этого вопроса не найдены.")
        with st.container(border=True):
            st.caption("Raw facts для аудита")
            _dataframe(facts_to_rows(payload), empty="Raw facts отсутствуют.")

    with st.expander("Источники и evidence"):
        rows = evidence_to_user_rows(payload)
        if rows:
            _dataframe(rows, empty="")
        else:
            st.warning("Для найденных фактов отсутствуют цитаты-источники. Это снижает доверие к ответу.")
        with st.container(border=True):
            st.caption("Raw evidence/source rows")
            _dataframe(evidence_to_rows(payload), empty="Evidence/source rows отсутствуют.")

    with st.expander("История решений"):
        _dataframe(payload.get("decision_history") or [], empty="История решений не возвращена для этого вопроса.")

    with st.expander("Пробелы в данных"):
        _dataframe(payload.get("data_gaps") or payload.get("gaps") or [], empty="Пробелы не найдены.")

    with st.expander("Частичные совпадения"):
        partial = payload.get("partial_matches") or {}
        rendered = False
        for key, rows in partial.items():
            if rows:
                rendered = True
                st.caption(str(key))
                _dataframe(rows, empty="")
        if not rendered:
            st.info("Частичные совпадения отсутствуют или не требовались.")

    with st.expander("Диагностика"):
        response_diagnostics = payload.get("diagnostics") or {}
        st.json(
            {
                "selected_preset_from_ui": st.session_state.get("last_selected_preset_from_ui"),
                "request_payload": st.session_state.get("last_request_payload"),
                "response_diagnostics_preset_id": response_diagnostics.get("preset_id"),
            }
        )
        st.json(diagnostics_to_safe_summary(payload))
        st.json(
            {
                "constraints": payload.get("constraints"),
                "graph_context": payload.get("graph_context"),
                "diagnostics": payload.get("diagnostics"),
                "retrieval": payload.get("retrieval"),
                "llm": payload.get("llm"),
                "technical_answer": payload.get("technical_answer"),
            }
        )


def _run_question(question: str, preset_id: str) -> None:
    if not question.strip():
        st.warning("Введите исследовательский вопрос.")
        return
    request_payload = build_ask_payload(question.strip(), preset_id=preset_id)
    with st.spinner("Ищу ответ в графе, evidence и активных документах..."):
        try:
            payload = api_post("/ask", json_body=request_payload, timeout=90)
        except Exception as exc:
            st.error(f"Ошибка /ask: {exc}")
            return
    st.session_state["last_question"] = question.strip()
    st.session_state["last_request_payload"] = request_payload
    st.session_state["last_selected_preset_from_ui"] = preset_id
    st.session_state["last_answer_payload"] = payload
    st.session_state["answer_graph_modal_open"] = False


def _render_document_controls() -> None:
    with st.expander("Документы", expanded=False):
        st.markdown("**Загрузка документов**")
        uploaded_files = st.file_uploader(
            "Файлы PDF/DOCX/PPTX/XLSX/CSV/HTML/TXT/MD",
            type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "csv", "txt", "md"],
            accept_multiple_files=True,
        )
        upload_col, refresh_col = st.columns([0.55, 0.45])
        with upload_col:
            upload_mode = st.radio(
                "После загрузки",
                ["Только загрузить документы", "Загрузить и обновить граф"],
                horizontal=True,
            )
            if st.button("Загрузить в базу", type="primary") and uploaded_files:
                files_param = [
                    ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
                    for file in uploaded_files
                ]
                with st.spinner("Парсинг документов и сохранение chunks..."):
                    response = requests.post(f"{API_BASE}/ingest/documents", files=files_param, timeout=160)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Документы загружены.")
                    _render_ingestion_result(result)
                    if upload_mode == "Загрузить и обновить граф":
                        _refresh_graph()
                else:
                    st.error(response.text)
        with refresh_col:
            st.markdown("**Граф и active corpus**")
            st.caption("Обновляет retrieval/fallback cache по активным документам; Neo4j sync выполняется, если backend доступен.")
            if st.button("Обновить граф по активным документам"):
                _refresh_graph()

        st.divider()
        st.markdown("**Управление документами**")
        docs_payload = _safe_get("/documents", default=[])
        rows = documents_to_rows(docs_payload)
        if rows:
            edited = st.data_editor(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Активен": st.column_config.CheckboxColumn("Активен", help="Включает документ в retrieval/graph QA."),
                    "doc_id": st.column_config.TextColumn("doc_id", disabled=True, width="small"),
                },
                disabled=["Документ", "Тип", "Chunks", "Дата загрузки", "Parser", "Blocks", "Tables", "doc_id"],
                key="documents_active_editor",
            )
            changes = active_document_changes(rows, edited)
            if changes:
                for doc_id, active in changes:
                    try:
                        api_patch(f"/documents/{doc_id}/active", {"active": active})
                        st.success("Документ включён." if active else "Документ выключен.")
                    except Exception as exc:
                        st.error(f"Не удалось обновить документ {doc_id}: {exc}")
                        return
                st.rerun()
        else:
            st.info("Документы пока не загружены.")
        if rows:
            labels = [f"{row['Документ']} | {row['doc_id']}" for row in rows]
            selected = st.selectbox("Документ для просмотра metadata", labels)
            selected_id = selected.rsplit(" | ", 1)[-1]
            with st.container(border=True):
                st.caption("Metadata выбранного документа")
                doc_items = docs_payload.get("items", []) if isinstance(docs_payload, dict) else docs_payload
                original = next((item for item in doc_items if item.get("doc_id") == selected_id), {})
                st.json(original)


def _render_ingestion_result(result: dict[str, Any]) -> None:
    items = result.get("ingested")
    if isinstance(items, dict):
        items = [items]
    rows = []
    for item in items or []:
        diagnostics = item.get("parser_diagnostics") or {}
        rows.append(
            {
                "filename": item.get("filename") or item.get("url"),
                "status": item.get("status"),
                "parser": item.get("parser"),
                "chunks": item.get("chunks"),
                "blocks": diagnostics.get("blocks_count"),
                "tables": diagnostics.get("tables_count"),
                "images": diagnostics.get("images_count"),
                "parser_error": item.get("parser_error"),
            }
        )
    _dataframe(rows, empty="Нет строк результата ingestion.")


def _refresh_graph() -> None:
    with st.spinner("Обновляю active corpus и graph projection..."):
        try:
            result = api_post("/graph/refresh", timeout=160)
        except Exception as exc:
            st.error(f"Ошибка обновления графа: {exc}")
            return
    st.success("Граф/индекс обновлены.")
    st.json(result)


def _render_question_block(preset_id: str) -> None:
    with st.expander("Примеры вопросов"):
        selected = st.selectbox("Выбрать пример", [""] + EXAMPLE_QUESTIONS)
        if selected and st.button("Подставить пример"):
            st.session_state["question_input"] = selected
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    st.text_area("Введите исследовательский вопрос", key="question_input", height=90)
    if st.button("Найти ответ", type="primary"):
        _run_question(st.session_state.get("question_input", ""), preset_id)


def _render_sidebar_diagnostics() -> None:
    with st.sidebar:
        st.subheader("Состояние")
        health = _safe_get("/health", default={"status": "unavailable"})
        catalog = health.get("catalog") or {}
        st.write(f"API: {health.get('status')}")
        st.write(f"KG backend: {health.get('kg_backend_active', 'unknown')}")
        st.write(f"Активных документов: {catalog.get('active_documents', catalog.get('documents', 0))}")
        st.write(f"Активных chunks: {catalog.get('active_chunks', catalog.get('chunks', 0))}")
        llm = health.get("llm") or {}
        st.write(f"LLM: {'готов' if llm.get('ready') else 'не готов'}")
        with st.expander("Advanced diagnostics"):
            st.caption("LLM diagnostics")
            st.write(f"Provider: {llm.get('provider') or 'не задан'}")
            st.write(f"Base URL: {llm.get('base_url') or 'не задан'}")
            st.write(f"Model: {llm.get('model') or 'не задана'}")
            if not llm.get("ready"):
                st.warning(str(llm.get("last_error") or "LLM не готов."))
            st.json(health)
            if st.button("Проверить LLM"):
                try:
                    st.json(api_post("/system/test-llm", timeout=60))
                except Exception as exc:
                    st.error(f"LLM test failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Scientific Knowledge Graph", layout="wide")
    st.title("Scientific Knowledge Graph")
    st.caption(
        "Система связывает документы, эксперименты, материалы, режимы, свойства, оборудование, лаборатории, выводы и пробелы в данных."
    )

    _render_sidebar_diagnostics()
    if st.session_state.get("preset_title") not in PRESET_TITLE_TO_ID:
        st.session_state["preset_title"] = DEFAULT_PRESET_TITLE
    st.radio("Режим работы", list(PRESET_TITLE_TO_ID), horizontal=True, key="preset_title")
    preset_id = _selected_preset_id()

    _render_document_controls()
    _render_question_block(preset_id)

    payload = st.session_state.get("last_answer_payload")
    if not payload:
        st.info("Загрузите документы при необходимости, затем задайте вопрос.")
        return

    left, right = st.columns([0.56, 0.44], gap="large")
    with left:
        _render_answer(payload)
    with right:
        answer_graph = build_answer_graph(payload)
        answer_key = _answer_graph_key(payload)
        _render_graph_header(answer_key)
        _render_interactive_graph(payload, answer_graph, answer_key)
    _render_details(payload)


if __name__ == "__main__":
    main()
