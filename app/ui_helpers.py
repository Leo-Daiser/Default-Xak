"""Formatting helpers for the Streamlit product UI."""

from __future__ import annotations

import html
import math
import re
from typing import Any


INTERNAL_ID_RE = re.compile(r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+)\b")


def format_answer_markdown(payload: dict[str, Any]) -> str:
    """Return the main answer without exposing raw JSON or missing values."""

    answer = str(payload.get("answer") or "").strip()
    return answer or "Ответ не сформирован."


def format_status_badge(payload: dict[str, Any]) -> str:
    """Return a compact status label."""

    status = str(payload.get("status") or "unknown")
    mapping = {
        "ok": "Найдены подтверждённые данные",
        "partial": "Найден частичный контекст",
        "no_exact_match": "Точного факта нет",
        "error": "Ошибка",
    }
    return mapping.get(status, status)


def facts_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return display-safe raw fact rows."""

    rows = payload.get("facts") or []
    return [row if isinstance(row, dict) else {"value": row} for row in rows]


def facts_to_user_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fact rows with user-oriented column names."""

    rows = payload.get("primary_facts") or payload.get("facts") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "Материал": row.get("material"),
                "Режим": row.get("regime"),
                "Свойство": row.get("property"),
                "Значение": row.get("value") if row.get("value") is not None else row.get("raw_value"),
                "Ед.": row.get("unit"),
                "Эффект": _effect_label(row.get("effect")),
                "Оборудование": _join(row.get("equipment")),
                "Лаборатория": _join(row.get("laboratory") or row.get("laboratories")),
                "Experiment ID": row.get("experiment_id"),
                "Chunk ID": row.get("source_chunk_id") or row.get("chunk_id"),
            }
        )
    return result


def evidence_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return display-safe evidence/source rows."""

    evidence = payload.get("evidence") or []
    sources = payload.get("sources") or []
    rows = evidence if evidence else sources
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            result.append({"quote": str(row)})
            continue
        result.append(
            {
                "source_name": row.get("source_name") or row.get("title") or row.get("filename"),
                "doc_id": row.get("document_id") or row.get("doc_id"),
                "chunk_id": row.get("chunk_id"),
                "page": row.get("page") or row.get("page_start"),
                "score": row.get("score"),
                "retrieval_backend": row.get("retrieval_backend"),
                "evidence_type": row.get("evidence_type"),
                "quote": row.get("quote") or "Цитата не сохранена; источник доступен в деталях.",
            }
        )
    return result


def evidence_to_user_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evidence rows with readable column names."""

    return [
        {
            "Источник": row.get("source_name"),
            "Фрагмент": row.get("chunk_id"),
            "Страница": row.get("page"),
            "Тип": row.get("evidence_type") or row.get("retrieval_backend"),
            "Цитата": row.get("quote"),
        }
        for row in evidence_to_rows(payload)
    ]


def subgraph_to_tables(subgraph: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return node and edge tables from a UI-compatible subgraph."""

    if not isinstance(subgraph, dict):
        return [], []
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []
    return (
        [node for node in nodes if isinstance(node, dict)],
        [edge for edge in edges if isinstance(edge, dict)],
    )


def graph_to_display(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a UI-compatible graph payload."""

    subgraph = payload.get("subgraph")
    return subgraph if isinstance(subgraph, dict) else {"nodes": [], "edges": []}


def no_exact_match_warning(payload: dict[str, Any]) -> str | None:
    """Return user-facing trust warning for no-exact-match answers."""

    if payload.get("status") != "no_exact_match":
        return None
    return (
        "Точного факта в графе не найдено. Ниже показаны только частичные "
        "совпадения, evidence и предполагаемый пробел в данных; это не "
        "положительный ответ на исходное сочетание ограничений."
    )


def graph_context_stats(payload: dict[str, Any]) -> dict[str, int]:
    """Return stable graph context counters with defaults."""

    context = payload.get("graph_context") or {}
    return {
        "facts_count": int(context.get("facts_count") or len(payload.get("facts") or [])),
        "sources_count": int(context.get("sources_count") or len(payload.get("sources") or [])),
        "evidence_count": int(context.get("evidence_count") or len(payload.get("evidence") or [])),
        "subgraph_nodes": int(context.get("subgraph_nodes") or len((payload.get("subgraph") or {}).get("nodes") or [])),
        "subgraph_edges": int(context.get("subgraph_edges") or len((payload.get("subgraph") or {}).get("edges") or [])),
    }


def build_compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Return top-row product metrics."""

    stats = graph_context_stats(payload)
    return {
        "Статус": format_status_badge(payload),
        "Факты": stats["facts_count"],
        "Источники": stats["sources_count"],
        "Evidence": stats["evidence_count"],
        "Узлы графа": stats["subgraph_nodes"],
    }


def diagnostics_to_safe_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return diagnostics useful for users without dumping every payload field."""

    diagnostics = payload.get("diagnostics") or {}
    retrieval = payload.get("retrieval") or {}
    return {
        "preset_id": diagnostics.get("preset_id"),
        "preset_title": diagnostics.get("preset_title"),
        "kg_backend_active": retrieval.get("kg_backend_active") or diagnostics.get("kg_backend_active"),
        "answer_mode": payload.get("answer_mode"),
        "analytical_intent": payload.get("analytical_intent"),
        "warnings": diagnostics.get("warnings") or [],
    }


def documents_to_rows(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format /documents payload for document management UI."""

    items = payload.get("items") if isinstance(payload, dict) else payload
    result: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        diagnostics = item.get("parser_diagnostics") or {}
        intelligence = item.get("document_intelligence") or {}
        result.append(
            {
                "Документ": item.get("filename") or item.get("title"),
                "Тип": item.get("source_type") or intelligence.get("source_type") or "file",
                "Chunks": item.get("chunks"),
                "Активен": bool(item.get("active", True)),
                "Дата загрузки": item.get("updated_at") or item.get("created_at"),
                "Parser": item.get("parser"),
                "Blocks": intelligence.get("blocks_count") or diagnostics.get("blocks_count"),
                "Tables": intelligence.get("tables_count") or diagnostics.get("tables_count"),
                "doc_id": item.get("doc_id"),
            }
        )
    return result


def active_document_changes(original_rows: Any, edited_rows: Any) -> list[tuple[str, bool]]:
    """Return changed (doc_id, active) pairs from document data editor rows."""

    original = _rows_from_any(original_rows)
    edited = _rows_from_any(edited_rows)
    original_active = {str(row.get("doc_id")): bool(row.get("Активен")) for row in original if row.get("doc_id")}
    changes: list[tuple[str, bool]] = []
    for row in edited:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        new_active = bool(row.get("Активен"))
        if original_active.get(str(doc_id)) != new_active:
            changes.append((str(doc_id), new_active))
    return changes


def _rows_from_any(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        return list(rows.to_dict("records"))
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def clean_graph_label(node: dict[str, Any]) -> str:
    """Return a short human-readable graph label without internal ids."""

    raw = str(node.get("label") or node.get("name") or node.get("canonical_name") or "").strip()
    node_type = str(node.get("type") or node.get("label_type") or "Entity")
    if not raw or INTERNAL_ID_RE.search(raw) or len(raw) > 80:
        raw = _fallback_node_label(node_type, node)
    raw = raw.replace("PropertyValue", "").replace("SourceChunk", "Источник").replace("Experiment", "Эксперимент").strip(" :\n")
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.replace("effect: increase", "эффект: рост")
    raw = raw.replace("effect: decrease", "эффект: снижение")
    raw = raw.replace("effect: unknown", "эффект не указан")
    return _shorten(raw or node_type, 42)


def graph_to_interactive_html(subgraph: dict[str, Any] | None, *, max_nodes: int = 20, max_edges: int = 30) -> str:
    """Build self-contained interactive SVG graph HTML with zoom/pan/drag."""

    nodes, edges = subgraph_to_tables(subgraph)
    nodes = nodes[:max_nodes]
    allowed = {str(node.get("id")) for node in nodes}
    edges = [edge for edge in edges if str(edge.get("source")) in allowed and str(edge.get("target")) in allowed][:max_edges]
    if not nodes:
        return "<div style='padding:16px;color:#64748b'>Для ответа нет связанного графа.</div>"

    width, height = 860, 520
    center_x, center_y = width / 2, height / 2
    radius_x, radius_y = 300, 180
    positioned: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(nodes):
        angle = 2 * math.pi * idx / max(len(nodes), 1)
        node_id = str(node.get("id"))
        positioned[node_id] = {
            **node,
            "x": center_x + radius_x * math.cos(angle),
            "y": center_y + radius_y * math.sin(angle),
            "display_label": clean_graph_label(node),
        }
    edge_lines = []
    for edge in edges:
        source = positioned.get(str(edge.get("source")))
        target = positioned.get(str(edge.get("target")))
        if not source or not target:
            continue
        label = _shorten(str(edge.get("label") or edge.get("type") or ""), 24)
        edge_lines.append(
            f"<line class='edge' x1='{source['x']:.1f}' y1='{source['y']:.1f}' x2='{target['x']:.1f}' y2='{target['y']:.1f}' />"
            f"<text class='edge-label' x='{(source['x'] + target['x']) / 2:.1f}' y='{(source['y'] + target['y']) / 2:.1f}'>{html.escape(label)}</text>"
        )
    node_items = []
    for idx, (node_id, node) in enumerate(positioned.items()):
        color = _node_color(str(node.get("type") or "Entity"))
        label = html.escape(str(node["display_label"]))
        title = html.escape(f"{node.get('type', 'Entity')}: {node['display_label']}")
        node_items.append(
            f"<g class='node' data-node-id='node-{idx}' transform='translate({node['x']:.1f},{node['y']:.1f})'>"
            f"<title>{title}</title><circle r='26' fill='{color}'></circle>"
            f"<text text-anchor='middle' y='42'>{label}</text></g>"
        )
    return f"""
<div class="kg-graph-wrap">
  <div class="kg-graph-help">Wheel: zoom · drag background: pan · drag node: move</div>
  <svg id="kgGraphSvg" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"></path></marker></defs>
    <g id="kgViewport">{"".join(edge_lines)}{"".join(node_items)}</g>
  </svg>
</div>
<style>
.kg-graph-wrap {{ border:1px solid #d8dee9; border-radius:8px; background:#ffffff; position:relative; }}
.kg-graph-help {{ position:absolute; right:10px; top:8px; font:12px Arial; color:#64748b; z-index:2; background:rgba(255,255,255,.86); padding:2px 6px; border-radius:4px; }}
#kgGraphSvg {{ cursor:grab; touch-action:none; }}
.edge {{ stroke:#64748b; stroke-width:1.2; marker-end:url(#arrow); opacity:.72; }}
.edge-label {{ fill:#475569; font:10px Arial; paint-order:stroke; stroke:#fff; stroke-width:3px; stroke-linejoin:round; }}
.node circle {{ stroke:#334155; stroke-width:1.2; filter: drop-shadow(0 2px 3px rgba(15,23,42,.18)); }}
.node text {{ fill:#0f172a; font:12px Arial; pointer-events:none; }}
.node {{ cursor:move; }}
</style>
<script>
(function() {{
 const svg = document.getElementById('kgGraphSvg');
 const viewport = document.getElementById('kgViewport');
 let state = {{x:0, y:0, scale:1}};
 let drag = null;
 function apply() {{ viewport.setAttribute('transform', `translate(${{state.x}},${{state.y}}) scale(${{state.scale}})`); }}
 svg.addEventListener('wheel', function(ev) {{
   ev.preventDefault();
   const delta = ev.deltaY < 0 ? 1.12 : 0.89;
   state.scale = Math.max(0.25, Math.min(4, state.scale * delta));
   apply();
 }}, {{passive:false}});
 svg.addEventListener('pointerdown', function(ev) {{
   const node = ev.target.closest && ev.target.closest('.node');
   drag = {{kind: node ? 'node' : 'pan', node: node, startX: ev.clientX, startY: ev.clientY, x: state.x, y: state.y}};
   svg.setPointerCapture(ev.pointerId);
 }});
 svg.addEventListener('pointermove', function(ev) {{
   if (!drag) return;
   const dx = ev.clientX - drag.startX, dy = ev.clientY - drag.startY;
   if (drag.kind === 'pan') {{ state.x = drag.x + dx; state.y = drag.y + dy; apply(); }}
   else {{
     const current = drag.node.getAttribute('transform').match(/translate\\(([-0-9.]+),([-0-9.]+)\\)/);
     const baseX = parseFloat(current[1]), baseY = parseFloat(current[2]);
     drag.node.setAttribute('transform', `translate(${{baseX + dx / state.scale}},${{baseY + dy / state.scale}})`);
     drag.startX = ev.clientX; drag.startY = ev.clientY;
   }}
 }});
 svg.addEventListener('pointerup', function(ev) {{ drag = null; try {{ svg.releasePointerCapture(ev.pointerId); }} catch(e) {{}} }});
 apply();
}})();
</script>
"""


def _fallback_node_label(node_type: str, node: dict[str, Any]) -> str:
    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    if node_type == "Material":
        return str(props.get("canonical_name") or "Материал")
    if node_type in {"ProcessRegime", "Regime"}:
        return str(props.get("canonical_name") or "Режим")
    if node_type in {"Property", "Measurement"}:
        value = props.get("value")
        unit = props.get("unit") or ""
        prop = props.get("property") or props.get("canonical_name") or "Свойство"
        return f"{prop} {value or ''} {unit}".strip()
    if node_type == "Experiment":
        return "Эксперимент"
    if node_type in {"Document", "DocumentChunk", "SourceChunk"}:
        return "Источник"
    if node_type == "DataGap":
        return "Пробел в данных"
    return node_type


def _node_color(node_type: str) -> str:
    return {
        "Material": "#bfdbfe",
        "ProcessRegime": "#bbf7d0",
        "Property": "#fde68a",
        "Measurement": "#fed7aa",
        "Experiment": "#ddd6fe",
        "Equipment": "#fecdd3",
        "Laboratory": "#ccfbf1",
        "ResearchTeam": "#ccfbf1",
        "DataGap": "#fecaca",
        "Document": "#e2e8f0",
        "DocumentChunk": "#e2e8f0",
        "SourceChunk": "#e2e8f0",
    }.get(node_type, "#f1f5f9")


def _shorten(value: str, limit: int) -> str:
    value = INTERNAL_ID_RE.sub("", value).strip()
    return value[: limit - 1] + "…" if len(value) > limit else value


def _join(value: Any) -> str | None:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value) if value else None


def _effect_label(effect: Any) -> str | None:
    return {
        "increase": "рост",
        "decrease": "снижение",
        "no_change": "без заметного изменения",
        "unchanged": "без заметного изменения",
        "mixed": "смешанный эффект",
        "unknown": "не указан явно",
        None: None,
        "": None,
    }.get(str(effect), str(effect))
