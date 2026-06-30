# Название странное чтобы никто не нашел в поиске и не украл идеи

Проект для хакатона: система для анализа научно-технических документов, построения knowledge graph и ответа на исследовательские вопросы по материалам, экспериментам, режимам обработки, свойствам, источникам и пробелам в данных.

Система умеет:

* загружать документы: PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD;
* извлекать материалы, режимы обработки, свойства, измерения, источники и пробелы;
* строить локальный knowledge graph;
* использовать Neo4j как графовую базу данных;
* отвечать на вопросы через API `/ask`;
* показывать ответ, факты, источники, диагностику и граф в Streamlit UI;
* работать через Docker.

---

## Быстрый запуск

### 1. Установить

Нужно установить:

* Git;
* Docker Desktop.

---

### 2. Скачать проект

```powershell
git clone https://github.com/Leo-Daiser/Default-Xak.git
cd Default-Xak
```

---

### 3. Проверить `.env`

В приватный репозиторий уже добавлен файл `.env` с настройками для командного запуска.

Проверь, что он есть:

```powershell
dir .env
```

Внутри должны быть настройки Neo4j и выбранного LLM provider. Для Mistral ключ
вставляется только в локальный `.env`, не в `.env.example` и не в код:

```env
KG_BACKEND=auto
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<neo4j-password>
NEO4J_DATABASE=neo4j

LLM_ENABLED=true
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-small-latest

# optional fallback
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
```

---

### 4. Запустить проект

```powershell
docker compose --profile full up -d --build
```

Первый запуск может занять несколько минут.

---

### 5. Открыть интерфейсы

Streamlit UI:

```text
http://localhost:8501
```

API docs:

```text
http://localhost:8000/docs
```

Neo4j Browser:

```text
http://localhost:7474
```

Neo4j login:

```text
user: neo4j
password: значение NEO4J_PASSWORD из вашего .env
```

---

## Проверка запуска

Проверить контейнеры:

```powershell
docker compose ps
```

Должны быть запущены контейнеры:

```text
api
ui
neo4j
qdrant
```

Проверить API:

```powershell
curl http://localhost:8000/health
```

Для Docker-демо с доступным Neo4j ожидаются поля:

```json
{
  "kg_backend_configured": "auto",
  "kg_backend_active": "neo4j",
  "neo4j_available": true,
  "neo4j_error": "",
  "kg_backend_decision": {
    "selected": "neo4j"
  }
}
```

Проверить подключение к Neo4j из API-контейнера:

```powershell
docker compose exec api python scripts/check_neo4j_connection.py
```

Ожидаемый результат:

```json
{
  "available": true,
  "reason": "Neo4j connection check succeeded with RETURN 1"
}
```

---

## Пересборка графа

После загрузки новых документов или если нужно обновить граф:

```powershell
docker compose exec api python scripts/init_neo4j_schema.py
docker compose exec api python scripts/sync_graph_to_neo4j.py
docker compose exec api python scripts/smoke_neo4j_graph.py
```

---

## Как пользоваться UI

Открыть:

```text
http://localhost:8501
```

Основной сценарий:

1. загрузить документы;
2. проверить список документов;
3. оставить активными нужные документы;
4. обновить граф;
5. задать исследовательский вопрос;
6. посмотреть ответ, граф, факты, источники и диагностику.

Примеры вопросов:

```text
Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?
Сравни ВТ6 и 7075-T6 по прочности.
Что уже делали по ВТ6?
Какие пробелы есть по коррозионной стойкости?
Какая лаборатория занималась 12Х18Н10Т?
```

---

## Режимы работы

В UI есть три режима.

### Лучший ответ

Основной режим для демо. Даёт человекочитаемый ответ, ограничения, вывод, источники и граф.

### Строгая проверка

Аудиторский режим. Проверяет, есть ли точная цепочка в графе:

```text
материал → эксперимент → режим → измерение → свойство
```

Подходит для проверки, что система не придумывает факты.

### Офлайн-режим

Работает через локальный fallback без обязательного Neo4j/LLM. Нужен как запасной режим.

---

## API

Главный endpoint:

```text
POST /ask
```

Пример:

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Что уже делали по ВТ6?\",\"preset_id\":\"expert_max\"}"
```

Health:

```text
GET /health
```

Документация API:

```text
http://localhost:8000/docs
```

---

## Структура проекта

```text
app/          основной код API, UI, retrieval, graph, extraction
scripts/      служебные скрипты
evaluation/   eval-скрипты
tests/        тесты
demo_data/    демонстрационные документы
docs/         архитектура и runbook
```

---

## Важные команды

Остановить проект:

```powershell
docker compose down
```

Перезапустить:

```powershell
docker compose up -d
```

Полностью пересобрать:

```powershell
docker compose --profile full up -d --build
```

Посмотреть логи API:

```powershell
docker compose logs api --tail=100
```

Посмотреть логи UI:

```powershell
docker compose logs ui --tail=100
```

Посмотреть логи Neo4j:

```powershell
docker compose logs neo4j --tail=100
```

Полностью сбросить контейнеры и volumes:

```powershell
docker compose down -v
docker compose --profile full up -d --build
```

---

## Если что-то не работает

### UI не открывается

```powershell
docker compose ps
docker compose logs ui --tail=100
```

### API не открывается

```powershell
docker compose ps
docker compose logs api --tail=100
```

### Neo4j не подключается

Проверь `.env`:

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<neo4j-password>
NEO4J_DATABASE=neo4j
```

Проверка:

```powershell
docker compose exec api python scripts/check_neo4j_connection.py
```

Если `KG_BACKEND=auto`, `/health` делает принудительную короткую повторную проверку Neo4j. Поэтому после старта Neo4j поле `kg_backend_active` должно перейти в `neo4j`, если прямой `RETURN 1` из API-контейнера успешен. Пароль в `/health` не выводится; проверяется только `neo4j_password_configured`.

### Как подключить Mistral

1. Создай API key в Mistral Studio / La Plateforme.
2. Скопируй ключ в локальный `.env`. Не вставляй ключ в `.env.example`, README или код.
3. Укажи provider и модель:

```env
LLM_ENABLED=true
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_BASE_URL=https://api.mistral.ai/v1
MISTRAL_MODEL=mistral-small-latest
MISTRAL_TIMEOUT_SECONDS=60
MISTRAL_MAX_TOKENS=1200
MISTRAL_TEMPERATURE=0.2
```

Модель можно поменять на доступную в твоём аккаунте. Проверь список доступных
моделей и лимиты в Mistral Studio / API Limits.

Перезапусти контейнеры:

```powershell
docker compose down
docker compose --profile full up -d --build
```

Проверь подключение:

```powershell
docker compose exec api python scripts/check_mistral_connection.py
curl http://localhost:8000/health
```

В `/health` должны быть безопасные поля без секретов:

```json
"llm_provider_configured": "mistral",
"llm_provider_active": "mistral",
"mistral_model": "mistral-small-latest",
"mistral_api_key_configured": true,
"llm_ready": true
```

### OpenRouter / LLM не работает

Проверь `.env`:

```env
LLM_ENABLED=true
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=...
LLM_MODEL=openrouter/free
```

Проверить health:

```powershell
curl http://localhost:8000/health
```

В блоке `llm` должно быть:

```json
"provider": "openrouter",
"ready": true
```

При `LLM_PROVIDER=auto` приложение выбирает Mistral, если задан
`MISTRAL_API_KEY`, иначе OpenRouter при наличии `OPENROUTER_API_KEY`, иначе
использует offline/template fallback.

### Как включить local hybrid embeddings

Phase 1 embeddings не заменяют Neo4j и не создают факты. Они только расширяют
candidate retrieval: BM25 остаётся включённым, dense retrieval добавляет
семантически похожие chunks, а затем graph/extraction/constraints проверяют
структурированные факты.

Для Docker нужно явно установить optional dependency:

```env
EXTRA_REQUIREMENTS=requirements-embeddings.txt
RETRIEVAL_MODE=hybrid
ENABLE_LOCAL_EMBEDDINGS=true
EAGER_LOCAL_EMBEDDINGS=false
DIRECT_QDRANT_PROJECTION=false
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Перезапуск:

```powershell
docker compose down
docker compose --profile full up -d --build
```

Проверка:

```powershell
curl http://localhost:8000/health
python evaluation/eval_semantic_retrieval.py
```

В `/health.retrieval` смотри:

```json
"retrieval_mode": "hybrid",
"bm25_ready": true,
"embedding_dependency_available": true,
"local_embeddings_enabled": true,
"local_embeddings_ready": true,
"hybrid_dense_enabled": true,
"hybrid_degraded_reason": ""
```

Если `sentence-transformers` или модель недоступны, приложение не падает:
`effective_retrieval_mode` станет `hybrid_degraded_to_bm25`, а причина будет в
`hybrid_degraded_reason`. Qdrant для Phase 1 не нужен; `DIRECT_QDRANT_PROJECTION`
оставь `false`.

---

## Что не нужно коммитить

В репозиторий не нужно добавлять локальные runtime-файлы:

```text
data/
volumes/
dist/
__pycache__/
*.pyc
*.sqlite3
*.jsonl
logs/
```

Файл `.env` добавлен специально для командного запуска в приватном репозитории. Не переносить его в публичный репозиторий.

---

## Проверки для разработчика

```powershell
docker compose exec api python -m pytest -q
docker compose exec api python evaluation/eval_demo.py
docker compose exec api python evaluation/eval_runtime_presets.py
docker compose exec api python evaluation/eval_answer_quality.py
docker compose exec api python evaluation/eval_ui_product.py
```

---

## Краткая архитектура

```text
Документы
  → парсинг
  → chunks
  → extraction pipeline
  → факты и evidence
  → knowledge graph / Neo4j
  → retrieval + graph queries
  → ответ + источники + граф
```

Neo4j используется как графовая база данных для хранения связей между материалами, экспериментами, режимами, свойствами, измерениями и источниками.

Fallback-режим остаётся рабочим, если Neo4j недоступен.
