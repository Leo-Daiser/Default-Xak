# File Corpus Readiness

## 1. Current Format Support

The production ingestion path is `ParserRouter -> chunks -> deterministic extraction -> canonical facts`.

Currently supported extensions are:

- `.pdf`: fallback parser uses `pypdf` text extraction. If optional Docling is installed and `PARSER_BACKEND=auto`, Docling may be used first.
- `.docx`: fallback parser uses `python-docx` paragraphs and tables.
- `.pptx`: fallback parser uses `python-pptx` text frames and tables.
- `.xlsx`: fallback parser uses `pandas.read_excel` and turns rows into table-row chunks.
- `.csv`: fallback parser uses `pandas.read_csv` with delimiter inference.
- `.html` / `.htm`: fallback parser uses BeautifulSoup, removes `script`, `style`, `nav`, `footer`, and extracts text, tables and images.
- `.md` / `.txt`: fallback parser uses line/heading/markdown-table parsing.

Optional parser backends:

- `docling`: useful for richer PDFs, if installed.
- `markitdown`: useful for Office/HTML conversion, if installed.
- OCR is not executed by default. `ENABLE_OCR=false` in resource-efficient profiles.

## 2. Where Data Can Be Lost

- Image-only/scanned PDFs can have zero extracted text with `pypdf`. This must be reported as `ocr_required`, not as a scientific no-data result.
- PDFs with a poor text layer can produce low-density text, broken units and broken material names.
- DOCX/PPTX tables can lose structure when cells contain nested formatting or merged cells.
- CSV delimiter inference can fail on severely malformed legacy files.
- HTML parsing removes navigation/footer noise, but malformed tables can still lose headers.
- Markdown/TXT OCR exports can contain broken line wraps and mixed Cyrillic/Latin unit symbols.

## 3. Text-Layer Quality Detection

The readiness profiler measures:

- `text_chars`;
- estimated pages;
- characters per page;
- `text_density`: `empty`, `very_low`, `low`, `medium`, `high`;
- dirty OCR signals such as split `M Pa`, mixed `МРа`, `НV`, `ВТ 6`, `7075 Т6`, hyphenated line breaks and many short OCR lines.

Low density is a parser/text-layer warning. It is not the same as extraction failure.

## 4. OCR-Required Detection

For PDFs, the parser checks page count and extracted text length.

If a PDF has pages but text length is below `SCANNED_PDF_MIN_TEXT_CHARS`, the document is marked with:

- `parse_status = ocr_required`;
- warning `ocr_required`;
- no claim that scientific facts are absent.

OCR is intentionally not enabled in `economy_core`. The system reports the need for OCR rather than silently hallucinating missing facts.

## 5. Table Handling

Tables are converted into row-level chunks with:

- table id;
- row id;
- table columns;
- source file metadata;
- stable chunk ids.

The table extractor reads common columns:

- material/alloy;
- regime/process;
- property/metric;
- value/result;
- unit;
- effect/conclusion;
- gap/data_gap.

Table-heavy documents are flagged so reviewers can inspect parser quality.

## 6. Parser Failure vs Extraction Miss

The profiler separates:

- `parser_failure`: parser crashed or file cannot be read;
- `ocr_required`: parser worked but text layer is absent/too sparse;
- `zero_facts`: parser produced chunks but deterministic extraction found no accepted facts/gaps;
- `extraction_miss`: facts may be absent because current deterministic patterns do not cover the wording;
- `retrieval_miss`: facts exist but a query does not retrieve them;
- `answer_synthesis_error`: facts are present but answer formatting/routing is wrong.

This separation is required for product readiness. A scanned PDF is not an extraction miss.

## 7. Product Metrics

Per file:

- parser backend;
- parse status;
- text density;
- page/table/image/chunk counts;
- raw and canonical facts;
- facts without evidence;
- conflicts;
- data gaps;
- warnings.

Corpus-level:

- files by extension;
- parse status counts;
- parser failures;
- OCR-required documents;
- zero-fact documents;
- dirty OCR documents;
- table-heavy documents;
- total chunks;
- total raw/canonical facts;
- facts without evidence;
- conflict groups;
- data gaps.

## 8. Resource Efficiency

The readiness profiler is small-model-first:

- no LLM calls;
- no LLM extraction;
- no embeddings;
- no Qdrant;
- no Neo4j dependency;
- deterministic parser and extraction only.

The report can run in `economy_core` and is suitable for low-resource deployment checks.

## 9. Economy Core Requirements

In `economy_core`:

- `ENABLE_LLM=false`;
- `LLM_PROVIDER=offline`;
- `ENABLE_LOCAL_EMBEDDINGS=false`;
- `RETRIEVAL_MODE=bm25`;
- facts are extracted deterministically;
- every accepted fact must have evidence;
- scanned/image-only documents must be marked as OCR-required instead of being treated as no-data evidence.

## 10. Commands

Build the product readiness report:

```powershell
python scripts/corpus_readiness_report.py --input demo_data --output artifacts/corpus_readiness_report.json --markdown artifacts/corpus_readiness_report.md
```

Run file/corpus readiness eval:

```powershell
python evaluation/eval_file_corpus_readiness.py
```

Run end-to-end dirty corpus eval:

```powershell
python evaluation/eval_dirty_demo_corpus.py
```

Interpretation:

- `PASS`: no blocking parser/extraction/provenance failures.
- `WARN`: controlled limitation, for example unsupported archive format or OCR-required scanned PDF.
- `FAIL`: parser crash, facts without evidence, raw leak, hallucinated numeric no-data answer, or economy profile violation.
