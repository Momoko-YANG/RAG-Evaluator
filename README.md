# RAG Evaluation Framework

A **reusable, config-driven evaluation framework** for Retrieval-Augmented Generation (RAG) systems, with first-class support for **Azure AI Foundry** and **Azure OpenAI**. It ships adapters for plain HTTP APIs and for an [evaluation endpoint](docs/rag-eval-api.md) you can add to your app (see [Evaluating Your RAG App](#evaluating-your-rag-app)).

Evaluate any RAG application by adding a YAML config and a JSONL dataset — no core code changes required.

---

## Key Features

- **21 pluggable evaluators** in three layers (retrieval, generation, e2e) — deterministic checks, ranking metrics (Hit / MRR / Recall / nDCG), Azure AI Evaluation SDK, LLM-as-a-judge
- **Local and cloud execution** — run evaluators in-process or submit batch jobs to Azure AI Foundry
- **Config-driven** — one YAML file per project; no code changes to onboard new apps
- **Strict evaluator catalogs** — mode-aware validation prevents misconfiguration
- **JSON, Markdown & HTML reports** with per-record and aggregate pass/fail
- **Threshold enforcement** — structured breach detection with CLI exit codes for CI/CD
- **Extensible** — add custom evaluators by subclassing `BaseEvaluator`

## Supported Evaluators

| Evaluator | Tier | Local | Cloud | Description |
|---|---|:---:|:---:|---|
| `answer_presence` | Scaffold | ✅ | — | Response contains a substantive answer |
| `context_presence` | Scaffold | ✅ | — | At least one non-empty context retrieved |
| `exact_match_accuracy` | Scaffold | ✅ | — | Ground truth appears in the response |
| `retrieval_hit_rate` | Retrieval ranking | ✅ | — | A labelled relevant document is in the top k (Hit@k) |
| `retrieval_mrr` | Retrieval ranking | ✅ | — | Reciprocal rank of the first relevant document (MRR@k) |
| `retrieval_ndcg` | Retrieval ranking | ✅ | — | nDCG@k from qrels grades (or top-k evidence labels), pure Python |
| `retrieval_recall` | Retrieval ranking | ✅ | — | Share of judged relevant chunks (qrels) in the top k (Recall@k); see [Ranking metrics](docs/evaluators.md#ranking-metrics) |
| `groundedness` | Azure SDK | ✅ | ✅ | Response grounded in provided context |
| `relevance` | Azure SDK | ✅ | ✅ | Response relevant to the question |
| `retrieval` | Azure SDK | ✅ | ✅ | Quality of retrieved context passages |
| `response_completeness` | Azure SDK | ✅ | ✅ | Response completeness vs. ground truth |
| `document_retrieval` | Azure SDK | ✅ | — | nDCG@3 and other ranking metrics from labelled chunks (no LLM) |
| `accuracy_judge` | LLM Judge | ✅ | — | Factual accuracy (1–5 → 0–1) |
| `hallucination_judge` | LLM Judge | ✅ | — | Claims not supported by context |
| `citation_judge` | LLM Judge | ✅ | — | Evidence usage from context |
| `policy_compliance_judge` | LLM Judge | ✅ | — | Compliance with a policy document |
| `task_success` | E2E | ✅ | — | Human label, else LLM judge, else rule (correct abstention + cited relevant chunk) |
| `tool_selection` | E2E | ✅ | — | Expected tools called, no forbidden one (from the trace) |
| `query_construction` | E2E | ✅ | — | Expected terms in the model's search text |
| `abstention` | E2E | ✅ | — | Declines exactly when the question is unanswerable |
| `access_control` | E2E | ✅ | — | Only allowed indexes / documents searched, retrieved, cited |

Retrieval, generation and e2e runs, traces and the Observability section (latency per
stage, tokens, cost): see [docs/evaluation-layers.md](docs/evaluation-layers.md).

## Architecture

Technical documentation is in [docs/](docs/) (see [Documentation](#documentation)).

```
Mode: local                           Mode: cloud
┌────────────────────────┐            ┌──────────────────────────┐
│  YAML Config           │            │  YAML Config             │
│  ↓                     │            │  ↓                       │
│  LocalRunner           │            │  FoundryRunner           │
│  ├─ Scaffold evals     │            │  ├─ FoundryAdapter       │
│  ├─ Azure SDK evals    │            │  │  └─ azure.ai.projects │
│  └─ LLM Judge evals    │            │  └─ Result normaliser    │
│  ↓                     │            │  ↓                       │
│  Reports (JSON/MD/HTML)│            │  Reports (JSON/MD/HTML)  │
└────────────────────────┘            └──────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone <repository-url>
cd rag-eval

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -e ".[dev,collect]"
```

**Optional extras:**

```bash
pip install -e ".[collect]"  # requests — needed by scripts/collect_rag_data.py
pip install -e ".[azure]"    # Azure SDK evaluators (local mode)
pip install -e ".[judges]"   # LLM-as-a-judge evaluators
pip install -e ".[cloud]"    # Cloud mode (Azure AI Foundry)
pip install -e ".[all]"      # Everything
```

### 2. Run the Sample Evaluation

The bundled sample (`datasets/sample/`, fictional HR documents in the layout and record format of a
collected dataset) is scored offline with deterministic and ranking evaluators — no Azure credentials needed:

```bash
rag-eval --config project-configs/samples/sample.yaml
```

### 3. View Reports

Reports are written to `output/<project-name>/`:

```bash
cat output/sample/eval-report-sample.md
```

### 4. Run Tests

```bash
pytest
```

---

## Evaluating Your RAG App

There are two ways to collect answers and retrieved chunks from an app:

- **Any HTTP API** — `--adapter http` (configurable request / response fields) or
  `azure_chat_stream` (NDJSON streaming), or your own `RagAdapter` subclass.
- **An evaluation endpoint in the app** — add `POST /api/rag-eval`, which reuses the app's
  own retrieval and prompt and returns the answer, the retrieved chunks with their index keys,
  the settings used and (optionally) a trace of tool calls. It enables retrieval-only runs,
  Top-K / KNN / prompt tuning and the end-to-end layer. The request / response contract
  and how to secure the endpoint are in [The /api/rag-eval contract](docs/rag-eval-api.md).

### Workflow

Keep private documents and questions in the git-ignored `datasets/<project>/`, in the
layout of [Datasets](docs/datasets.md#dataset-directory) (`datasets/sample/` is a complete example).

1. **Questions** — the bank is `datasets/<project>/bank/questions.csv` with the columns
   `id, question, ground_truth_answer, ground_truth_documents, ground_truth_evidence, metadata_*`.
   `ground_truth_documents` (file names) and `ground_truth_evidence` (a short passage the right chunk
   contains) are `|`-separated retrieval labels; leave them empty for questions the documents do not answer.
   For an index, write questions from **random chunks** (`scripts/sample_index_chunks.py`, via
   `POST /api/rag-eval/sample`) so they are not biased towards chunks the retriever already finds; add your
   documents' repeated headers / footers with `--boilerplate-pattern` so they are not sampled.
2. **Question sets** — run `python scripts/prepare_dataset.py --dataset datasets/<project>`. It gives each new
   question a set (`metadata_set`; existing questions keep theirs, questions from one chunk share a set),
   validates the bank, updates the single-gold qrels and the manifest, and writes the files to collect with:

   | File | Questions | Run |
   |---|---|---|
   | `sets/smoke.csv` | 30, part of golden | after every code or prompt change; catches serious breakage in minutes |
   | `sets/golden.csv` | the rest except holdout (aim for 200–500) | for every settings decision |
   | `sets/holdout.csv` | 20% of new questions (aim for 50–100) | only now and then, never for tuning, so the settings are not fitted to golden |

3. **Collect**:
   ```bash
   export RAG_EVAL_API_KEY=<endpoint key>   # PowerShell: $env:RAG_EVAL_API_KEY="..."
   python scripts/collect_rag_data.py --adapter rag_eval_api \
     --rag-url http://localhost:3000/api/rag-eval \
     --extension-id <id> \
     --questions datasets/<project>/sets/golden.csv \
     --output output/<run name>/golden.jsonl
   ```
   Use `--chat-thread-id <id>` instead to search the documents uploaded to a chat thread. Each record gets
   `retrieved_documents` (file name, chunk, index key, search score) and `metadata.rag_params`.
4. **Retrieval (fast, no LLM)** — collect with `--retrieval-only --top-k 10` and score with a config of
   ranking evaluators (Hit@3, MRR@3, nDCG@3, Recall@k from a qrels file — see
   [Ranking metrics](docs/evaluators.md#ranking-metrics) and `project-configs/samples/sample.yaml`).
5. **Layered (retrieval / generation / end-to-end)** — `scripts/run_layer.py` collects and scores each layer
   ([Evaluation layers](docs/evaluation-layers.md)); then
   `python scripts/summarize_layers.py --report <report.json> [--report ...] --dataset <eval.jsonl> --output layers.md`
   shows the scores per layer and which layer failed for each question.

### Tuning RAG with local runs

Local mode is enough for tuning; cloud mode only uploads the same results to Azure AI Foundry.

1. Keep the questions and the documents fixed, and run once with the app's defaults as the baseline.
2. Change one thing per run with the collector flags `--top-k`, `--knn`, `--system-prompt-file` or `--model`.
   The settings used are stored in each record's `metadata.rag_params`.
3. Write each run to its own directory (report file names are fixed, so runs would otherwise overwrite each other):
   ```bash
   python scripts/collect_rag_data.py --adapter rag_eval_api ... --top-k 5 --output output/topk5/golden.jsonl
   rag-eval --config my-project-e2e.yaml --dataset output/topk5/golden.jsonl --output output/topk5-golden
   ```
4. Compare `retrieval_hit_rate` / `retrieval_mrr` for retrieval changes and the answer-quality evaluators
   (`accuracy_judge`, `groundedness`, ...) for prompt / model changes; `scripts/compare_runs.py` lists
   the questions that moved between two runs.

Use enough questions (50–100+) that one question does not swing the score, confirm the final settings on the
holdout set, and keep the judge deployment fixed.

### Documents uploaded to a chat

For apps that index files uploaded to a chat thread, upload the evaluation documents to one dedicated thread
and collect with `--chat-thread-id`. Keep the original files in `datasets/<project>/source/` and what was
uploaded in `upload/`. Upload pipelines often skip `.xlsx`: convert workbooks with
`python scripts/xlsx_to_markdown.py datasets/<project>/source/*.xlsx --output-dir datasets/<project>/upload`.
Check the labels once per upload by collecting with `--retrieval-only --top-k 50`: a question without a hit
even in the top 50 usually has an evidence passage that the extracted (OCR) text does not contain.
Chunk size is applied at upload, so changing it needs the documents uploaded again to a new thread.

---

## Onboarding a New Project

1. Copy the template:
   ```bash
   cp project-configs/templates/rag-eval/project-template.yaml my-project.yaml
   ```
2. Prepare a JSONL dataset at `datasets/my-project/eval.jsonl` (see [Datasets](docs/datasets.md)),
   or collect one with `scripts/collect_rag_data.py` (adapters: `http`, `azure_chat_stream`, `rag_eval_api`)
3. Edit your config — choose evaluators, set thresholds, configure Azure credentials if needed
4. Run:
   ```bash
   rag-eval --config my-project.yaml
   ```

See [Configuration](docs/configuration.md) and [Evaluators](docs/evaluators.md) for the details.

---

## Configuration Overview

Each project is driven by a YAML config with a `mode` field and a corresponding mode block:

```yaml
# Local mode — all 21 evaluators available
project_name: my-rag
dataset_path: datasets/my-project/eval.jsonl
mode: local

local_mode:
  evaluators:
    - answer_presence
    - groundedness
    - accuracy_judge
  thresholds:
    answer_presence: 0.90
    groundedness: 0.70
  azure:
    endpoint: https://your-resource.openai.azure.com
    deployment_name: gpt-4
    credential_type: default
  judge:
    model: gpt-4.1
```

```yaml
# Cloud mode — Azure SDK evaluators via Azure AI Foundry
mode: cloud

cloud_mode:
  evaluators:
    - groundedness
    - relevance
  thresholds:
    groundedness: 0.80
  azure:
    endpoint: https://your-resource.openai.azure.com
  foundry:
    endpoint: https://your-project.services.ai.azure.com/api/projects/your-proj
    subscription_id: "00000000-0000-0000-0000-000000000000"
    resource_group: your-resource-group
    project_name: your-foundry-project
```

See [Configuration Reference](docs/configuration.md) for all fields.

---

## Repository Structure

```
rag-eval/
├── pyproject.toml                     # Python project metadata and dependencies
├── README.md                          # This file
├── .env.example                       # Environment variable template
├── src/rag_eval_framework/
│   ├── config/                        # YAML config loading, Pydantic models, evaluator catalog + layers
│   ├── core/                          # Schemas, results, protocols, registry, aggregation, provenance
│   ├── datasets/                      # JSONL loading/validation, question banks, qrels, layout, splits
│   ├── adapters/                      # RAG system adapters (rag_eval_api/, generic), chat client
│   ├── evaluators/
│   │   ├── retrieval/                 # hit, mrr, recall, ndcg, context relevance / presence, relevance/
│   │   ├── generation/                # faithfulness, correctness, completeness, citation, relevance, ...
│   │   ├── e2e/                       # task_success, tool_selection, query_construction, abstention, access_control
│   │   └── common/                    # Azure AI Evaluation SDK wrapper, LLM judge base + prompts
│   ├── runners/
│   │   ├── local.py / cloud.py        # Scorers: in-process / Azure AI Foundry (+ foundry/)
│   │   └── retrieval.py / generation.py / e2e.py   # Layer runners: collect, then score
│   ├── observability/                 # Traces, latency per stage, token usage, cost
│   ├── reporting/                     # JSON, Markdown, HTML report generators
│   ├── cli/                           # CLI entrypoint (click-based)
│   └── utils/                         # Normalisation helpers
├── project-configs/
│   ├── templates/rag-eval/project-template.yaml  # Starter template
│   └── samples/sample.yaml                       # Sample config (installation check, offline)
├── datasets/
│   ├── sample/                        # Fictional sample in the dataset layout, with an example run
│   └── <project>/                     # Your datasets (git-ignored)
├── scripts/
│   ├── run_layer.py                   # Run a layer (retrieval / generation / e2e): collect + score
│   ├── collect_rag_data.py            # Data collection CLI (adapters in rag_eval_framework.adapters)
│   ├── prepare_dataset.py             # Rebuild / check a dataset's splits, qrels, manifest, validation
│   ├── build_review_sheet.py          # Sample sheet for reviewing questions / labels by hand
│   ├── sample_index_chunks.py         # Random chunks of a search index, for writing questions
│   ├── summarize_layers.py            # Scores per layer and the failing layer per question
│   ├── compare_runs.py                # Two runs question by question (hit in both / only A / only B)
│   └── xlsx_to_markdown.py            # Excel workbooks to Markdown for chat upload
├── docs/                              # Technical documentation
└── tests/                             # Unit tests
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Data flow, package layout, layers, extension points |
| [The /api/rag-eval contract](docs/rag-eval-api.md) | The evaluation endpoint an app can expose: request, response, trace, security |
| [Evaluation layers](docs/evaluation-layers.md) | Retrieval / generation / e2e runners, e2e evaluators, observability |
| [Evaluators](docs/evaluators.md) | All 21 evaluators, ranking metrics (relevance policies, qrels, Recall@k, nDCG@k), adding one |
| [Configuration](docs/configuration.md) | YAML fields, evaluator options, CLI, thresholds |
| [Evaluation modes](docs/modes.md) | Local vs. cloud (Azure AI Foundry) |
| [Datasets](docs/datasets.md) | JSONL record, question bank CSV, collectors and helper scripts |
| [Reporting](docs/reporting.md) | Report sections and JSON structure |

---

## CLI Exit Codes

`rag-eval` exits with a code that CI jobs and scripts can check:

| Exit Code | Meaning |
|---|---|
| `0` | All thresholds passed |
| `1` | Config load error |
| `2` | Evaluation runtime error |
| `3` | Threshold breaches detected |

---

## Development

### Prerequisites

- Python ≥ 3.10
- (Optional) Azure CLI for `az login` when using Azure SDK or LLM judge evaluators

### Run Tests

```bash
pytest                          # all tests
pytest tests/test_config.py     # specific module
pytest -x --tb=short            # fail fast with short tracebacks
```

### Lint

```bash
ruff check src/ tests/ scripts/
mypy src/
```

To add an evaluator, see [Evaluators — Adding an evaluator](docs/evaluators.md#adding-an-evaluator).
