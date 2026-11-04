# Architecture

## Overview

The framework evaluates a RAG system in three layers: retrieval, generation and
end-to-end (e2e). Observability (latency, tokens, cost) runs across all three.
Each module has one job:

```
Adapter ──► EvaluationRecord ──► Layer runner ──► Scorer ──► EvaluatorResult[] ──► Aggregation ──► Reporting
 (system      (core/schemas)     (collects data:   (Local /      (pure scoring)      (core/           (JSON / MD /
  specific)                       retrieval,        Foundry)                          aggregation)     HTML)
                                  generation, e2e)                         ▲
                                         │                                 │
                                         └── record.trace ── observability/ (latency, tokens, cost)
```

- **Adapters** turn one system's responses into record fields. Evaluators never know
  which system produced a record.
- **Layer runners** collect data. Each writes a resumable JSONL with retries and
  timing, then hands it to a scorer. Evaluator scoring happens in one loop only.
- **Evaluators** are pure scoring components. LLM judges call models through a client;
  generation runs use the `ChatClient` protocol (`core/protocols.py`).
- **Aggregation** summarises evaluator results and never re-derives a metric.
  **Reporting** only renders what aggregation and observability produced.

## Package layout (`src/rag_eval_framework/`)

```
core/          schemas.py (EvaluationRecord) · results.py (EvaluatorResult, EvaluationRunResult,
               MetricResult) · protocols.py (Evaluator, Adapter, ChatClient, Scorer) ·
               registry.py · aggregation.py (means, per-k summaries, metric_results) ·
               provenance.py (dataset hash, framework version, retrieval settings)
config/        ProjectConfig (+ observability.pricing), evaluator catalogs, EVALUATOR_LAYERS
datasets/      loader.py (JSONL) · validation.py · questions.py (question bank CSV) ·
               qrels.py (relevance judgments) · layout.py (dataset directory) ·
               splits.py (smoke / golden / holdout) · single_gold.py (qrels from a bank)
adapters/      base.py (RagAdapter, RagResponse) · generic.py (HTTP, NDJSON stream, agent stub) ·
               rag_eval_api/ (/api/rag-eval adapter, default RAG prompt) · chat_client.py
evaluators/
  retrieval/   hit · mrr · recall · ndcg (retrieval_ndcg, document_retrieval) ·
               context_relevance · context_presence · relevance/ (matcher, judgments)
  generation/  faithfulness · correctness · completeness · citation · relevance ·
               answer_presence · policy_compliance
  e2e/         task_success · tool_selection · query_construction · abstention · access_control
  common/      azure_sdk.py · llm_judge.py · judge_prompts.py
runners/       local.py (LocalRunner) · cloud.py + foundry/ (FoundryRunner) — scorers
               retrieval.py · generation.py · e2e.py — layer runners · collection.py (shared)
observability/ traces.py · latency.py · token_usage.py · cost.py · summary.py
reporting/     engine.py · json/markdown/html renderers · common.py
cli/           rag-eval (score a dataset with a config)
```

`scripts/run_layer.py` runs a layer runner from the command line.
`scripts/collect_rag_data.py` is the general collector, with any adapter.

## Layers

| Layer | Runner collects | Evaluators | Typical config |
|---|---|---|---|
| Retrieval | retrieval-only results per Top-K / KNN variant | Hit, MRR, Recall, nDCG (relevance policies `evidence_match` / `qrels`), context relevance / presence | `my-project-retrieval.yaml` |
| Generation | answers from **fixed** contexts of a saved retrieval run (same chunks for every model) | faithfulness, relevance, correctness, completeness, citation, abstention, answer presence | `my-project-generation.yaml` |
| E2E | the real assistant flow (model chooses tool calls and search text) with a trace | tool selection, query construction, abstention, access control, task success (+ generation and retrieval metrics on the same records) | `my-project-e2e.yaml` |

Details are in [evaluation-layers.md](evaluation-layers.md), and for the ranking
metrics in [evaluators.md](evaluators.md#ranking-metrics).

## App side

The RAG app can expose an evaluation-only endpoint, `POST /api/rag-eval`, that reuses
its own retrieval and prompt code; Python only collects, scores and reports. The
request / response contract and how to secure it are in
[The /api/rag-eval contract](rag-eval-api.md). Any other HTTP API can be collected
with the `http` adapter or a custom `RagAdapter`.

## Mode Separation

- Each mode has its own **evaluator catalog** in `config/models.py`. The local catalog
  has 21 evaluators; the cloud catalog has the 4 Azure SDK RAG evaluators.
- Cross-mode use is rejected when the config is validated. Reports carry a mode banner.

See [Evaluation Modes](modes.md).

## Evaluator Lifecycle

1. **Registry lookup**: the scorer resolves evaluator names to instances (`core/registry.py`).
2. **`setup(config)`**: options, clients, qrels.
3. **`definition()`**: version, layer and what the score depends on. Stored as
   `metric_definitions` in every run.
4. **Field check**: if a `required_fields` field is missing, the result is SKIPPED.
5. **`evaluate(record)`** returns an `EvaluatorResult` (score, status, reason, metadata).
6. **Aggregation**: means over SUCCESS results; per-k summaries; `metric_results`.
7. **Observability**: `observability_summary(records, pricing)` from traces / timings.

## Extension Points

| Extension point | Where |
|---|---|
| New metric | `evaluators/<layer>/`, register in `core/registry.py`, add to `LOCAL_EVALUATOR_CATALOG` and `EVALUATOR_LAYERS` |
| New system | an adapter in `adapters/` that fills `RagResponse` (and `trace` for e2e) |
| New model API | a `ChatClient` implementation (`core/protocols.py`) |
| Prices | `observability.pricing` in the project config |
| Report format | `reporting/` renderer registered in `ReportEngine` |

## Design Principles

1. **Config-driven**: project specifics live in YAML.
2. **Pure evaluators**: scores come from the record only. Collection, retries and
   timing belong to runners.
3. **Traceable**: every value carries its metric version, relevance policy / label
   source and dataset hash.
4. **Explicit gaps**: unavailable metrics, missing labels and unpriced cost are
   reported as such, never as 0.
5. **Testable**: all evaluators, runners and summaries are tested offline with fakes.
