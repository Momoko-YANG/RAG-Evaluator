# Evaluation layers: retrieval, generation, end-to-end, observability

Each layer has a runner that **collects** data and then **scores** it with a project
config (`LocalRunner` + `ReportEngine`):

```bash
python scripts/run_layer.py retrieval  --config … --questions … --extension-id … --variant name:topK[:knn] … --output-dir …
python scripts/run_layer.py generation --config … --retrieval <saved retrieval.jsonl> --model <deployment> --context-k 3 --output …
python scripts/run_layer.py e2e        --config … --questions … --extension-id … --output …
```

All three runners write a checkpointed JSONL; rerun the same command to resume.
They retry only transient errors (429, 5xx, timeouts), up to 3 attempts. They record
`metadata.eval_latency_ms` and `metadata.attempts`. The `/api/rag-eval` key comes from
`--rag-api-key`, `RAG_EVAL_API_KEY` or `--env-file` and is never printed. Generation
and e2e runs call models and so cost money: use `--limit` / `smoke.csv` first. Keep
holdout questions out of tuning.

## Retrieval (Phase 1)

`RetrievalRunner` collects retrieval-only results for each variant (e.g.
`baseline:3:10 top10:10:10 knn50:3:50`) and scores the full `retrieval.jsonl` of each.
The ranking metrics do not depend on how many results were collected; a k beyond
the depth is reported as not available. `manifest.json` lists the variants, record
counts, dataset hashes and scores. Metric definitions are in [evaluators.md](evaluators.md#ranking-metrics).

## Generation (Phase 2)

`GenerationRunner` answers each question from the **first `context_k` chunks of a
saved retrieval run**. Every model therefore sees exactly the same question and
context, so models can be compared with the same reference answers and the same judge.

- **Prompt**: the RAG prompt of `adapters/rag_eval_api/prompts.py` (replace the system
  prompt with your app's to measure it). Its `prompt_sha256`, the model and `context_k`
  are stored in `metadata.generation`.
- **Client**: `OpenAIChatClient` (Azure OpenAI from the config's `azure` block). It
  streams the answer and records TTFT, total time and token usage in `record.trace`.
- **Scoring**: `my-project-generation.yaml` uses faithfulness (groundedness,
  hallucination judge), relevance, correctness (accuracy judge), completeness,
  citation, abstention and answer presence.

## End-to-end (Phase 3)

`/api/rag-eval` with `e2e: true` (extension only) should run the same code path as the
app's chat: the same system prompt, the same tools and the same tool loop. The model
decides whether to search and writes the search text itself. Tools with side effects
(e.g. saving a memory) keep their definitions, but their effects should be recorded
instead of executed, and no history or memory should be loaded
(see [the contract](rag-eval-api.md)).

The API response adds `trace`. The collector converts it to
`observability.traces.Trace`:

```json
{"mode": "e2e",
 "tool_calls": [{"name": "search", "query": "annual leave days", "latency_ms": 120,
                 "result_count": 3, "status": "executed"},
                {"name": "remember", "status": "stubbed"}],
 "timings": {"client_ms": 2600, "total_ms": 2400, "ttft_ms": 900, "search_ms": 120},
 "usage": [{"stage": "generation", "model": "gpt-5-mini-2025-08-07",
            "input_tokens": 1000, "output_tokens": 200, "calls": 2}],
 "model_calls": 2, "citations": [{"name": "Leave-Procedures.pdf", "id": "doc-0"}], "attempts": 1}
```

`retrieved_documents` are the results of the model's **last** search (with
`chunk_id`), so the retrieval metrics score what the model actually retrieved.

### E2E evaluators

| Evaluator | Measures | How |
|---|---|---|
| `tool_selection` | the expected tools were called, no forbidden one | rule on `trace.tool_calls`; `expected_tools` per question or option; `expect_search: true` = any search call counts |
| `query_construction` | the model's search text contains the needed terms | rule; share of `expected_query_terms` (per question) found; queries kept for review |
| `abstention` | declines exactly when the question is unanswerable | rule; expected from `expected_abstain` or "no ground-truth labels"; detected by phrases ("No information found", …) |
| `access_control` | only allowed indexes / documents were searched, retrieved, cited | rule; `allowed_indexes`, `allowed_search_services`, title patterns, `metadata.forbidden_documents` |
| `task_success` | the user got what they asked | human label (`metadata.human_task_success`) > LLM judge (`use_llm_judge`) > rule (no error, correct abstention, cites a chunk matching the labels); `source` and each check recorded |

Question-bank columns for expectations (all optional): `expected_tools` and
`expected_query_terms` (`|`-separated), `expected_abstain` (true/false).
`access_control` checks the evaluation run's behaviour. It does not test the
application's per-user permission model.

## Observability

These are not evaluators. Runners record timings and usage in `record.trace` /
`record.metadata`, and every report has an **Observability** section
(`EvaluationRunResult.observability`):

- **Latency per stage** (n, mean, P50, P95 nearest-rank, max):
  - `client_ms`: collector, incl. network and retries
  - `total_ms`: server, request to final answer
  - `ttft_ms`: first answer token
  - `search_ms`: search tool calls

  The stages are nested (`search_ms` ⊂ `total_ms` ⊂ `client_ms`) and are never added together.
- **Tokens**: input / output by stage and model. Records without usage are counted, not zeroed.
- **Cost**: only from `observability.pricing` in the config (prices with their source;
  longest-prefix model match). No prices are built in.
  - Unpriced tokens and missing usage make the cost `partial` and are listed.
  - `fixed_monthly` costs (e.g. Azure AI Search) are shown but never spread over questions.
  - Not included: question embedding and Search queries (not metered per request),
    and the evaluation's own judge calls.

## Not yet covered

- Multi-turn conversations, thread history and memory injection (each question is a
  new thread).
- Chat-with-file (uploaded documents) e2e: the e2e mode covers extensions only.
- Judge-call token usage is not recorded.
- Thresholds / acceptance criteria are not set (`thresholds: {}`).
