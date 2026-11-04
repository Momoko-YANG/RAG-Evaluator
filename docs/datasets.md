# Datasets

## Evaluation dataset (JSONL)

`rag-eval` reads only `.jsonl` files: one `EvaluationRecord` (`core/schemas.py`) per
line. Collectors and layer runners write this format; a hand-written file works too.

```json
{
  "id": "q-001",
  "question": "How many vacation days do employees get per year?",
  "response": "Full-time employees receive up to 20 vacation days per year.",
  "contexts": ["…"],
  "ground_truth_answer": "20 vacation days",
  "retrieved_documents": [
    {"id": "doc-0", "chunk_id": "<index key>", "title": "Leave-Procedures.pdf", "content": "…", "score": 0.83}
  ],
  "ground_truth_documents": ["Leave-Procedures.pdf"],
  "ground_truth_evidence": ["up to 20 vacation days per year"],
  "metadata": {"set": "golden", "rag_params": {"topK": 3, "knn": 10}}
}
```

| Field | Default | Used by |
|---|---|---|
| `id` | required | — |
| `question` | required, non-empty | all |
| `response` | required, non-empty | generation evaluators (retrieval-only runs store a placeholder; evaluators that need `response` are skipped for them) |
| `contexts` | `[]` | `groundedness`, `retrieval`, `hallucination_judge`, `citation_judge`, `context_presence` |
| `ground_truth_answer` | `""` | `exact_match_accuracy`, `response_completeness`, `accuracy_judge` |
| `retrieved_documents` | `[]` | ranking evaluators, in rank order; `title` = file name, `chunk_id` = Search key |
| `ground_truth_documents` | `[]` | `evidence_match` policy: file names of the answering documents |
| `ground_truth_evidence` | `[]` | `evidence_match` policy: passages the relevant chunk contains |
| `trace` | none | e2e evaluators and observability (tool calls, timings, token usage) |
| `expected_tools`, `expected_query_terms`, `expected_abstain` | empty | `tool_selection`, `query_construction`, `abstention` |
| `metadata` | `{}` | free-form; `rag_params` (actual settings), `retrieval_depth`, `policy`, `human_task_success`, `forbidden_documents`, `eval_latency_ms` |

Loading is strict in the CLI: an invalid line or record stops the run.
`load_dataset(path, strict=False)` skips invalid records with a warning instead.
A record with an empty field required by an evaluator is scored SKIPPED for that
evaluator only.

## Question bank (CSV)

Collectors read a CSV bank and write the JSONL above.

| Column | Meaning |
|---|---|
| `question` | required |
| `id` | optional (generated when missing) |
| `ground_truth_answer` | reference answer |
| `ground_truth_documents`, `ground_truth_evidence` | `|`-separated retrieval labels; leave empty for unanswerable questions |
| `expected_tools`, `expected_query_terms` | `|`-separated e2e expectations |
| `expected_abstain` | `true` / `false` |
| `metadata_*` | copied to `metadata.*` (e.g. `metadata_set`, `metadata_position`) |

Qrels files for Recall@k / nDCG@k sit beside the bank; see [Ranking metrics](evaluators.md#qrels-files).

## Collecting

```bash
python scripts/collect_rag_data.py --adapter <name> --questions <bank.csv> --output <eval.jsonl> [options]
```

| Adapter | System |
|---|---|
| `rag_eval_api` | an app's `POST /api/rag-eval` ([contract](rag-eval-api.md); `--chat-thread-id` or `--extension-id`) |
| `http` | Generic REST: POSTs `{"question": …}`, reads `--response-field` / `--contexts-field` |
| `azure_chat_stream` | NDJSON streaming chat endpoint |

Main options: `--rag-url`, `--rag-api-key` (`rag_eval_api` also reads `RAG_EVAL_API_KEY`), `--retrieval-only`,
`--top-k`, `--knn`, `--model primary|latest`, `--system-prompt-file`,
`--request-timeout`, `--delay`. See `--help`.

Add an adapter by subclassing `RagAdapter` (`adapters/base.py`) and registering it in
`ADAPTER_REGISTRY` (`adapters/__init__.py`).

For layered runs (retrieval variants, generation from fixed contexts, e2e with a
trace) use `scripts/run_layer.py`; see [evaluation-layers.md](evaluation-layers.md).

Helper scripts:

| Script | Purpose |
|---|---|
| `sample_index_chunks.py` | Random chunks of an extension index (via `/api/rag-eval/sample`) to write questions from |
| `prepare_dataset.py` | Rebuild every derived file of a dataset from its bank (below): sets (`datasets/splits.py`), single-gold qrels (`datasets/single_gold.py`), manifest; `--check` only verifies |
| `build_review_sheet.py` | CSV for reviewing a sample by hand (one question per source document; source chunk, other retrieved chunks of a run, review columns). Write it under `output/` |
| `xlsx_to_markdown.py` | Excel workbooks to Markdown for chat upload |

## Dataset directory

Every collected dataset (`datasets/<name>/`, e.g. `datasets/sample/`) uses one layout,
defined in `rag_eval_framework/datasets/layout.py`, that keeps hand-edited and derived files apart:

```
datasets/<name>/
├── README.md
├── manifest.json              provenance, file SHA-256, counts, validation
├── bank/                      edited by hand — the only files to edit
│   ├── questions.csv
│   └── excluded.csv           removed questions with exclusion_reason
├── sets/                      generated: smoke.csv, golden.csv (smoke + golden), holdout.csv
├── qrels/                     relevance judgments: <single-gold version>.jsonl (generated), pooled / reviewed (by hand)
├── chunks/                    index datasets: sampled_chunks.jsonl (index sample the questions came from)
└── source/, upload/           chat-upload datasets: original documents / files to upload
```

| Path | Written by |
|---|---|
| `bank/*` | hand |
| `chunks/sampled_chunks.jsonl` | `sample_index_chunks.py` (snapshot of the index; never edit, keep it) |
| `sets/*` | `prepare_dataset.py` |
| `qrels/<single-gold version>.jsonl` | `prepare_dataset.py`, while every entry is `single_gold` + `model_draft` |
| other `qrels/*.jsonl` (pooled / reviewed) | hand; never regenerated |
| `manifest.json`: `files`, counts, `validation`, `singleGoldQrels` | `prepare_dataset.py` (created if missing) |
| `manifest.json`: other fields (`revisions`, index identity, …) | hand |

Run results are not kept here: they go to `output/<run name>/` with their config.

```bash
python scripts/prepare_dataset.py --dataset datasets/my-project --qrels-version single-gold-v2  # once
python scripts/prepare_dataset.py --dataset datasets/my-project          # after editing the bank
python scripts/prepare_dataset.py --dataset datasets/my-project --check  # verify, write nothing
```

Validation covers structure and consistency (ids, duplicate questions, evidence / title /
content hash against the sampled chunk, golden–holdout source overlap, excluded
boilerplate), not label quality. If it fails, nothing is written.

## Sample

`datasets/sample/` is a small fictional dataset (10 questions on made-up HR documents) in the
same layout and record format as the experiment data: `bank/`, generated `sets/` and
`qrels/single-gold-v1.jsonl`, `chunks/`, `manifest.json`. One question is not answered by
the documents. As an exception to "no runs in a dataset", it bundles a collected run of the
golden side, `example-run/golden.jsonl` (answers, top-3 chunks with `chunk_id`, `rag_params`),
so that `project-configs/samples/sample.yaml` can be scored offline.

Every dataset except `datasets/sample/` is git-ignored, so private documents and
questions stay local.
