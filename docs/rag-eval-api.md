# The `/api/rag-eval` contract

The `rag_eval_api` adapter (`--adapter rag_eval_api`), `scripts/run_layer.py` and
`scripts/sample_index_chunks.py` talk to an **evaluation-only endpoint that you add to
your RAG app**. The endpoint reuses the app's own retrieval and prompt code and returns
plain JSON, so the framework measures what users actually get, while Python only
collects, scores and reports.

The app owns the implementation; this page is the contract the framework expects.
Apps without such an endpoint can use the `http` adapter instead (see
[Datasets — Collectors](datasets.md)).

```
question bank CSV ─► collect_rag_data.py / run_layer.py ─► POST /api/rag-eval ─► app's search (+ answer model)
                                                                     │
                     rag-eval (LocalRunner + ReportEngine) ◄── JSONL ┘
```

## Security

The endpoint bypasses the app's normal user session, so guard it:

- Require a long random key in the `x-rag-eval-key` header (the collector reads it from
  `--rag-api-key` or `RAG_EVAL_API_KEY`). Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- Allow only listed chat threads / extensions (for example `RAG_EVAL_ALLOWED_THREAD_IDS`,
  `RAG_EVAL_ALLOWED_EXTENSION_IDS`), so the key cannot read arbitrary documents.
- Enable it only where it is needed (for example only in development builds, behind
  `RAG_EVAL_ENABLED=true`), and return 404 otherwise.
- Do not write evaluation questions to chat history, memory or citations storage.

## `POST /api/rag-eval`

Request (JSON). Only `question` is required; the overrides are for tuning experiments
and default to the app's own settings.

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The user question |
| `chatThreadId` | string | Search the documents uploaded to this chat thread |
| `extensionId` | string | Search this extension's search index instead (not with `chatThreadId`) |
| `selectedModel` | string | Model key of the app (collector `--model`, default `primary`) |
| `topK` | int | Chunks to retrieve (`--top-k`) |
| `knn` | int | Vector candidates in a hybrid search (`--knn`) |
| `systemPrompt` | string | Replaces the RAG system prompt (`--system-prompt-file`) |
| `retrievalOnly` | bool | Skip answer generation (`--retrieval-only`) |
| `e2e` | bool | Run the real assistant flow (the model chooses tool calls and search text) and return a `trace` (`--e2e`, extensions only) |

Response (JSON):

```json
{
  "response": "answer (citation markup removed)",
  "rawResponse": "answer with {% citation items=[{name:\"file.pdf\",id:\"doc-0\"}] /%}",
  "contexts": ["retrieved chunk", "..."],
  "documents": [
    {"id": "doc-0", "chunkId": "<index key>", "metadata": "file.pdf",
     "pageContent": "retrieved chunk", "score": 0.83}
  ],
  "deploymentName": "gpt-4.1",
  "params": {"topK": 3, "knn": 10, "systemPrompt": "default", "source": "extension",
             "searchName": "my-search-service", "indexName": "index-my-project",
             "retrievalOnly": false},
  "trace": {}
}
```

| Field | Used for |
|---|---|
| `documents[].id` | the citation id in the prompt (the rank) |
| `documents[].chunkId` | the search index key; stored as `chunk_id` for Recall@k / nDCG@k with qrels (optional) |
| `documents[].metadata` | the file name; matched against `ground_truth_documents` |
| `params` | stored in `metadata.rag_params` and in the report's provenance; `searchName` / `indexName` are checked by `access_control` |
| `rawResponse` | citations for the e2e trace and `citation_judge` |

An error is a 4xx / 5xx status with `{"error": "message"}`; 429 and 5xx are retried.

### `trace` (with `e2e: true`)

```json
{
  "toolCalls": [
    {"name": "search", "arguments": {}, "query": "annual leave days", "latencyMs": 120,
     "documentCount": 3, "executed": "search"},
    {"name": "remember", "executed": "stubbed"}
  ],
  "totalMs": 2400, "ttftMs": 900, "model": "gpt-5-mini", "modelCalls": 2,
  "usage": {"promptTokens": 1000, "completionTokens": 200}
}
```

`executed` is `search` (or `executed`) for a tool that ran and `stubbed` for one whose
side effects were only recorded. `documents` are the results of the model's **last**
search. The collector converts the trace to `observability.traces.Trace`
(see [Evaluation layers](evaluation-layers.md#end-to-end-phase-3)).

## `POST /api/rag-eval/sample`

Lists the chunks of an allowed extension index page by page, for
`scripts/sample_index_chunks.py` (questions written from random chunks are not biased
towards chunks the retriever already finds).

Request: `{"extensionId": "...", "skip": 0, "top": 500}`.
Response: `{"total": 1234, "chunks": [{"position": 0, "metadata": "file.pdf", "pageContent": "..."}]}`
(no vectors).
