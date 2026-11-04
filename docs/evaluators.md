# Evaluators

An evaluator scores one record and returns an `EvaluatorResult` with a score in
0.0–1.0. The authoritative list is `LOCAL_EVALUATOR_CATALOG` /
`CLOUD_EVALUATOR_CATALOG` in `config/models.py`; the layer of each evaluator is
`EVALUATOR_LAYERS` (reports group scores by layer).

## Catalog

**Deterministic** evaluators need no Azure access. **Azure SDK** evaluators wrap
`azure-ai-evaluation` (`evaluators/common/azure_sdk.py`). **LLM judges** call Azure
OpenAI with the prompts in `evaluators/common/judge_prompts.py`. SDK and judge
scores are 1–5, normalised as `(raw − 1) / 4`.

### Retrieval layer (`evaluators/retrieval/`)

| Evaluator | Kind | Score | Needs |
|---|---|---|---|
| `retrieval_hit_rate` | Deterministic | 1 if a relevant document is in the top k (Hit@k); metadata keeps `rank` | `retrieved_documents` + labels |
| `retrieval_mrr` | Deterministic | 1 / rank of the first relevant document in the top k (MRR@k) | same |
| `retrieval_recall` | Deterministic | Recall@k per k | qrels file |
| `retrieval_ndcg` | Deterministic | nDCG@k per k | qrels file (or labels with `evidence_match`) |
| `document_retrieval` | Azure SDK, no LLM | nDCG@3 from `DocumentRetrievalEvaluator`; XDCG, fidelity, holes in metadata | labels; `.[azure]` |
| `retrieval` | Azure SDK | Usefulness of the contexts for the question (not recall) | `question`, `contexts` |
| `context_presence` | Deterministic | 1 if at least one non-empty context | — |

The first five are ranking metrics; see [Ranking metrics](#ranking-metrics).

### Generation layer (`evaluators/generation/`)

| Evaluator | Kind | Score | Needs |
|---|---|---|---|
| `answer_presence` | Deterministic | 1 if ≥ 2 words, or (text with CJK) ≥ 4 characters excluding spaces/punctuation (v2.0) | `response` |
| `exact_match_accuracy` | Deterministic | 1 if the normalised ground truth is a substring of the normalised response; 0.5 if no ground truth | `response` |
| `groundedness` | Azure SDK | Response supported by the contexts | `question`, `response`, `contexts` |
| `hallucination_judge` | LLM judge | Higher = fewer unsupported claims | `question`, `response`, `contexts` |
| `relevance` | Azure SDK | Response addresses the question | `question`, `response` |
| `accuracy_judge` | LLM judge | Factual agreement with the ground truth | `question`, `response`, `ground_truth_answer` |
| `response_completeness` | Azure SDK | Covers the ground truth's points | `question`, `response`, `ground_truth_answer` |
| `citation_judge` | LLM judge | Uses evidence from the contexts (no link verification) | `question`, `response`, `contexts` |
| `policy_compliance_judge` | LLM judge | Complies with `metadata.policy` ("(no policy provided)" if absent) | `question`, `response` |

Normalisation for `exact_match_accuracy`: lower-case, punctuation removed,
whitespace collapsed.

### E2E layer (`evaluators/e2e/`)

`task_success`, `tool_selection`, `query_construction`, `abstention`,
`access_control` — rule-based checks on the trace of an e2e run (`task_success`
can use a human label or an LLM judge). See [evaluation-layers.md](evaluation-layers.md).

### Cloud mode

Only `groundedness`, `relevance`, `retrieval` and `response_completeness`, run by
Azure AI Foundry. See [Evaluation Modes](modes.md).

---

## Ranking metrics

Hit@k, MRR@k, Recall@k and nDCG@k all get "is this retrieved chunk relevant?" from
one relevance layer (`evaluators/retrieval/relevance/`). Every metric names its
**relevance policy**, and reports record it, so two metrics agree whenever they use
the same policy. Everything runs offline from saved retrieval results; nothing calls
Azure AI Search.

### Relevance policies

| Policy | Relevant means | Relevant set | Default for |
|---|---|---|---|
| `evidence_match` | the chunk's `title` is in `ground_truth_documents` and its text contains a `ground_truth_evidence` quote (NFKC, lower-case, whitespace ignored) | **open**: any chunk containing the quote counts, e.g. an overlapping neighbour; the corpus total is unknown | `retrieval_hit_rate`, `retrieval_mrr`, `document_retrieval` |
| `qrels` | the chunk *is* a judged chunk of a qrels file with grade ≥ `relevance_threshold` (default 2) | **fixed** per question | `retrieval_recall`, `retrieval_ndcg` |

- Recall needs a fixed denominator, so `retrieval_recall` accepts only `qrels`.
- Hit and MRR accept either (`relevance_policy: qrels` plus `qrels_path`). Under
  single-gold qrels, Hit@k equals Recall@k.
- `retrieval_ndcg` accepts either; under `evidence_match` it gives exactly the SDK's
  nDCG@3.
- With `evidence_match`, either label alone is enough; a record with neither is
  skipped (use this for unanswerable questions).

Matching a retrieved chunk to a judged chunk (policy `qrels`) uses, in order:
1. `chunk_id`, the Azure AI Search key (the eval API returns `chunkId`, stored as `chunk_id`).
2. `chunk_id_sha256`, a hash of that key.
3. `content_sha256`, a hash of the normalised text, only when one side has no key
   (older runs).

The rank-based `id` (`doc-0`, …) is the prompt's citation id and is never used as an identity.

### Definitions

```text
Hit@k(q)    = 1 if a relevant chunk is in the top k, else 0
MRR@k(q)    = 1 / rank of the first relevant chunk in the top k, else 0
Recall@k(q) = |distinct relevant chunks in top k| / |relevant chunks of q|            (qrels)
nDCG@k(q)   = DCG@k / IDCG@k,  DCG = Σ gain(rank) / log2(rank + 1)
              qrels:          gain = 2^grade − 1 (unjudged 0); IDCG from the best k judged grades
              evidence_match: gain 1/0 from the top k; IDCG from those labels, or one relevant
                              unretrieved document when none is found (nDCG = 0)
Macro metric@k = mean over the questions where the value is available
```

Repeated results count once (nDCG drops them before ranking, as the SDK does).
Nothing is silently 0:

| Situation | Result |
|---|---|
| No labels / no qrels entry / wrong `annotation_version` / not `reviewed` while `require_reviewed: true` | skipped, with reason |
| No judged chunk with grade ≥ threshold | skipped |
| Fewer than k results stored and collection depth < k (e.g. `topK=3`, any `knn`) | `not_available`: "collection depth topK=3 < k=10" |
| Fewer than k results stored, but the search returned fewer than requested | computed (the list is complete) |
| Fewer than k results stored, depth unknown | `not_available` |
| Nothing retrieved | 0 |

Collection depth is `metadata.retrieval_depth` if set, otherwise
`metadata.rag_params.topK`; `knn` is not the depth. A copy cut to the top 3 must carry
`retrieval_depth: 3`.

**`document_retrieval` 2.0.** The Azure SDK computes IDCG from every labelled document
it is given. Version 1.0 labelled every stored result, so relevant chunks at ranks
4–10 entered IDCG@3 and a Top-10 file scored lower than its top-3 copy. Version 2.0
labels the top 3 only (relevant 4 / not 0); top-3 inputs give the same value in both
versions.

### Qrels files

JSONL, one line per question, kept beside the question bank (the bank is not changed).

```json
{"question_id": "q-001",
 "relevant_chunks": [
   {"chunk_id": "<index key>", "grade": 3, "source_title": "…", "evidence_quote": "…"},
   {"chunk_id_sha256": "<sha256 of key>", "content_sha256": "<sha256>", "grade": 2},
   {"chunk_id": "<index key>", "grade": 1}],
 "annotation_status": "partially_reviewed",
 "judgment_scope": "pooled",
 "annotation_version": "pooled-v1",
 "index_identity": {"service": "…", "index": "…", "indexCount": 1000, "schemaSha256": "…"}}
```

Grades: 0 irrelevant · 1 related but not enough · 2 partially supports · 3 fully
supports. A chunk that is not listed is **unjudged**, not irrelevant; each value
reports `unjudged_in_top_k`.

| `judgment_scope` + `annotation_status` | Judgment kind in reports | Describe the value as |
|---|---|---|
| `single_gold`, any status | `single_gold_proxy` | Single-gold proxy: Hit@k of that exact chunk |
| `pooled`, or `exhaustive` without `reviewed` | `pooled_partial` | Known-relevant (pooled); other relevant chunks may exist |
| `exhaustive` + `reviewed` | `human_reviewed` | Reviewed, exhaustive judgments |

`model_draft` labels are never `reviewed`; only a person who checked them sets that
status. `scripts/prepare_dataset.py` writes single-gold qrels from a bank
(`rag_eval_framework/datasets/single_gold.py`).

### Moving to multi-chunk (pooled) judgments

1. Collect deep results for golden questions only (e.g. `--top-k 50`) from the
   configurations to compare; new runs store `chunk_id`.
2. Pool the distinct chunks per question (top 10–20 of each run) plus the gold chunk.
3. A reviewer grades each pooled chunk 0–3. Model pre-grades stay `model_draft` until
   a person confirms them.
4. Write `qrels/pooled-v1.jsonl` with `judgment_scope: pooled` and
   `annotation_status: reviewed` or `partially_reviewed`.
5. Score with `annotation_version: pooled-v1`, optionally `require_reviewed: true`, and
   report the result as known-relevant (pooled) Recall / nDCG.
6. Keep holdout questions out of pooling and grading until the final check.

### Scoring a saved run

```bash
rag-eval --config my-project-retrieval.yaml \
  --dataset output/<run>/retrieval.jsonl --output output/<run>-scores
```

The report's "Recall@k", "nDCG@k", "Metric Definitions" and "Provenance" sections and
the JSON `metric_results` / `per_k_summaries` record the policy, qrels path and
SHA-256, k and version of every value (see [Reporting](reporting.md)).

---

## Results and statuses

```python
class EvaluatorResult(BaseModel):
    score: float                       # 0.0–1.0
    reason: str
    status: EvaluatorStatus            # success | skipped | error
    metadata: dict[str, Any]
    raw_output: dict[str, Any] | None
```

- **SKIPPED**: a field in `required_fields` is missing or empty (checked by
  `LocalRunner`), or a ranking evaluator has no labels for the record. Records of a
  retrieval-only collection (`metadata.rag_params.retrievalOnly: true`) have a
  placeholder `response`, so every evaluator that requires `response` is skipped for
  them; retrieval evaluators still score them. The same holds for a question the system
  did not answer (collectors store `response: "(empty answer)"`, `metadata.empty_answer`
  and a `trace.error`, e.g. the e2e tool loop ending without a final answer): the record
  stays in the dataset, answer metrics skip it, and `task_success` counts it as failed.
- **ERROR**: the Azure SDK / judge is not configured or its call failed.
- Means use `SUCCESS` results only; always read them together with the skipped /
  error counts.

Each evaluator also reports `definition()` (version, layer, relevance policy, k,
judge model and prompt hash, SDK version), stored as `metric_definitions` in every
run. Bump `version` whenever scores would change.

## Adding an evaluator

1. Create a module in `evaluators/<layer>/` and subclass one of:
   - `BaseEvaluator` (`evaluators/base.py`): implement `name` and `evaluate()`;
     override `required_fields` (default `question`, `response`), `version`,
     `definition()` and `setup(config)` as needed.
   - `AzureEvaluatorBase` (`evaluators/common/azure_sdk.py`): implement
     `_sdk_class_name`, `_build_sdk_input()` and `_normalise_sdk_output()`.
   - `BaseLLMJudge` (`evaluators/common/llm_judge.py`): implement `system_prompt`
     and `user_prompt_template` (add the prompts to `judge_prompts.py`).
   - `RetrievalEvaluatorBase` (`evaluators/retrieval/base.py`) for ranking metrics
     that need a relevance policy.
2. Register the class in `_create_default_registry()` (`core/registry.py`).
3. Add the name and description to `LOCAL_EVALUATOR_CATALOG` and its layer to
   `EVALUATOR_LAYERS` (`config/models.py`).
4. Add offline tests (mock SDK / LLM clients), and list it in this file.

```python
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult


class KeywordEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "keyword_presence"

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        found = "important keyword" in record.response.lower()
        return EvaluatorResult(score=1.0 if found else 0.0, reason="found" if found else "missing")
```
