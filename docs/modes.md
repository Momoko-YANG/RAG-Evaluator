# Evaluation Modes — Local vs. Cloud

`mode` in the project YAML picks the scorer. Each mode has its own config block
and evaluator catalog (`LOCAL_EVALUATOR_CATALOG` / `CLOUD_EVALUATOR_CATALOG` in
`config/models.py`); naming an evaluator outside the mode's catalog fails config
validation.

| | Local (`LocalRunner`) | Cloud (`FoundryRunner`) |
|---|---|---|
| Where evaluators run | In the Python process | Azure AI Foundry, via `azure.ai.evaluation.evaluate()` |
| Evaluators | All 21 (retrieval, generation, e2e) | 4: `groundedness`, `relevance`, `retrieval`, `response_completeness` |
| Azure needed | Only for Azure SDK evaluators and LLM judges | Always |
| Install | `pip install -e .` (+ `[azure]`, `[judges]`) | `pip install -e ".[cloud]"` |
| Observability, metric definitions, provenance | Yes | No |

Local mode is enough for most work. Cloud mode uploads the same kind of Azure SDK
scores to Foundry and is less tested than local mode.

Scores of the same evaluator are **not comparable across modes**: the execution
context, batching and model routing differ. Set thresholds per mode.

```bash
rag-eval --config my-project.yaml               # uses `mode` from the config
rag-eval --config my-project.yaml --mode cloud  # override
```

---

## Local mode

```yaml
project_name: my-rag
dataset_path: datasets/my-project/eval.jsonl
mode: local

local_mode:
  evaluators:
    - answer_presence        # deterministic
    - context_presence       # deterministic
    - groundedness           # Azure SDK (needs azure)
    - accuracy_judge         # LLM judge (needs azure + judge model)
  thresholds:
    answer_presence: 0.90
    groundedness: 0.70
  azure:
    endpoint: https://your-resource.openai.azure.com
    deployment_name: gpt-4.1
    credential_type: default   # or key (reads api_key_env, default AZURE_OPENAI_API_KEY)
    # is_reasoning_model: true # o-series / models that need max_completion_tokens
  judge:
    model: gpt-4.1
    temperature: 0.0
    max_tokens: 1024
```

What each evaluator family needs:

| Family | Install | Config |
|---|---|---|
| Deterministic (scaffold, ranking, e2e rules) | — | ranking: labels or a qrels file (see [Ranking metrics](evaluators.md#ranking-metrics)) |
| Azure SDK (`groundedness`, `relevance`, `retrieval`, `response_completeness`, `document_retrieval`) | `.[azure]` | `azure` block (`document_retrieval` calls no model) |
| LLM judges (`*_judge`, `task_success` with `use_llm_judge`) | `.[judges]` | `azure.endpoint` (or `judge.azure_endpoint`) and a judge model |

If the SDK / `openai` package or the configuration is missing, those evaluators
return `ERROR` results with "not configured" in the reason; the run still finishes.
LLM judges retry with `max_completion_tokens` automatically when the model rejects
`max_tokens`; Azure SDK evaluators need `is_reasoning_model: true` for that.

Authentication with `credential_type: default` uses `DefaultAzureCredential`;
`az login` is usually enough.

---

## Cloud mode

```yaml
project_name: my-rag-cloud
dataset_path: datasets/my-project/eval.jsonl
mode: cloud

cloud_mode:
  evaluators: [groundedness, relevance, retrieval, response_completeness]
  thresholds:
    groundedness: 0.80
  azure:
    endpoint: https://your-resource.openai.azure.com
    deployment_name: gpt-4.1
  foundry:
    endpoint: https://your-project.services.ai.azure.com/api/projects/your-proj
    subscription_id: "00000000-0000-0000-0000-000000000000"
    resource_group: your-resource-group
    project_name: your-foundry-project
    credential_type: default
```

How it runs:

1. Builds an `AIProjectClient` from `foundry.endpoint` (required) and the credential.
2. Calls `azure.ai.evaluation.evaluate()` with the JSONL and the selected evaluators.
   The call is synchronous: it returns when Foundry has finished. Large datasets can
   take several minutes. `poll_interval_seconds` / `poll_timeout_seconds` are
   accepted in the config but not used.
3. Normalises the returned metrics and rows (1–5 → 0–1; keys such as
   `groundedness`, `groundedness.groundedness`, `groundedness.gpt_groundedness`),
   applies thresholds and writes the same reports as local mode, with a CLOUD
   banner and `cloud_run_id` in the run metadata.

Your identity needs access to the Foundry project (e.g. Azure AI Developer) and
to the Azure OpenAI deployment.

### Troubleshooting

| Error | Fix |
|---|---|
| `Azure AI Projects SDK is not installed` | `pip install -e ".[cloud]"` |
| `Failed to initialise cloud evaluation` | Check `az account show`, `foundry.endpoint` and permissions |
| `Evaluator 'X' is not in the cloud catalog` | Use only the 4 cloud evaluators, or switch to local mode |
| `max_tokens is not supported` | `azure.is_reasoning_model: true` |

See [Configuration](configuration.md) for every field.
