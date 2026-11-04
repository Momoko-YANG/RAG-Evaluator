# Configuration Reference

One YAML file per project, validated by `ProjectConfig` (`config/models.py`).
Every model uses `extra="forbid"`: unknown keys (including the old flat
`execution_mode` / top-level `evaluators` format) are rejected.

Start from `project-configs/templates/rag-eval/project-template.yaml`; a runnable
example is `project-configs/samples/sample.yaml`.

## Example

```yaml
project_name: my-project-e2e
dataset_path: datasets/my-project/eval.jsonl
mode: local
report_format: [json, markdown, html]
output_dir: output
metadata:
  team: my-team

local_mode:
  evaluators: [retrieval_hit_rate, groundedness, accuracy_judge, tool_selection]
  thresholds: {}
  azure:
    endpoint: https://your-resource.openai.azure.com
    deployment_name: gpt-4.1
    credential_type: key
  judge:
    model: gpt-4.1
  evaluator_options:
    retrieval_hit_rate: {k: 3}
    tool_selection: {expect_search: true}

observability:
  pricing:
    currency: USD
    source: "Azure price sheet 2026-10"
    models:
      gpt-4.1: {input_per_1m: 2.0, output_per_1m: 8.0}
    fixed_monthly:
      azure-ai-search-standard: 250.0
```

## Top-level fields

| Field | Default | Description |
|---|---|---|
| `project_name` | required | Used for the report directory and file names |
| `dataset_path` | required | JSONL dataset (see [Datasets](datasets.md)) |
| `mode` | `local` | `local` or `cloud`; the matching block is required |
| `local_mode` / `cloud_mode` | — | Mode blocks, below |
| `report_format` | `[json, markdown]` | Any of `json`, `markdown`, `html` |
| `output_dir` | `output` | Reports go to `{output_dir}/{project_name}/` |
| `metadata` | `{}` | Copied into the run metadata of every report |
| `observability.pricing` | none | Model prices for the cost summary; without it cost is "not configured", never 0 |

## `local_mode`

| Field | Default | Description |
|---|---|---|
| `evaluators` | required | Names from the local catalog (21; see [Evaluators](evaluators.md)) |
| `thresholds` | `{}` | Minimum mean score per evaluator, 0.0–1.0 |
| `azure` | none | Azure OpenAI settings for Azure SDK evaluators and judges |
| `judge` | none | Judge settings, below |
| `judge_model` | `gpt-4.1` | Judge deployment when `judge.model` is empty |
| `evaluator_options` | `{}` | Per-evaluator options, below |

## `cloud_mode`

`evaluators` (the 4 cloud evaluators), `thresholds`, `azure`, `foundry`.
See [Evaluation Modes](modes.md).

## `azure`

| Field | Default | Description |
|---|---|---|
| `endpoint` | `""` | Azure OpenAI endpoint |
| `deployment_name` | `""` | Model for Azure SDK evaluators; falls back to the judge model |
| `api_version` | `2024-12-01-preview` | |
| `credential_type` | `default` | `default` (DefaultAzureCredential), `key`, `env` |
| `api_key_env` | `AZURE_OPENAI_API_KEY` | Env var with the key for `credential_type: key` |
| `project_connection_string_env` | `AZURE_AI_PROJECT_CONNECTION_STRING` | |
| `is_reasoning_model` | `false` | Passed to the Azure SDK evaluators (`max_completion_tokens`) |

## `judge`

| Field | Default | Description |
|---|---|---|
| `model` | `""` | Judge deployment (overrides `judge_model`) |
| `temperature` | `0.0` | 0.0–2.0 |
| `max_tokens` | `1024` | |
| `azure_endpoint` | `""` | Falls back to `azure.endpoint` |
| `api_version` | `2024-12-01-preview` | |

Judge model: `judge.model` → `judge_model`. Judge endpoint: `judge.azure_endpoint`
→ `azure.endpoint`.

## `foundry` (cloud only)

| Field | Default | Description |
|---|---|---|
| `endpoint` | `""` | Foundry project endpoint (required at run time) |
| `subscription_id`, `resource_group`, `project_name` | `""` | Used only when `endpoint` is empty |
| `credential_type` | `default` | `default`, `key` (uses `azure.api_key_env`), `env` |
| `connection_string_env` | `AZURE_AI_PROJECT_CONNECTION_STRING` | For `credential_type: env` |
| `poll_interval_seconds` / `poll_timeout_seconds` | `10` / `1800` | Validated (1–300 / ≥ 30) but unused: `evaluate()` is synchronous |

## `evaluator_options`

Keyed by evaluator name. Options an evaluator does not read are ignored.

| Evaluator | Options |
|---|---|
| `retrieval_hit_rate`, `retrieval_mrr` | `k` (int, default all retrieved), `relevance_policy` (`evidence_match` default, or `qrels`) and the qrels options |
| `retrieval_recall` | `k` (int or list, default 3), qrels options; policy is always `qrels` |
| `retrieval_ndcg` | `k` (int or list, default 3), `relevance_policy` (`qrels` default, or `evidence_match`), qrels options |
| qrels options | `qrels_path`, `relevance_threshold` (1–3, default 2), `annotation_version`, `require_reviewed` (default false) |
| LLM judges | `temperature`, `max_tokens` |
| `task_success` | `use_llm_judge` (false), `judge_threshold` (0.75), `require_citation` (true), `abstain_patterns` |
| `tool_selection` | `expected_tools`, `forbidden_tools`, `allow_extra` (true), `expect_search` (false) |
| `abstention` | `patterns`, `infer_from_labels` (true) |
| `access_control` | `allowed_indexes`, `allowed_search_services`, `allowed_title_patterns`, `forbidden_title_patterns` |

Relevance policies and qrels files: [Ranking metrics](evaluators.md#ranking-metrics). E2E options:
[evaluation-layers.md](evaluation-layers.md).

## CLI

```bash
rag-eval --config <yaml> [--dataset <jsonl>] [--output <dir>] [--mode local|cloud] [--verbose]
```

`--dataset`, `--output` and `--mode` override `dataset_path`, `output_dir` and `mode`.

| Exit code | Meaning |
|---|---|
| `0` | Passed (no threshold breached) |
| `1` | Config could not be loaded or validated |
| `2` | Evaluation raised an error |
| `3` | At least one threshold breached |

## Thresholds

- **Run**: the mean of each evaluator's `SUCCESS` results (skipped and errored
  results excluded) is compared with its threshold. Any breach fails the run (exit 3).
- **Record**: a record fails when any non-skipped score is below its threshold.
- Evaluators without a threshold are reported but never fail anything.

With `thresholds: {}` a run always passes, so PASSED does not mean an accuracy
target was met.
