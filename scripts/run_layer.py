"""Run one evaluation layer: collect data, then score it with a project config.

    # Retrieval: several Top-K / KNN variants (retrieval-only, no answers)
    python scripts/run_layer.py retrieval \\
        --config my-project-retrieval.yaml \\
        --questions datasets/my-project/sets/golden.csv --extension-id <id> \\
        --variant baseline:3:10 --variant top10:10:10 --variant knn50:3:50 \\
        --output-dir output/my-project-retrieval-<date>

    # Generation: answers from the fixed top-k chunks of a saved retrieval run
    python scripts/run_layer.py generation \\
        --config my-project-generation.yaml \\
        --retrieval output/my-project-retrieval-<date>/top10/retrieval.jsonl \\
        --model gpt-4.1 --context-k 3 --output output/my-project-generation-<date>/gpt-4.1.jsonl

    # End-to-end: the real extension chat flow with traces
    python scripts/run_layer.py e2e \\
        --config my-project-e2e.yaml \\
        --questions datasets/my-project/sets/smoke.csv --extension-id <id> \\
        --output output/my-project-e2e-<date>/e2e.jsonl

The ``/api/rag-eval`` key comes from ``--rag-api-key``, env ``RAG_EVAL_API_KEY`` or
``--env-file`` (``KEY=VALUE`` lines, e.g. the app's ``.env``); it is never printed.
Generation and e2e call models (cost): start with ``--limit`` or smoke.csv.
Holdout questions should not be used while tuning.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval_framework.adapters.chat_client import OpenAIChatClient  # noqa: E402
from rag_eval_framework.adapters.rag_eval_api import RagEvalApiAdapter  # noqa: E402
from rag_eval_framework.config.loader import load_config  # noqa: E402
from rag_eval_framework.datasets.questions import read_question_bank  # noqa: E402
from rag_eval_framework.runners.e2e import E2ERunner  # noqa: E402
from rag_eval_framework.runners.generation import GenerationRunner  # noqa: E402
from rag_eval_framework.runners.retrieval import RetrievalRunner, RetrievalVariant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_layer")


def _api_key(args: argparse.Namespace) -> str:
    if args.rag_api_key:
        return str(args.rag_api_key)
    if args.env_file:
        for line in Path(args.env_file).read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key == "RAG_EVAL_API_KEY":
                return value.strip().strip('"').strip("'")
    return os.environ.get("RAG_EVAL_API_KEY", "")


def _questions(args: argparse.Namespace) -> list[dict]:
    questions = read_question_bank(Path(args.questions))
    return questions[: args.limit] if args.limit else questions


def _summary(name: str, result: object) -> None:
    scores = getattr(result, "aggregate_scores", {})
    logger.info("%s: %s", name, ", ".join(f"{k}={v}" for k, v in scores.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="layer", required=True)
    for name in ("retrieval", "generation", "e2e"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--limit", type=int, default=None)
        if name != "generation":
            p.add_argument("--questions", required=True)
            p.add_argument("--extension-id", required=True)
            p.add_argument("--rag-url", default="http://localhost:3000/api/rag-eval")
            p.add_argument("--rag-api-key", default="")
            p.add_argument("--env-file", default="")
            p.add_argument("--model", choices=["primary", "latest"], default="primary")
            p.add_argument("--request-timeout", type=int, default=180)
    sub.choices["retrieval"].add_argument(
        "--variant", action="append", required=True, help="name:topK[:knn], repeatable"
    )
    sub.choices["retrieval"].add_argument("--output-dir", required=True, type=Path)
    for name in ("generation", "e2e"):
        sub.choices[name].add_argument("--output", required=True, type=Path)
    gen = sub.choices["generation"]
    gen.add_argument("--retrieval", required=True, type=Path, help="Saved retrieval.jsonl")
    gen.add_argument("--model", required=True, help="Chat deployment, e.g. gpt-4.1")
    gen.add_argument("--context-k", type=int, default=3)
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--system-prompt-file", default="")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.layer == "retrieval":

        def make_adapter(variant: RetrievalVariant) -> RagEvalApiAdapter:
            return RagEvalApiAdapter.create(
                rag_url=args.rag_url,
                api_key=_api_key(args),
                extension_id=args.extension_id,
                model=args.model,
                top_k=variant.top_k,
                knn=variant.knn,
                retrieval_only=True,
                request_timeout=args.request_timeout,
            )

        results = RetrievalRunner(make_adapter).run(
            _questions(args),
            [RetrievalVariant.parse(v) for v in args.variant],
            args.output_dir,
            config,
            questions_path=Path(args.questions),
        )
        for name, result in results.items():
            _summary(name, result)
    elif args.layer == "generation":
        system_prompt = (
            Path(args.system_prompt_file).read_text(encoding="utf-8")
            if args.system_prompt_file
            else None
        )
        runner = GenerationRunner(
            OpenAIChatClient.from_config(config),
            model=args.model,
            context_k=args.context_k,
            system_prompt=system_prompt,
            temperature=args.temperature,
        )
        _summary("generation", runner.run(args.retrieval, args.output, config, limit=args.limit))
    else:
        adapter = RagEvalApiAdapter.create(
            rag_url=args.rag_url,
            api_key=_api_key(args),
            extension_id=args.extension_id,
            model=args.model,
            e2e=True,
            request_timeout=args.request_timeout,
        )
        _summary("e2e", E2ERunner(adapter).run(_questions(args), args.output, config))


if __name__ == "__main__":
    main()
