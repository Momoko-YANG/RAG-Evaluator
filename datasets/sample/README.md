# Sample dataset (fictional)

Ten questions on made-up HR documents, in the layout and record format of the experiment
datasets (`docs/datasets.md`, "Dataset directory").

```
datasets/sample/
├── manifest.json                 index identity, split settings, file SHA-256, validation
├── bank/questions.csv            the bank (edit here, then run prepare_dataset.py)
├── sets/                         smoke 3 / golden 7 (incl. smoke) / holdout 3, generated
├── qrels/single-gold-v1.jsonl    one source chunk per answerable golden-side question, generated
├── chunks/sampled_chunks.jsonl   the chunk each answerable question was written from
└── example-run/golden.jsonl      collected run of sets/golden.csv (sample-only exception)
```

- Questions from one document share a set (the HR handbook's three questions are holdout).
- `smp-010` is not answered by the documents: no evidence, no chunk, skipped by label-based metrics.
- In `example-run/`, `smp-004` misses its source chunk (and answers wrongly); `smp-003` and
  `smp-009` find it at rank 2.

```bash
python scripts/prepare_dataset.py --dataset datasets/sample --check
rag-eval --config project-configs/samples/sample.yaml
```
