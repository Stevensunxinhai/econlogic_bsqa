# EconLogic-BSQA baseline evaluation (open source)

Zero-shot **direct A–D** evaluation on cleaned CSV splits (`*_clean.csv`). This matches protocol **§20.1** in [`../datapipeline/econlogic_bsqa_transformation_protocol.md`](../datapipeline/econlogic_bsqa_transformation_protocol.md).

**Not included:** belief-driven JSON protocol (§20.3), SJA, BAC, citation metrics.

## Setup

```bash
cd opensource/evaluation
pip install -r requirements.txt
```

Copy [`../datapipeline/.env.example`](../datapipeline/.env.example) to the repository root as `.env` and set `OPENAI_API_KEY`.

## Default data

By default, `--csv-dir` points to [`../dataset`](../dataset) (`train_clean.csv`, `val_clean.csv`, `test_clean.csv`).

## Run

All splits:

```bash
python evaluate_bsqa_zero_shot.py --split all
```

Smoke (first row of val):

```bash
python evaluate_bsqa_zero_shot.py --split val --limit 1
```

## Outputs

Under `results/<sanitized_model>/` (or `--results-dir`):

- `zero_shot/<split>/<instance_id>.json` — per-example record
- `zero_shot/<split>_summary.json` — aggregate metrics for the split
- `evaluation_master_summary.json` — all splits

Metrics include exact-match accuracy, parse/API failure counts, **UCR** / **ADR** on gold-insufficient rows, and per–`condition_type` accuracy.

## License

Add a **LICENSE** at the repository root before publishing if not already present.
