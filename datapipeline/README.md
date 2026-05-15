# EconLogic-BSQA data pipeline (open source)

This folder contains scripts to **reproduce the EconLogic → EconLogic-BSQA transformation** described in [`econlogic_bsqa_transformation_protocol.md`](econlogic_bsqa_transformation_protocol.md).

## What it does

1. **`transform_econlogic_bsqa.py`** — LLM-assisted pipeline (prompts 1–9): reads EconLogicQA CSV splits, writes JSONL (`train.jsonl`, `val.jsonl`, `test.jsonl`), plus `rejected.jsonl` and `run_summary.json`.
2. **`verify_bsqa_output.py`** — Structural checks on a JSONL file (JSON validity, duplicate `instance_id`, required keys).
3. **`clean_bsqa_to_csv.py`** — Dedupes, applies the documented exclusion of one real-entity leakage row, optionally drops failed validation rows, exports wide `*_clean.csv` and `cleaning_report.json`.

Released benchmark CSVs (if you do not regenerate) live under [`../dataset`](../dataset).

## Setup

```bash
cd opensource/datapipeline
pip install -r requirements.txt
```

Copy `.env.example` to the **repository root** `.env` (two levels above this folder) and set `OPENAI_API_KEY`. The scripts load `<repo>/.env` by default.

## Run

Smoke (one train row):

```bash
python transform_econlogic_bsqa.py --split train --limit 1
```

Full transform (all splits; adjust workers as needed):

```bash
python transform_econlogic_bsqa.py --split all --workers 20 --timeout 120 --retries 2
```

Custom EconLogicQA CSV directory:

```bash
python transform_econlogic_bsqa.py --input-dir /path/to/econ_logic_qa --output-dir /path/to/out
```

Verify JSONL:

```bash
python verify_bsqa_output.py /path/to/train.jsonl
```

Export cleaned CSVs:

```bash
python clean_bsqa_to_csv.py --input-dir /path/to/jsonl_dir --output-dir /path/to/clean_csv
```

## Outputs

- **JSONL kept rows:** `quality_score >= 4`, `final_keep`, passing local schema (see protocol).
- **Rejected:** reasons include filter-out, validation failure, pipeline errors.

## Licensing / data

- Redistribution of raw **EconLogicQA** CSVs depends on that dataset’s license; this pipeline accepts any directory with `train.csv`, `val.csv`, `test.csv` in the expected column format.
- Add a **LICENSE** at the repository root before publishing if not already present.
