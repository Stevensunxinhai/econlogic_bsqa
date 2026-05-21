# EconLogic-BSQA

**EconLogic-BSQA** is a **closed-world** benchmark for evaluating whether language models can make **business decisions only when belief is sufficient**, using explicit local rules and observed events. It stresses **uncertainty-aware behavior** (when to request information or defer), **abstention from over-commitment**, and **resistance to global-knowledge leakage** when the scenario is meant to be solved from the text alone.

The benchmark is built by transforming **EconLogicQA** event-ordering items into multi-step decision instances with gold **sufficiency** labels and A–D answer choices (commit vs. defer / request information). This repository ships **released CSV splits**, a **regeneration datapipeline** (LLM-assisted, optional), and a **zero-shot baseline evaluator** aligned with protocol §20.1. The project name indicates dataset lineage and task transformation; it does not imply affiliation with, sponsorship by, or endorsement from the original EconLogicQA authors.

---

## What’s in this repository

| Path | Description |
|------|-------------|
| [`dataset/`](dataset/) | Released cleaned splits: `train_clean.csv`, `val_clean.csv`, `test_clean.csv`. |
| [`datapipeline/`](datapipeline/) | EconLogicQA → EconLogic-BSQA transformation (prompts 1–9), JSONL verification, CSV export. Full specification: [`datapipeline/econlogic_bsqa_transformation_protocol.md`](datapipeline/econlogic_bsqa_transformation_protocol.md). |
| [`evaluation/`](evaluation/) | **Baseline only:** zero-shot direct A–D evaluation on `*_clean.csv` (no belief-driven JSON / SJA / BAC in this script). |

Additional documentation lives in each subdirectory’s `README.md`.

---

## Dataset (released CSV)

The cleaned splits are **wide** CSV exports: each row includes `qa_text` (full scenario), `gold_answer` (A–D), `gold_sufficiency`, `gold_decision`, `condition_type`, provenance fields, and more. Splits are sized for reproducibility papers (order of magnitude: **hundreds** of examples per split after QC; see protocol and cleaning logic in `datapipeline/`).

- **Do not commit API keys.** Use `.env` locally; see `datapipeline/.env.example` and `evaluation/.env.example`.
- **Raw EconLogicQA redistribution** is subject to the original dataset’s license. The datapipeline accepts any directory with `train.csv`, `val.csv`, `test.csv` in the expected column format. Users should obtain source EconLogicQA data from the original release and follow its licensing terms.

---

## Quick start

### 1. Baseline evaluation (recommended first step)

Uses the released CSVs under `dataset/` and an OpenAI-compatible API (e.g. DeepSeek).

```bash
cd evaluation
pip install -r requirements.txt
# Copy ../datapipeline/.env.example or evaluation/.env.example to repo root as .env and set OPENAI_API_KEY

python evaluate_bsqa_zero_shot.py --split all --csv-dir ../dataset
```

Per-example JSON, split summaries, and `evaluation_master_summary.json` are written under `evaluation/results/<model_slug>/` by default.

### 2. Regenerate JSONL / CSV from EconLogicQA (optional, API-heavy)

```bash
cd datapipeline
pip install -r requirements.txt
# .env at repository root (see datapipeline/.env.example)

python transform_econlogic_bsqa.py --split train --limit 1   # smoke
python verify_bsqa_output.py /path/to/train.jsonl
python clean_bsqa_to_csv.py --input-dir /path/to/jsonl --output-dir /path/to/clean_csv
```

---

## Metrics (baseline evaluator)

The zero-shot script reports **exact match** on `gold_answer`, parse/API failure counts, **per–`condition_type` accuracy**, and on the gold-**insufficient** subset **UCR** (unsupported commitment rate: predicted A/B) and **ADR** (appropriate deferral rate: predicted C/D). Protocol-only metrics that require structured belief output (e.g. SJA, BAC, citation rate) are **out of scope** for this baseline script.

---

## Citation

If you use EconLogic-BSQA, please cite the accompanying paper or technical report when available, and cite **EconLogicQA** per its authors’ instructions when using or contrasting with the source ordering task.

---

## License

This repository uses separate licenses for software and data.

- **Code license:** Source code, evaluation scripts, data-processing utilities, and other software components are licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE).
- **Data license:** The EconLogic-BSQA benchmark dataset and associated data artifacts are derived from **EconLogicQA**, which is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. Accordingly, the derived benchmark dataset and associated data artifacts are released under **CC BY-NC-SA 4.0**. See [`LICENSE-DATA`](LICENSE-DATA).

The data license applies to released benchmark artifacts, including files under `dataset/`, transformed examples, prompt-generated benchmark instances, prediction files, run metadata, and other data artifacts derived from or closely tied to EconLogicQA. Users must provide attribution to the original EconLogicQA authors, use the derived dataset only for non-commercial purposes unless separately authorized, distribute derivative datasets under the same license, and clearly indicate modifications.

The software license does **not** relicense EconLogicQA or EconLogicQA-derived data artifacts as Apache-2.0.

---

## Attribution and provenance

EconLogic-BSQA is a derivative benchmark constructed from EconLogicQA. Please cite the original EconLogicQA work when using this benchmark, especially when using, transforming, or contrasting with the source event-ordering task. A summary of third-party attribution and licensing notes is provided in [`NOTICE`](NOTICE).

---

## Contributing / issues

For bugs in the datapipeline or evaluator, open an issue on the hosting repository with a minimal reproducer (command line, split, and row limit).
