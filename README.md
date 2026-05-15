# Open-source bundles

| Path | Description |
|------|-------------|
| [`dataset/`](dataset/) | Released cleaned CSV splits (`*_clean.csv`). |
| [`datapipeline/`](datapipeline/) | EconLogicQA → EconLogic-BSQA transformation, JSONL verification, CSV export. |
| [`evaluation/`](evaluation/) | Zero-shot baseline evaluation on cleaned CSVs (no belief-driven protocol). |

See each folder’s `README.md` for setup and commands. Do not commit API keys; use `.env` only at the repository root (see `.env.example` files under `datapipeline/` and `evaluation/`).
