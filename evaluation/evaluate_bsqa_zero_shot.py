#!/usr/bin/env python3
"""Baseline zero-shot evaluation for EconLogic-BSQA (cleaned CSV splits).

Reads ``*_clean.csv``, calls an OpenAI-compatible chat API, parses ``Final answer: X``.
Does **not** implement belief-driven JSON evaluation.

Outputs default to ``opensource/evaluation/results/<sanitized_model>/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CSV_DIR = REPO_ROOT / "opensource" / "dataset"
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "results"


def slugify_model_name(model: str) -> str:
    s = (model or "unknown").strip()
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.ASCII)
    s = s.strip("._-") or "unknown_model"
    return s


ZERO_SHOT_SYSTEM = (
    "You are a careful assistant for closed-world business scenarios. "
    "Answer using only the scenario text provided. "
    "Your final line must be exactly: Final answer: X where X is A, B, C, or D."
)

ZERO_SHOT_USER_TEMPLATE = """You are given a business scenario.

{qa_text}

Answer the question directly. Choose one answer: A, B, C, or D.
Provide a brief justification, then end with a line exactly:
Final answer: X"""

SPLITS = ("train", "val", "test")


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_final_answer_zero_shot(text: str) -> str | None:
    m = re.search(r"Final\s*answer\s*:\s*([ABCD])\b", text, re.I)
    if m:
        return m.group(1).upper()
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if re.fullmatch(r"[ABCD]", line, re.I):
            return line.upper()
    letters = re.findall(r"\b([ABCD])\b", text.upper())
    return letters[-1] if letters else None


def call_chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    timeout_s: int,
) -> str:
    resp = client.with_options(timeout=timeout_s).chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def evaluate_one_row(
    client: OpenAI,
    model: str,
    row: dict[str, str],
    row_index: int,
    timeout_s: int,
    max_retries: int,
) -> dict[str, Any]:
    qa_text = (row.get("qa_text") or "").strip()
    gold = (row.get("gold_answer") or "").strip().upper()
    gold_suff = (row.get("gold_sufficiency") or "").strip().lower()
    gold_decision = (row.get("gold_decision") or "").strip().lower()
    condition_type = (row.get("condition_type") or "").strip()
    instance_id = row.get("instance_id") or f"row_{row_index}"

    if gold not in ("A", "B", "C", "D"):
        return {
            "instance_id": instance_id,
            "row_index": row_index,
            "gold_answer": gold,
            "mode": "zero_shot",
            "parse_failure": True,
            "api_failure": False,
            "correct": False,
            "error": "invalid_gold_answer",
            "raw": "",
        }

    user_content = ZERO_SHOT_USER_TEMPLATE.format(qa_text=qa_text)
    raw = ""
    last_err = ""
    for attempt in range(max_retries):
        try:
            raw = call_chat(
                client, model, ZERO_SHOT_SYSTEM, user_content, timeout_s=timeout_s
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(1.2 * (attempt + 1))

    if raw == "":
        return {
            "instance_id": instance_id,
            "row_index": row_index,
            "gold_answer": gold,
            "gold_sufficiency": gold_suff,
            "gold_decision": gold_decision,
            "condition_type": condition_type,
            "mode": "zero_shot",
            "parse_failure": False,
            "api_failure": True,
            "correct": False,
            "error": last_err,
            "raw": "",
        }

    pred = extract_final_answer_zero_shot(raw)
    parse_failure = pred is None
    correct = pred == gold if pred else False

    return {
        "instance_id": instance_id,
        "row_index": row_index,
        "gold_answer": gold,
        "gold_sufficiency": gold_suff,
        "gold_decision": gold_decision,
        "condition_type": condition_type,
        "mode": "zero_shot",
        "pred_answer": pred,
        "parse_failure": parse_failure,
        "api_failure": False,
        "correct": correct,
        "raw": raw,
    }


def safe_filename(instance_id: str) -> str:
    return re.sub(r"[^\w\-.]", "_", instance_id)[:200]


def result_path(results_dir: Path, split: str, instance_id: str) -> Path:
    return results_dir / "zero_shot" / split / f"{safe_filename(instance_id)}.json"


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    parse_fail = sum(1 for r in results if r.get("parse_failure"))
    api_fail = sum(1 for r in results if r.get("api_failure"))

    insufficient_ctx = [
        r
        for r in results
        if not r.get("api_failure")
        and (r.get("gold_sufficiency") or "").lower() == "insufficient"
    ]
    n_insuff = len(insufficient_ctx)
    ucr = 0
    adr = 0
    for r in insufficient_ctx:
        pred = r.get("pred_answer")
        if isinstance(pred, str) and pred in ("A", "B", "C", "D"):
            if pred in ("A", "B"):
                ucr += 1
            if pred in ("C", "D"):
                adr += 1

    by_condition: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        if r.get("api_failure"):
            continue
        ct = str(r.get("condition_type") or "unknown")
        by_condition[ct]["n"] += 1
        if r.get("correct"):
            by_condition[ct]["correct"] += 1

    cond_accuracy = {
        k: (v["correct"] / v["n"] if v["n"] else 0.0) for k, v in by_condition.items()
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / max(1, total - api_fail) if total else 0.0,
        "accuracy_all_rows": correct / max(1, total),
        "parse_failures": parse_fail,
        "api_failures": api_fail,
        "insufficient_context_count": n_insuff,
        "unsupported_commitment_rate": ucr / max(1, n_insuff) if n_insuff else None,
        "appropriate_deferral_rate": adr / max(1, n_insuff) if n_insuff else None,
        "by_condition_type": {k: {"n": v["n"], "correct": v["correct"]} for k, v in by_condition.items()},
        "condition_type_accuracy": cond_accuracy,
    }


def run_split(
    client: OpenAI,
    model: str,
    split: str,
    csv_dir: Path,
    results_dir: Path,
    workers: int,
    timeout_s: int,
    retries: int,
    limit: int | None,
    force: bool,
) -> dict[str, Any]:
    csv_path = csv_dir / f"{split}_clean.csv"
    rows = read_csv_rows(csv_path)
    if limit is not None:
        rows = rows[:limit]

    out_dir = results_dir / "zero_shot" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run: list[tuple[int, dict[str, str]]] = []
    for i, row in enumerate(rows):
        iid = row.get("instance_id") or f"row_{i}"
        rp = result_path(results_dir, split, iid)
        if not force and rp.is_file():
            try:
                json.loads(rp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                to_run.append((i, row))
        else:
            to_run.append((i, row))

    if to_run:

        def _job(item: tuple[int, dict[str, str]]) -> tuple[str, dict[str, Any]]:
            idx, r = item
            res = evaluate_one_row(
                client, model, r, idx, timeout_s=timeout_s, max_retries=retries
            )
            rp = result_path(results_dir, split, str(res["instance_id"]))
            rp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(res["instance_id"]), res

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(_job, item) for item in to_run]
            for fut in tqdm(
                concurrent.futures.as_completed(futs),
                total=len(futs),
                desc=f"zero_shot/{split}",
            ):
                fut.result()

    all_results: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        iid = row.get("instance_id") or f"row_{i}"
        rp = result_path(results_dir, split, iid)
        if rp.is_file():
            all_results.append(json.loads(rp.read_text(encoding="utf-8")))
        else:
            all_results.append(
                {
                    "instance_id": iid,
                    "row_index": i,
                    "error": "missing_result_file",
                    "api_failure": True,
                    "correct": False,
                }
            )

    metrics = aggregate(all_results)
    summary_path = results_dir / "zero_shot" / f"{split}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": split,
                "mode": "zero_shot",
                "model": model,
                "csv_path": str(csv_path),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"split": split, "metrics": metrics, "summary_path": str(summary_path)}


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline evaluation for EconLogic-BSQA (clean CSV)"
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Output root for this run. Default: "
            "opensource/evaluation/results/<sanitized model name> "
            "(under this repository)"
        ),
    )
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if result JSON exists")
    args = parser.parse_args()

    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else DEFAULT_RESULTS_ROOT / slugify_model_name(args.model)
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("OPENAI_API_KEY missing in .env", flush=True)
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)
    splits = list(SPLITS) if args.split == "all" else [args.split]

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"results_dir: {results_dir}", flush=True)

    master: dict[str, Any] = {
        "model": args.model,
        "base_url": base_url,
        "results_dir": str(results_dir),
        "protocol": "zero_shot_baseline_only",
        "splits": {},
    }

    for split in splits:
        csv_path = args.csv_dir / f"{split}_clean.csv"
        if not csv_path.is_file():
            print(f"Missing {csv_path}", flush=True)
            return 3
        out = run_split(
            client=client,
            model=args.model,
            split=split,
            csv_dir=args.csv_dir,
            results_dir=results_dir,
            workers=args.workers,
            timeout_s=args.timeout,
            retries=args.retries,
            limit=args.limit,
            force=args.force,
        )
        master["splits"][split] = out["metrics"]
        print(json.dumps({split: out["metrics"]}, ensure_ascii=False, indent=2))

    master_path = results_dir / "evaluation_master_summary.json"
    master_path.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved master summary: {master_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
