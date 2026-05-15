#!/usr/bin/env python3
"""Export cleaned EconLogic-BSQA JSONL splits to CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "econlogic_bsqa_output"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "econlogic_bsqa_clean_csv"
SPLITS = ("train", "val", "test")

EXCLUDED_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "instance_id": "train_0_full_context_0",
        "source_id": "train_0",
        "source_question": (
            "Coca-Cola and Molson Coors are partnering to launch Simply Spiked "
            "Lemonade, an alcoholic beverage. Arrange the following events in "
            "the logical sequence they would occur in the process of bringing "
            "this product to market."
        ),
        "source_events": {
            "A": "The companies identify a market opportunity for a new alcoholic beverage.",
            "B": "The companies develop the product, including determining the flavors and alcohol content.",
            "C": "The companies launch the product, making it available for purchase.",
            "D": "The companies analyze the performance of the product in the market.",
        },
        "source_gold_order": ["A", "B", "C", "D"],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean EconLogic-BSQA JSONL splits and export train/val/test CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl. Default: <repo>/econlogic_bsqa_output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cleaned CSV files. Default: <repo>/econlogic_bsqa_clean_csv",
    )
    parser.add_argument(
        "--keep-validation-failures",
        action="store_true",
        help="Keep rows whose _validation.valid or _validation.final_keep is false.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} is not a JSON object")
            records.append(record)
    return records


def matches_excluded_record(record: dict[str, Any]) -> bool:
    return any(all(record.get(key) == value for key, value in excluded.items()) for excluded in EXCLUDED_RECORDS)


def has_validation_failure(record: dict[str, Any]) -> bool:
    validation = record.get("_validation")
    if not isinstance(validation, dict):
        return False
    return validation.get("valid") is False or validation.get("final_keep") is False


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def collect_fieldnames(split_records: dict[str, list[dict[str, Any]]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for split in SPLITS:
        for record in split_records[split]:
            for key in record:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    return fieldnames


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: serialize_cell(record.get(key)) for key in fieldnames})


def clean_records(
    records: list[dict[str, Any]],
    *,
    keep_validation_failures: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_instance_ids: set[str] = set()

    for record in records:
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            counts["missing_instance_id"] += 1
            continue
        if instance_id in seen_instance_ids:
            counts["duplicate_instance_id"] += 1
            continue
        seen_instance_ids.add(instance_id)

        if matches_excluded_record(record):
            counts["excluded_target_record"] += 1
            continue
        if not keep_validation_failures and has_validation_failure(record):
            counts["validation_failure"] += 1
            continue

        cleaned.append(record)
        counts["kept"] += 1

    return cleaned, counts


def assert_excluded_records_absent(split_records: dict[str, list[dict[str, Any]]]) -> None:
    remaining = [
        record.get("instance_id", "<missing>")
        for records in split_records.values()
        for record in records
        if matches_excluded_record(record)
    ]
    if remaining:
        raise ValueError(f"Excluded records remain in cleaned output: {remaining}")


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"input directory not found: {input_dir}", file=sys.stderr)
        return 2

    cleaned_by_split: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    try:
        for split in SPLITS:
            input_path = input_dir / f"{split}.jsonl"
            if not input_path.is_file():
                raise FileNotFoundError(f"missing split file: {input_path}")
            raw_records = read_jsonl(input_path)
            cleaned_records, counts = clean_records(
                raw_records,
                keep_validation_failures=args.keep_validation_failures,
            )
            cleaned_by_split[split] = cleaned_records
            report["splits"][split] = {
                "input_rows": len(raw_records),
                "output_rows": len(cleaned_records),
                "removed_rows": len(raw_records) - len(cleaned_records),
                "counts": dict(counts),
            }

        assert_excluded_records_absent(cleaned_by_split)
        fieldnames = collect_fieldnames(cleaned_by_split)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            write_csv(output_dir / f"{split}_clean.csv", cleaned_by_split[split], fieldnames)

        report["columns"] = fieldnames
        report_path = output_dir / "cleaning_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (OSError, ValueError) as exc:
        print(f"cleaning failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
