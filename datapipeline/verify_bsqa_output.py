#!/usr/bin/env python3
"""Verify BSQA JSONL: one JSON per line, duplicate instance_id check, required keys."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_KEYS = (
    "instance_id",
    "source_id",
    "source_question",
    "source_events",
    "source_gold_order",
    "scenario_type",
    "abstract_domain",
    "abstract_entities",
    "abstracted_events",
    "ordered_chain",
    "local_rules",
    "condition_type",
    "observed_events",
    "decision_question",
    "answer_choices",
    "gold_answer",
    "gold_sufficiency",
    "gold_decision",
    "required_rules",
    "required_events",
    "missing_information",
    "global_knowledge_leakage_trap",
    "qa_text",
)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify BSQA JSONL output")
    p.add_argument("jsonl", type=Path, help="Path to *.jsonl")
    args = p.parse_args()
    path: Path = args.jsonl
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    seen: dict[str, int] = {}
    n = 0
    errors = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"L{lineno} json_error: {e}", file=sys.stderr)
                errors += 1
                continue
            if not isinstance(obj, dict):
                print(f"L{lineno} not_object", file=sys.stderr)
                errors += 1
                continue
            iid = str(obj.get("instance_id", ""))
            if not iid:
                print(f"L{lineno} missing instance_id", file=sys.stderr)
                errors += 1
            else:
                seen[iid] = seen.get(iid, 0) + 1
            missing = [k for k in REQUIRED_KEYS if k not in obj]
            if missing:
                print(f"L{lineno} instance_id={iid!r} missing: {missing[:5]}...", file=sys.stderr)
                errors += 1

    dups = [iid for iid, c in seen.items() if c > 1]
    print(json.dumps({"path": str(path), "lines": n, "duplicate_ids": dups, "errors": errors}, indent=2))
    return 1 if errors or dups else 0


if __name__ == "__main__":
    raise SystemExit(main())
