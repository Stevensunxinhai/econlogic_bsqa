#!/usr/bin/env python3
"""
Batch-transform EconLogicQA CSV rows into EconLogic-BSQA JSONL using DeepSeek
(OpenAI-compatible API). Follows prompts in econlogic_bsqa_transformation_protocol.md.

Features:
- Prompts 1-9 pipeline per row (filter, abstract, skeleton, rules, conditions, final QA, sequential, validate)
- Parallel row processing (default 20 workers)
- Unique instance_id, dedupe, atomic JSONL writes
- tqdm progress bar
- Optional real-time token throughput (from API usage), see --token-report-interval
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# This package lives at opensource/datapipeline/; repository root is two levels up.
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "econlogic_bsqa_output"


class TokenMonitor:
    """Thread-safe counters from chat.completions usage; supports rolling tok/s."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt = 0
        self._completion = 0
        self._total = 0
        self._requests = 0
        self.t0 = time.perf_counter()

    def record(self, usage: Any) -> None:
        if usage is None:
            return
        pt = int(getattr(usage, "prompt_tokens", None) or 0)
        ct = int(getattr(usage, "completion_tokens", None) or 0)
        tt = getattr(usage, "total_tokens", None)
        if tt is None:
            tt = pt + ct
        else:
            tt = int(tt)
        with self._lock:
            self._prompt += pt
            self._completion += ct
            self._total += tt
            self._requests += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self._prompt,
                "completion_tokens": self._completion,
                "total_tokens": self._total,
                "requests": self._requests,
            }


def _token_reporter_loop(
    monitor: TokenMonitor,
    stop: threading.Event,
    interval_s: float,
) -> None:
    last_t = time.perf_counter()
    last_total = 0
    while not stop.wait(interval_s):
        now = time.perf_counter()
        snap = monitor.snapshot()
        tot = snap["total_tokens"]
        dt = max(now - last_t, 1e-9)
        dtot = tot - last_total
        inst = dtot / dt
        wall = max(now - monitor.t0, 1e-9)
        avg = tot / wall
        last_t = now
        last_total = tot
        tqdm.write(
            f"[tokens] {inst:.1f} tok/s (window) | {avg:.1f} tok/s (wall avg) | "
            f"total={tot} prompt={snap['prompt_tokens']} completion={snap['completion_tokens']} "
            f"requests={snap['requests']}",
            file=sys.stderr,
        )


SYSTEM_PROMPT_0 = """You are transforming event-ordering QA data into closed-world business decision-making QA data.

Your goal is not to answer the original question. Your goal is to create new dataset instances that test whether a model can decide if it has sufficient belief to make a business or policy-relevant decision.

Important constraints:
1. Do not preserve real-world entity names unless they are generic business concepts.
2. Replace real companies, people, countries, institutions, and historically specific events with fictional abstract entities.
3. Do not rely on external world knowledge.
4. Preserve only the causal, procedural, or decision-relevant structure of the original event chain.
5. Each transformed instance must be solvable only from the provided local rules and observed events.
6. Each transformed instance must include verifiable signals: required rules, required events, missing information, and gold sufficiency label.
7. Avoid making the answer obvious from commonsense alone.
8. If the sample cannot be transformed into a meaningful business decision scenario, reject it (keep=false in filtering).
"""


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating markdown fences and leading prose."""
    raw = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?", raw, re.IGNORECASE)
    if fence:
        raw = raw[fence.end() :]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()

    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    def _scan(start_char: str, end_char: str) -> Any | None:
        start = raw.find(start_char)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                quote = ch
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    for pair in (("{", "}"), ("[", "]")):
        obj = _scan(pair[0], pair[1])
        if obj is not None:
            return obj

    raise ValueError("Could not parse JSON from model response")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_gold_order(answer_field: str) -> list[str]:
    parts = [p.strip() for p in answer_field.replace("，", ",").split(",")]
    return [p for p in parts if p]


def prompt_filter(row: dict[str, str]) -> str:
    q = row.get("Question", "").strip()
    return f"""Given the original EconLogicQA sample below, decide whether it should be kept for transformation into a closed-world business belief-sufficiency QA instance.

Original Question:
{q}

Events:
A. {row.get("A", "").strip()}
B. {row.get("B", "").strip()}
C. {row.get("C", "").strip()}
D. {row.get("D", "").strip()}

Gold Order:
{row.get("Answer", "").strip()}

Classify the sample.

Return JSON only:

{{
  "keep": true,
  "reason": "...",
  "business_relevance": "high/medium/low/none",
  "scenario_type": "...",
  "decision_potential": "high/medium/low/none",
  "exclude_if": [],
  "transformable_business_decision": "..."
}}

Keep the sample only if:
- it contains a business, economic, operational, market, platform, regulatory, supply-chain, product, or policy-relevant decision structure;
- the events can be interpreted as a causal or procedural chain;
- a decision question can be generated from the chain;
- at least one event or rule can be removed to create an insufficient-belief condition.

Reject if:
- it is merely a historical timeline;
- it depends mainly on named people or political chronology;
- it has no business or market consequence;
- it cannot support a decision-sufficiency task.
"""


def prompt_abstraction(row: dict[str, str]) -> str:
    q = row.get("Question", "").strip()
    return f"""Transform the kept sample into an abstract closed-world business scenario.

Original Question:
{q}

Events:
A. {row.get("A", "").strip()}
B. {row.get("B", "").strip()}
C. {row.get("C", "").strip()}
D. {row.get("D", "").strip()}

Gold Order:
{row.get("Answer", "").strip()}

Instructions:
1. Replace real entities with fictional neutral entities.
2. Replace countries, companies, public figures, institutions, and domain-specific objects with abstract names.
3. Preserve causal or procedural relationships.
4. Remove details that would trigger real-world knowledge.
5. Keep the transformed events understandable as a business or policy-relevant decision scenario.
6. Do not introduce facts not implied by the original event chain.

Return JSON only:

{{
  "abstract_domain": "...",
  "abstract_entities": {{}},
  "abstracted_events": {{"A":"...","B":"...","C":"...","D":"..."}},
  "gold_order_abstracted": ["A","B","C","D"],
  "removed_domain_specific_cues": [],
  "remaining_leakage_risks": []
}}
"""


def prompt_skeleton(abstracted_events: dict[str, str], gold_order: list[str]) -> str:
    ga = abstracted_events.get("A", "")
    gb = abstracted_events.get("B", "")
    gc = abstracted_events.get("C", "")
    gd = abstracted_events.get("D", "")
    go = ", ".join(gold_order)
    return f"""Given the abstracted events and the gold order, extract the causal or procedural skeleton.

Abstracted Events:
A. {ga}
B. {gb}
C. {gc}
D. {gd}

Gold Order:
{go}

Return JSON only:

{{
  "ordered_chain": [
    {{
      "event_id": "E1",
      "source_option": "A",
      "event_text": "...",
      "role": "upstream_condition",
      "decision_relevance": "high",
      "why_decision_relevant": "..."
    }}
  ],
  "causal_structure_type": "linear",
  "critical_events": ["E1"],
  "removable_events_for_insufficient_condition": ["E2"],
  "decision_target_candidates": [
    "Should Firm A proceed with launching Product P?"
  ]
}}
"""


def prompt_local_rules(ordered_chain: list[dict[str, Any]], decision_target: str) -> str:
    return f"""Generate local business rules for the abstracted scenario.

Ordered Chain:
{json.dumps(ordered_chain, ensure_ascii=False)}

Decision Target:
{decision_target}

Return JSON only:

{{
  "local_rules": [
    {{
      "rule_id": "R1",
      "rule_text": "...",
      "rule_type": "dependency",
      "required_for_decision": true
    }}
  ],
  "decision_threshold_rule": "...",
  "minimum_sufficiency_conditions": [],
  "possible_missing_information": [],
  "closed_world_instruction": "Use only these local rules and observed events. Do not use external business assumptions."
}}

Include at least 3 rules. Include at least one decision_threshold or sufficiency_requirement rule.
"""


def prompt_multi_conditions(
    ordered_chain: list[dict[str, Any]],
    local_rules: list[dict[str, Any]],
    decision_target: str,
) -> str:
    return f"""Using the abstracted event chain and local rules, generate multiple belief-sufficiency QA conditions.

Ordered Chain:
{json.dumps(ordered_chain, ensure_ascii=False)}

Local Rules (baseline):
{json.dumps(local_rules, ensure_ascii=False)}

Decision Target:
{decision_target}

Required condition_type values (one object each when possible):
1. full_context
2. missing_evidence
3. missing_rule
4. ambiguous_evidence
5. counterintuitive_rule
6. sequential_reveal_step (if possible)

For each condition object include:
- condition_type (string)
- local_rules_shown: array of rule objects exactly as shown to the downstream model for this condition
- observed_event_ids: array of event_id strings like ["E1","E2"]
- decision_question: string
- gold_sufficiency: "sufficient" or "insufficient"
- gold_decision: one of proceed / do_not_proceed / request_more_information / defer_or_escalate
- gold_answer: one of A/B/C/D where A=sufficient+proceed, B=sufficient+do_not_proceed, C=insufficient+request_more_information, D=insufficient+defer_or_escalate
- required_rules: list of rule_id strings
- required_events: list of event_id strings
- missing_information: list of strings (non-empty for insufficient cases when applicable)
- global_knowledge_leakage_trap: boolean

Return JSON array only.
"""


def prompt_final_qa_instance(
    condition_type: str,
    local_rules: list[dict[str, Any]],
    observed_event_ids: list[str],
    ordered_chain: list[dict[str, Any]],
    decision_target: str,
) -> str:
    ev_map = {e.get("event_id"): e.get("event_text", "") for e in ordered_chain if isinstance(e, dict)}
    obs_lines = []
    for eid in observed_event_ids:
        obs_lines.append(f"{eid}: {ev_map.get(eid, '')}")
    obs_block = "\n".join(obs_lines)
    rules_block = json.dumps(local_rules, ensure_ascii=False)
    return f"""Create the final closed-world business belief-sufficiency QA instance.

Condition Type:
{condition_type}

Local Rules:
{rules_block}

Observed Events (id -> text):
{obs_block}

Decision Target:
{decision_target}

Generate the final QA in this format:

Business Micro-World:
You are evaluating a decision in a fictional business environment. Use only the local rules and observed events below. Do not use outside knowledge, real-world assumptions, or general business intuition.

Local Business Rules:
R1. ...
Observed Events:
E1. ...
Question:
Based only on the local rules and observed events, is there sufficient belief to decide whether {decision_target}?

Answer Choices:
A. Sufficient belief; proceed with the decision.
B. Sufficient belief; do not proceed with the decision.
C. Insufficient belief; request more information.
D. Insufficient belief; defer or escalate the decision.

Return JSON only:

{{
  "qa_text": "...",
  "answer_choices": {{
    "A": "Sufficient belief; proceed with the decision",
    "B": "Sufficient belief; do not proceed with the decision",
    "C": "Insufficient belief; request more information",
    "D": "Insufficient belief; defer or escalate the decision"
  }},
  "gold_answer": "A",
  "gold_sufficiency": "sufficient",
  "gold_decision": "proceed",
  "required_rules": [],
  "required_events": [],
  "missing_information": [],
  "condition_type": "{condition_type}",
  "global_knowledge_leakage_trap": true
}}
"""


def prompt_counterintuitive(
    ordered_chain: list[dict[str, Any]],
    decision_target: str,
) -> str:
    return f"""Create a counterintuitive closed-world version of the scenario.

Original abstracted chain:
{json.dumps(ordered_chain, ensure_ascii=False)}

Decision Target:
{decision_target}

Return JSON only:

{{
  "counterintuitive_rules": [],
  "observed_events": [],
  "decision_question": "...",
  "gold_sufficiency": "sufficient",
  "gold_decision": "proceed",
  "why_common_sense_would_fail": "...",
  "required_rule_following": []
}}
"""


def prompt_sequential_belief(
    ordered_chain: list[dict[str, Any]],
    local_rules: list[dict[str, Any]],
    decision_target: str,
) -> str:
    return f"""Create a sequential belief-update version of the QA instance.

Ordered Chain:
{json.dumps(ordered_chain, ensure_ascii=False)}

Local Rules:
{json.dumps(local_rules, ensure_ascii=False)}

Decision Target:
{decision_target}

Generate T0, T1, T2, T3 states where observed events are revealed progressively.

Return JSON only:

{{
  "sequential_instance_id": "seq_1",
  "decision_target": "...",
  "local_rules": [],
  "steps": [
    {{
      "step": "T0",
      "observed_events": [],
      "gold_sufficiency": "...",
      "gold_decision": "...",
      "missing_information": [],
      "expected_belief_direction": "uncertainty_should_remain_high"
    }}
  ],
  "final_gold_answer": "A"
}}
"""


def prompt_quality_validation(instance: dict[str, Any]) -> str:
    return f"""Validate the transformed closed-world business belief-sufficiency QA instance.

Instance:
{json.dumps(instance, ensure_ascii=False)}

Check the criteria in the protocol (abstraction, closed-world solvability, verifiable gold, sufficiency judgment, missing-evidence/missing-rule integrity, counterintuitive where applicable, leakage, triviality).

Return JSON only:

{{
  "valid": true,
  "quality_score": 5,
  "failure_reasons": [],
  "leakage_risks": [],
  "triviality_risk": "low",
  "recommended_fix": "",
  "final_keep": true
}}
"""


def call_chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 8192,
    timeout_s: int = 120,
) -> tuple[str, Any]:
    resp = client.with_options(timeout=timeout_s).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content
    if not content:
        raise RuntimeError("Empty model response")
    usage = getattr(resp, "usage", None)
    return content, usage


def call_chat_retry(
    client: OpenAI,
    model: str,
    user: str,
    *,
    max_tokens: int = 8192,
    timeout_s: int = 120,
    retries: int = 2,
    system: str = SYSTEM_PROMPT_0,
    token_monitor: TokenMonitor | None = None,
) -> str:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            content, usage = call_chat(
                client, model, system, user, max_tokens=max_tokens, timeout_s=timeout_s
            )
            if token_monitor is not None:
                token_monitor.record(usage)
            return content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def normalize_gold_order_abstracted(abstraction: dict[str, Any], row: dict[str, str]) -> list[str]:
    go = abstraction.get("gold_order_abstracted")
    if isinstance(go, list) and len(go) == 4:
        return [str(x).strip().upper()[:1] for x in go]
    return parse_gold_order(row.get("Answer", ""))


def normalize_conditions_payload(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    if isinstance(raw, dict):
        if "conditions" in raw and isinstance(raw["conditions"], list):
            return [c for c in raw["conditions"] if isinstance(c, dict)]
        if "instances" in raw and isinstance(raw["instances"], list):
            return [c for c in raw["instances"] if isinstance(c, dict)]
    return []


GOLD_ANSWER_TO_SUFF_ACTION: dict[str, tuple[str, str]] = {
    "A": ("sufficient", "proceed"),
    "B": ("sufficient", "do_not_proceed"),
    "C": ("insufficient", "request_more_information"),
    "D": ("insufficient", "defer_or_escalate"),
}


def normalize_decision_action(s: str) -> str:
    t = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "donotproceed": "do_not_proceed",
        "do_not_proceed": "do_not_proceed",
        "requestmoreinformation": "request_more_information",
        "request_more_information": "request_more_information",
        "deferorescalate": "defer_or_escalate",
        "defer_or_escalate": "defer_or_escalate",
        "proceed": "proceed",
    }
    return mapping.get(t, t)


def local_schema_validate(inst: dict[str, Any]) -> tuple[bool, list[str]]:
    errs: list[str] = []
    required_top = [
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
    ]
    for k in required_top:
        if k not in inst:
            errs.append(f"missing_field:{k}")

    ga = str(inst.get("gold_answer", "")).strip().upper()[:1]
    if ga not in GOLD_ANSWER_TO_SUFF_ACTION:
        errs.append("gold_answer_not_ABCD")

    exp = GOLD_ANSWER_TO_SUFF_ACTION.get(ga)
    if exp:
        if inst.get("gold_sufficiency") != exp[0]:
            errs.append("gold_sufficiency_mismatch")
        if normalize_decision_action(str(inst.get("gold_decision", ""))) != exp[1]:
            errs.append("gold_decision_mismatch")

    lr = inst.get("local_rules")
    if not isinstance(lr, list) or len(lr) < 3:
        errs.append("local_rules_lt3")

    oe = inst.get("observed_events")
    if not isinstance(oe, list) or len(oe) < 2:
        errs.append("observed_events_lt2")

    rr = inst.get("required_rules")
    re = inst.get("required_events")
    if not isinstance(rr, list) or len(rr) < 1:
        errs.append("required_rules_empty")
    if not isinstance(re, list) or len(re) < 1:
        errs.append("required_events_empty")

    ct = str(inst.get("condition_type", ""))
    if ct in ("missing_evidence", "missing_rule", "ambiguous_evidence"):
        mi = inst.get("missing_information")
        if not isinstance(mi, list) or len(mi) == 0:
            errs.append("missing_information_empty_for_condition")

    ac = inst.get("answer_choices")
    if not isinstance(ac, dict) or set(ac.keys()) != {"A", "B", "C", "D"}:
        errs.append("answer_choices_invalid")

    return len(errs) == 0, errs


def merge_full_instance(
    *,
    split: str,
    row_index: int,
    source_id: str,
    instance_id: str,
    row: dict[str, str],
    filter_obj: dict[str, Any],
    abstraction: dict[str, Any],
    skeleton: dict[str, Any],
    baseline_rules: list[dict[str, Any]],
    cond: dict[str, Any],
    p6: dict[str, Any],
    sequential_reveal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src_events = {
        "A": row.get("A", "").strip(),
        "B": row.get("B", "").strip(),
        "C": row.get("C", "").strip(),
        "D": row.get("D", "").strip(),
    }
    out: dict[str, Any] = {
        "instance_id": instance_id,
        "source_id": source_id,
        "source_question": row.get("Question", "").strip(),
        "source_events": src_events,
        "source_gold_order": parse_gold_order(row.get("Answer", "")),
        "scenario_type": filter_obj.get("scenario_type", "operations"),
        "abstract_domain": abstraction.get("abstract_domain", ""),
        "abstract_entities": abstraction.get("abstract_entities") or {},
        "abstracted_events": abstraction.get("abstracted_events") or {},
        "ordered_chain": skeleton.get("ordered_chain") or [],
        "local_rules": cond.get("local_rules_shown") or baseline_rules,
        "condition_type": cond.get("condition_type") or p6.get("condition_type", ""),
        "observed_events": cond.get("observed_event_ids") or [],
        "decision_question": cond.get("decision_question", ""),
        "answer_choices": p6.get("answer_choices", {}),
        "gold_answer": str(p6.get("gold_answer", "")).strip().upper()[:1],
        "gold_sufficiency": p6.get("gold_sufficiency", ""),
        "gold_decision": p6.get("gold_decision", ""),
        "required_rules": p6.get("required_rules", []),
        "required_events": p6.get("required_events", []),
        "missing_information": p6.get("missing_information", []),
        "global_knowledge_leakage_trap": bool(
            p6.get("global_knowledge_leakage_trap", cond.get("global_knowledge_leakage_trap", False))
        ),
        "qa_text": p6.get("qa_text", ""),
    }
    if sequential_reveal is not None:
        out["sequential_reveal"] = sequential_reveal
    return out


@dataclass
class RowTransformResult:
    split: str
    row_index: int
    source_id: str
    instances: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    row_error: str | None = None


def transform_one_row(
    client: OpenAI,
    model: str,
    split: str,
    row_index: int,
    row: dict[str, str],
    *,
    timeout_s: int,
    retries: int,
    token_monitor: TokenMonitor | None = None,
) -> RowTransformResult:
    source_id = f"{split}_{row_index}"
    out = RowTransformResult(split=split, row_index=row_index, source_id=source_id)

    def reject(reason: str, **extra: Any) -> None:
        rec: dict[str, Any] = {
            "split": split,
            "row_index": row_index,
            "source_id": source_id,
            "reason": reason,
        }
        rec.update(extra)
        out.rejects.append(rec)

    try:
        filt_raw = call_chat_retry(
            client,
            model,
            prompt_filter(row),
            timeout_s=timeout_s,
            retries=retries,
            max_tokens=2048,
            token_monitor=token_monitor,
        )
        filt = extract_json(filt_raw)
        if not isinstance(filt, dict):
            reject("filter_not_object")
            return out
        if not filt.get("keep"):
            reject("filtered_out", filter=filt)
            return out

        abs_raw = call_chat_retry(
            client,
            model,
            prompt_abstraction(row),
            timeout_s=timeout_s,
            retries=retries,
            max_tokens=4096,
            token_monitor=token_monitor,
        )
        abstraction = extract_json(abs_raw)
        if not isinstance(abstraction, dict):
            reject("abstraction_not_object")
            return out

        abs_events = abstraction.get("abstracted_events")
        if not isinstance(abs_events, dict):
            reject("abstraction_no_events")
            return out

        gold_order = normalize_gold_order_abstracted(abstraction, row)
        if len(gold_order) != 4:
            reject("invalid_gold_order", gold_order=gold_order)
            return out

        sk_raw = call_chat_retry(
            client,
            model,
            prompt_skeleton(abs_events, gold_order),
            timeout_s=timeout_s,
            retries=retries,
            max_tokens=4096,
            token_monitor=token_monitor,
        )
        skeleton = extract_json(sk_raw)
        if not isinstance(skeleton, dict):
            reject("skeleton_not_object")
            return out

        ordered_chain = skeleton.get("ordered_chain")
        if not isinstance(ordered_chain, list) or len(ordered_chain) < 2:
            reject("skeleton_chain_invalid")
            return out

        candidates = skeleton.get("decision_target_candidates")
        if not isinstance(candidates, list) or not candidates:
            reject("no_decision_target_candidates")
            return out
        decision_target = str(candidates[0])

        rules_raw = call_chat_retry(
            client,
            model,
            prompt_local_rules(ordered_chain, decision_target),
            timeout_s=timeout_s,
            retries=retries,
            max_tokens=4096,
            token_monitor=token_monitor,
        )
        rules_obj = extract_json(rules_raw)
        if not isinstance(rules_obj, dict):
            reject("rules_not_object")
            return out
        baseline_rules = rules_obj.get("local_rules")
        if not isinstance(baseline_rules, list) or len(baseline_rules) < 3:
            reject("rules_lt3")
            return out

        mc_raw = call_chat_retry(
            client,
            model,
            prompt_multi_conditions(ordered_chain, baseline_rules, decision_target),
            timeout_s=timeout_s,
            retries=retries,
            max_tokens=8192,
            token_monitor=token_monitor,
        )
        conditions = normalize_conditions_payload(extract_json(mc_raw))
        if not conditions:
            reject("no_conditions_from_prompt5")
            return out

        for ci, cond in enumerate(conditions):
            ctype = str(cond.get("condition_type", "unknown")).strip()
            suffix = f"{ctype}_{ci}"
            instance_id = f"{split}_{row_index}_{suffix}"

            try:
                lr_shown = cond.get("local_rules_shown")
                if not isinstance(lr_shown, list) or not lr_shown:
                    lr_shown = baseline_rules
                obs_ids = cond.get("observed_event_ids")
                if not isinstance(obs_ids, list):
                    oe_alt = cond.get("observed_events")
                    if isinstance(oe_alt, list):
                        obs_ids = []
                        for x in oe_alt:
                            if isinstance(x, str):
                                obs_ids.append(x)
                            elif isinstance(x, dict) and x.get("event_id"):
                                obs_ids.append(str(x["event_id"]))
                    else:
                        obs_ids = []

                if ctype == "counterintuitive_rule":
                    try:
                        ci_raw = call_chat_retry(
                            client,
                            model,
                            prompt_counterintuitive(ordered_chain, decision_target),
                            timeout_s=timeout_s,
                            retries=retries,
                            max_tokens=4096,
                            token_monitor=token_monitor,
                        )
                        ci_obj = extract_json(ci_raw)
                        if isinstance(ci_obj, dict) and isinstance(
                            ci_obj.get("counterintuitive_rules"), list
                        ):
                            extra = ci_obj.get("counterintuitive_rules")
                            if isinstance(lr_shown, list) and extra:
                                norm_extra: list[dict[str, Any]] = []
                                for i, r in enumerate(extra):
                                    if isinstance(r, dict) and "rule_text" in r:
                                        rid = r.get("rule_id") or f"CX{i+1}"
                                        norm_extra.append(
                                            {
                                                "rule_id": str(rid),
                                                "rule_text": str(r.get("rule_text", "")),
                                                "rule_type": str(
                                                    r.get("rule_type", "exception")
                                                ),
                                                "required_for_decision": bool(
                                                    r.get("required_for_decision", True)
                                                ),
                                            }
                                        )
                                    elif isinstance(r, str):
                                        norm_extra.append(
                                            {
                                                "rule_id": f"CX{i+1}",
                                                "rule_text": r,
                                                "rule_type": "exception",
                                                "required_for_decision": True,
                                            }
                                        )
                                lr_shown = list(lr_shown) + norm_extra
                    except Exception:
                        pass

                seq_obj: dict[str, Any] | None = None
                if ctype in ("sequential_reveal_step", "sequential_step"):
                    seq_raw = call_chat_retry(
                        client,
                        model,
                        prompt_sequential_belief(ordered_chain, baseline_rules, decision_target),
                        timeout_s=timeout_s,
                        retries=retries,
                        max_tokens=4096,
                        token_monitor=token_monitor,
                    )
                    seq_obj = extract_json(seq_raw)
                    if isinstance(seq_obj, dict) and isinstance(seq_obj.get("steps"), list):
                        last_step = seq_obj["steps"][-1] if seq_obj["steps"] else {}
                        obs_ids = last_step.get("observed_events") or obs_ids
                        gd = normalize_decision_action(str(last_step.get("gold_decision", cond.get("gold_decision", ""))))
                        cond = {
                            **cond,
                            "condition_type": "sequential_step",
                            "observed_event_ids": obs_ids,
                            "gold_sufficiency": last_step.get("gold_sufficiency", cond.get("gold_sufficiency")),
                            "gold_decision": gd,
                            "gold_answer": str(seq_obj.get("final_gold_answer", cond.get("gold_answer", "A"))).strip().upper()[:1],
                            "missing_information": last_step.get("missing_information", cond.get("missing_information", [])),
                        }

                p6_raw = call_chat_retry(
                    client,
                    model,
                    prompt_final_qa_instance(
                        str(cond.get("condition_type", ctype)),
                        lr_shown,
                        [str(x) for x in obs_ids],
                        ordered_chain,
                        decision_target,
                    ),
                    timeout_s=timeout_s,
                    retries=retries,
                    max_tokens=4096,
                    token_monitor=token_monitor,
                )
                p6 = extract_json(p6_raw)
                if not isinstance(p6, dict):
                    reject("prompt6_not_object", condition_index=ci, condition_type=ctype)
                    continue
                if "gold_decision" in p6:
                    p6["gold_decision"] = normalize_decision_action(str(p6["gold_decision"]))

                merged = merge_full_instance(
                    split=split,
                    row_index=row_index,
                    source_id=source_id,
                    instance_id=instance_id,
                    row=row,
                    filter_obj=filt,
                    abstraction=abstraction,
                    skeleton=skeleton,
                    baseline_rules=baseline_rules,
                    cond=cond,
                    p6=p6,
                    sequential_reveal=seq_obj,
                )

                val_raw = call_chat_retry(
                    client,
                    model,
                    prompt_quality_validation(merged),
                    timeout_s=timeout_s,
                    retries=retries,
                    max_tokens=2048,
                    token_monitor=token_monitor,
                )
                val = extract_json(val_raw)
                if not isinstance(val, dict):
                    reject("prompt9_not_object", instance_id=instance_id)
                    continue

                merged["_validation"] = val
                ok_local, local_errs = local_schema_validate(merged)
                score = int(val.get("quality_score", 0) or 0)
                fk = bool(val.get("final_keep", False))
                valid = bool(val.get("valid", False))

                if not ok_local:
                    merged["_local_schema_errors"] = local_errs
                    reject(
                        "local_schema_failed",
                        instance_id=instance_id,
                        errors=local_errs,
                    )
                    continue

                if valid and fk and score >= 4:
                    out.instances.append(merged)
                else:
                    reject(
                        "validation_failed",
                        instance_id=instance_id,
                        validation=val,
                    )
            except Exception as exc:  # noqa: BLE001
                reject("instance_pipeline_error", condition_index=ci, error=str(exc))

        if not out.instances and not any(r.get("reason") == "filtered_out" for r in out.rejects):
            reject("no_instances_kept")

    except Exception as exc:  # noqa: BLE001
        out.row_error = str(exc)
        reject("row_fatal_error", error=str(exc))

    return out


def dedupe_by_instance_id(instances: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep highest quality_score per instance_id."""
    best: dict[str, dict[str, Any]] = {}
    for inst in instances:
        iid = str(inst.get("instance_id", ""))
        if not iid:
            continue
        val = inst.get("_validation") or {}
        score = int(val.get("quality_score", 0) or 0)
        prev = best.get(iid)
        if prev is None:
            best[iid] = inst
            continue
        pscore = int((prev.get("_validation") or {}).get("quality_score", 0) or 0)
        if score > pscore:
            best[iid] = inst
    deduped = list(best.values())
    removed = len(instances) - len(deduped)
    deduped.sort(key=lambda x: (x.get("source_id", ""), x.get("instance_id", "")))
    return deduped, removed


def atomic_write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        shutil.move(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def process_split_parallel(
    client: OpenAI,
    model: str,
    split: str,
    rows: list[dict[str, str]],
    *,
    workers: int,
    timeout_s: int,
    retries: int,
    out_dir: Path,
    token_monitor: TokenMonitor | None = None,
    token_report_interval_s: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_instances: list[dict[str, Any]] = []
    all_rejects: list[dict[str, Any]] = []
    row_errors = 0

    worker_count = max(1, workers)
    stop_reporter = threading.Event()
    reporter: threading.Thread | None = None
    if token_monitor is not None and token_report_interval_s > 0:
        reporter = threading.Thread(
            target=_token_reporter_loop,
            args=(token_monitor, stop_reporter, token_report_interval_s),
            name="token-reporter",
            daemon=True,
        )
        reporter.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futs = {
                pool.submit(
                    transform_one_row,
                    client,
                    model,
                    split,
                    idx,
                    row,
                    timeout_s=timeout_s,
                    retries=retries,
                    token_monitor=token_monitor,
                ): idx
                for idx, row in enumerate(rows)
            }

            pbar = tqdm(
                concurrent.futures.as_completed(futs),
                total=len(futs),
                desc=f"{split}",
                unit="row",
                file=sys.stdout,
                dynamic_ncols=True,
            )
            kept_n = 0
            rej_n = 0

            for fut in pbar:
                res = fut.result()
                all_instances.extend(res.instances)
                all_rejects.extend(res.rejects)
                if res.row_error:
                    row_errors += 1
                kept_n += len(res.instances)
                rej_n += len(res.rejects)
                extra: dict[str, Any] = {"kept": kept_n, "rejects": rej_n}
                if token_monitor is not None:
                    snap = token_monitor.snapshot()
                    wall = max(time.perf_counter() - token_monitor.t0, 1e-9)
                    extra["tok_per_s"] = round(snap["total_tokens"] / wall, 1)
                pbar.set_postfix(**{str(k): v for k, v in extra.items()})
    finally:
        if reporter is not None:
            stop_reporter.set()
            reporter.join(timeout=5.0)

    tok_snap = token_monitor.snapshot() if token_monitor is not None else None

    deduped, dup_removed = dedupe_by_instance_id(all_instances)

    out_jsonl = out_dir / f"{split}.jsonl"
    atomic_write_jsonl(out_jsonl, deduped)

    summary_dict: dict[str, Any] = {
        "split": split,
        "rows": len(rows),
        "kept_instances": len(deduped),
        "raw_instances_before_dedupe": len(all_instances),
        "duplicates_removed": dup_removed,
        "reject_records": len(all_rejects),
        "row_errors": row_errors,
        "output": str(out_jsonl),
    }
    if tok_snap is not None and token_monitor is not None:
        wall = max(time.perf_counter() - token_monitor.t0, 1e-9)
        summary_dict["token_usage_end_of_split"] = tok_snap
        summary_dict["token_throughput_wall_avg_tok_per_s"] = round(
            tok_snap["total_tokens"] / wall, 4
        )
    return summary_dict, all_rejects


def main() -> int:
    parser = argparse.ArgumentParser(description="EconLogic -> BSQA transform (protocol prompts 1-9)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train.csv, val.csv, test.csv (EconLogicQA). "
            "Default: <repo root>/econ_logic_qa"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--token-report-interval",
        type=float,
        default=1.0,
        help="Seconds between stderr token speed lines (0 disables). Requires API usage fields.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else REPO_ROOT / "econ_logic_qa"

    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")
    if not api_key:
        print("OPENAI_API_KEY missing (.env)", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "workers": args.workers,
        "timeout_s": args.timeout,
        "retries": args.retries,
        "token_report_interval_s": args.token_report_interval,
        "splits": {},
    }

    token_monitor = TokenMonitor()

    combined_rejects: list[dict[str, Any]] = []
    rej_path = out_dir / "rejected.jsonl"
    if rej_path.exists():
        rej_path.unlink()

    for sp in splits:
        csv_path = input_dir / f"{sp}.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path}", file=sys.stderr)
            return 3
        rows = read_csv_rows(csv_path)
        if args.limit is not None:
            rows = rows[: args.limit]

        split_summary, split_rejects = process_split_parallel(
            client,
            model,
            sp,
            rows,
            workers=args.workers,
            timeout_s=args.timeout,
            retries=args.retries,
            out_dir=out_dir,
            token_monitor=token_monitor,
            token_report_interval_s=max(0.0, args.token_report_interval),
        )
        summary["splits"][sp] = split_summary
        combined_rejects.extend(split_rejects)

    atomic_write_jsonl(rej_path, combined_rejects)
    summary["total_reject_records"] = len(combined_rejects)
    summary["rejected_output"] = str(rej_path)
    summary["token_usage_final"] = token_monitor.snapshot()
    wall = max(time.perf_counter() - token_monitor.t0, 1e-9)
    summary["token_throughput_wall_avg_tok_per_s"] = round(
        summary["token_usage_final"]["total_tokens"] / wall, 4
    )

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
