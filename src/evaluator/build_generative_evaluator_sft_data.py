#!/usr/bin/env python3
"""Build CausalLM SFT data for Method 1 generative self-evaluation.

The output trains one generator to solve, critique, and judge. Verifier labels
are used only to construct targets; prompts contain only public task/interface
text and visible code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_DESCRIPTIONS = {
    "not_a_failure": "the submitted code appears correct for the public task contract",
    "syntax_or_parse_error": "the code is not valid, complete, parseable Python",
    "interface_contract_error": "the required public function or class interface is missing or incompatible",
    "runtime_exception_or_timeout": "the code is likely to raise an exception or fail to terminate reliably",
    "truncation_or_overgeneration": "the completion is incomplete, truncated, or includes extra invalid material",
    "output_format_or_type_error": "the returned value shape or output format does not match the contract",
    "numeric_formula_or_counting_error": "the implementation uses an incorrect formula, count, or numeric update",
    "sequence_or_state_transformation_error": "the implementation transforms sequences or state incorrectly",
    "predicate_condition_or_edge_case_error": "the predicate, branch condition, or edge-case handling is wrong",
    "string_pattern_or_text_error": "the string or pattern logic does not match the task",
    "logic_other_or_unknown": "a likely semantic failure whose fine-grained logical cause is not yet reliable",
    "unclear_other_or_not_failure": "the failure source is unclear or not suitable for a hard attribution label",
}


FAILURE_TYPE_FALLBACK = {
    "syntax_error": "syntax_or_parse_error",
    "runtime_error": "runtime_exception_or_timeout",
    "timeout": "runtime_exception_or_timeout",
    "generation_failure": "unclear_other_or_not_failure",
    "logic_error": "logic_other_or_unknown",
}


PROSE_MARKERS = (
    "The provided solution",
    "Python code:",
    "Here is",
    "This code",
    "Explanation:",
)

TYPING_NAMES = {"List", "Dict", "Tuple", "Set", "Optional", "Deque", "DefaultDict"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bool(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % 2 == 0


def stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def repeat_count(multiplier: float, key: str) -> int:
    if multiplier <= 0:
        return 0
    whole = int(multiplier)
    fraction = multiplier - whole
    if fraction > 0 and stable_unit_interval(key) < fraction:
        whole += 1
    return whole


def parse_rate_assignments(values: list[str], option_name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} entries must use NAME=RATE, got {value!r}")
        name, raw_rate = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"{option_name} entries must include a non-empty name")
        rate = float(raw_rate)
        if rate < 0:
            raise ValueError(f"{option_name} rates must be non-negative, got {value!r}")
        result[name] = rate
    return result


def clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[TRUNCATED]"


def public_interface_text(row: dict[str, Any]) -> str:
    values = row.get("public_interface") or row.get("interface_signatures") or row.get("interface_names") or []
    if not values:
        return "None specified."
    return "\n".join(f"- {value}" for value in values)


def clean_code(value: Any) -> str:
    return str(value or "").strip()


def source_split(row: dict[str, Any]) -> str:
    return str(row.get("split") or row.get("eval_split") or "train")


def task_text(row: dict[str, Any], limit: int) -> str:
    return clip_text(row.get("task") or row.get("prompt"), limit)


def error_label(row: dict[str, Any]) -> tuple[str, str, str]:
    if bool(row.get("passed")):
        return "not_a_failure", "verifier_pass", "high"
    label = row.get("error_attribution_label") or row.get("deterministic_error_label")
    source = row.get("error_attribution_source") or row.get("deterministic_label_source")
    confidence = row.get("error_attribution_confidence") or row.get("deterministic_label_confidence")
    if row.get("finish_reason") == "length":
        label = label or "truncation_or_overgeneration"
        source = source or "finish_reason_length"
        confidence = confidence or "high"
    if not label:
        label = FAILURE_TYPE_FALLBACK.get(str(row.get("failure_type")), "logic_other_or_unknown")
        source = f"failure_type_fallback:{row.get('failure_type')}"
        confidence = "medium" if label != "logic_other_or_unknown" else "low"
    if label not in LABEL_DESCRIPTIONS:
        label = "logic_other_or_unknown"
    return str(label), str(source or "unknown"), str(confidence or "medium")


def parse_python(code: str) -> tuple[ast.Module | None, str | None]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(code), None
    except SyntaxError as exc:
        message = exc.msg
        if exc.lineno is not None:
            message = f"{message} at line {exc.lineno}"
        return None, message


def code_feature_summary(code: str) -> dict[str, Any]:
    lines = [line for line in code.splitlines() if line.strip()]
    tree, syntax_error = parse_python(code)
    features: dict[str, Any] = {
        "line_count": len(lines),
        "syntax_error": syntax_error,
        "has_markdown_fence": "```" in code,
        "has_prose_marker": any(marker in code for marker in PROSE_MARKERS),
        "has_comment": any("#" in line for line in lines),
        "many_lines": len(lines) >= 50,
        "duplicate_defs": [],
        "top_level_call_count": 0,
        "typing_names_without_import": [],
    }
    if tree is None:
        return features

    definitions = Counter(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    features["duplicate_defs"] = sorted(name for name, count in definitions.items() if count > 1)
    features["top_level_call_count"] = sum(
        1
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    )
    imported_from_typing: set[str] = set()
    typing_module_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "typing" for alias in node.names):
                typing_module_imported = True
        elif isinstance(node, ast.ImportFrom) and node.module == "typing":
            imported_from_typing.update(alias.name for alias in node.names)
    used_typing_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in TYPING_NAMES
    }
    if not typing_module_imported:
        features["typing_names_without_import"] = sorted(used_typing_names - imported_from_typing)
    return features


def visible_feature_tags(features: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if features.get("syntax_error"):
        tags.append("syntax_error")
    for key in [
        "has_markdown_fence",
        "has_prose_marker",
        "has_comment",
        "many_lines",
    ]:
        if features.get(key):
            tags.append(key)
    if features.get("duplicate_defs"):
        tags.append("duplicate_defs")
    if features.get("top_level_call_count"):
        tags.append("top_level_calls")
    if features.get("typing_names_without_import"):
        tags.append("typing_names_without_import")
    return tags


def visible_evidence_lines(row: dict[str, Any], label: str) -> list[str]:
    code = clean_code(row.get("extracted_code") or row.get("generated_code"))
    features = code_feature_summary(code)
    lines: list[str] = []
    if features.get("syntax_error"):
        lines.append(f"Visible evidence: Python parsing fails ({features['syntax_error']}).")
    if features.get("has_prose_marker"):
        lines.append("Visible evidence: non-code prose appears inside the submitted code block.")
    if features.get("has_markdown_fence"):
        lines.append("Visible evidence: Markdown fences remain in the submitted code.")
    if features.get("duplicate_defs"):
        names = ", ".join(features["duplicate_defs"][:3])
        lines.append(f"Visible evidence: duplicate top-level definitions are present ({names}).")
    if features.get("typing_names_without_import"):
        names = ", ".join(features["typing_names_without_import"][:4])
        lines.append(f"Visible evidence: typing annotation names are used without a visible typing import ({names}).")
    if features.get("many_lines") and label in {"truncation_or_overgeneration", "syntax_or_parse_error"}:
        lines.append(f"Visible evidence: the completion is unusually long ({features['line_count']} non-empty lines), consistent with overgeneration or truncation.")
    if features.get("top_level_call_count"):
        if label == "not_a_failure":
            lines.append(
                "Calibration: top-level example calls are present, but harmless calls are not an automatic failure when the required implementation is still valid."
            )
        else:
            lines.append(
                "Visible evidence: top-level calls may execute during evaluation and can violate the required interface or output contract."
            )
    if features.get("has_comment") and label == "not_a_failure":
        lines.append("Calibration: comments are allowed when they do not make the Python invalid or change the required behavior.")
    if not lines:
        if label == "not_a_failure":
            lines.append("Visible evidence: the code is parseable and presents the required implementation without an obvious malformed fragment.")
        elif label == "logic_other_or_unknown":
            lines.append("Weak label note: no reliable visible fine-grained error evidence is available; this verifier-derived semantic failure should not dominate training.")
        else:
            lines.append(f"Training label evidence: the verifier-derived primary error is {label}.")
    return lines


def has_hard_positive_features(row: dict[str, Any]) -> bool:
    features = code_feature_summary(clean_code(row.get("extracted_code") or row.get("generated_code")))
    return bool(
        features.get("top_level_call_count")
        or features.get("has_comment")
        or features.get("many_lines")
        or features.get("duplicate_defs")
    )


def has_visible_negative_features(row: dict[str, Any], label: str, include_logic_other: bool) -> bool:
    if label == "logic_other_or_unknown" and not include_logic_other:
        return False
    features = code_feature_summary(clean_code(row.get("extracted_code") or row.get("generated_code")))
    if features.get("syntax_error") or features.get("has_prose_marker") or features.get("has_markdown_fence"):
        return True
    if features.get("duplicate_defs") and label in {"syntax_or_parse_error", "truncation_or_overgeneration", "interface_contract_error"}:
        return True
    if features.get("typing_names_without_import") and label == "runtime_exception_or_timeout":
        return True
    if features.get("many_lines") and label in {"syntax_or_parse_error", "truncation_or_overgeneration"}:
        return True
    return False


def rubric_block(calibrated_extra_content_policy: bool = False) -> str:
    if calibrated_extra_content_policy:
        extra_content_rule = (
            "- Reject malformed prose, Markdown fences, duplicated fragments, or test code that changes required execution; "
            "do not reject harmless comments or unused example calls by themselves."
        )
    else:
        extra_content_rule = "- Reject extra prose, tests, duplicate code fragments, or malformed output."
    return "\n".join(
        [
            "Rubric:",
            "- Check that the submitted Python is syntactically valid and complete.",
            "- Check that the required public interface is defined and callable.",
            "- Check runtime safety: imports, names, types, indexing, and termination.",
            "- Check task semantics against the public specification and examples.",
            extra_content_rule,
        ]
    )


def single_judge_prompt(
    row: dict[str, Any],
    task_chars: int,
    code_chars: int,
    answer_first: bool,
    calibrated_extra_content_policy: bool = False,
) -> str:
    code = clip_text(clean_code(row.get("extracted_code") or row.get("generated_code")), code_chars)
    if answer_first:
        return_fields = (
            "Return the following text fields in this exact order:\n"
            "Verdict: PASS or FAIL\n"
            "Primary error: <label>\n"
            "Confidence: high|medium|low\n"
            "Rubric:\n- ...\n"
            "Analysis:\n..."
        )
    else:
        return_fields = (
            "Return the following text fields:\n"
            "Rubric:\n- ...\nAnalysis:\n...\nVerdict: PASS or FAIL\nPrimary error: <label>\nConfidence: high|medium|low"
        )
    calibration = ""
    if calibrated_extra_content_policy:
        calibration = (
            "Calibration: comments and harmless example calls are not automatic failures. "
            "Reject extra material when it makes the Python invalid, changes required I/O behavior, "
            "duplicates malformed fragments, or violates the public interface.\n\n"
        )
    return (
        "You are the same Python model acting as a rubric-guided self-evaluator.\n"
        "Decide whether the submitted code should pass the task. Use only the public task, "
        "public interface, and submitted code. Do not assume hidden test results.\n\n"
        f"{calibration}"
        f"{return_fields}\n\n"
        "Public task:\n"
        f"{task_text(row, task_chars)}\n\n"
        "Public interface:\n"
        f"{public_interface_text(row)}\n\n"
        "Submitted code:\n"
        "```python\n"
        f"{code}\n"
        "```"
    )


def single_judge_completion(
    row: dict[str, Any],
    answer_first: bool,
    evidence_aware: bool = False,
    calibrated_extra_content_policy: bool = False,
) -> str:
    label, source, confidence = error_label(row)
    if label == "not_a_failure":
        verdict = "PASS"
    else:
        verdict = "FAIL"
    if evidence_aware:
        evidence = " ".join(visible_evidence_lines(row, label))
        if label == "not_a_failure":
            analysis = f"The submitted code should be accepted. {evidence}"
        else:
            description = LABEL_DESCRIPTIONS[label]
            analysis = f"The submitted code should be rejected because it matches {description}. {evidence}"
    elif label == "not_a_failure":
        analysis = "The submitted code defines the expected public entry point and presents a complete implementation."
    else:
        description = LABEL_DESCRIPTIONS[label]
        analysis = f"The submitted code should be rejected because it matches {description}."
    if answer_first:
        return (
            f"Verdict: {verdict}\n"
            f"Primary error: {label}\n"
            f"Confidence: {confidence}\n"
            f"{rubric_block(calibrated_extra_content_policy)}\n"
            "Analysis:\n"
            f"{analysis}"
        )
    return (
        f"{rubric_block(calibrated_extra_content_policy)}\n"
        "Analysis:\n"
        f"{analysis}\n"
        f"Verdict: {verdict}\n"
        f"Primary error: {label}\n"
        f"Confidence: {confidence}"
    )


def pair_judge_prompt(
    pair: dict[str, Any],
    task_chars: int,
    code_chars: int,
    answer_first: bool,
) -> tuple[str, str]:
    chosen_is_a = stable_bool(str(pair.get("pair_id") or pair.get("id")))
    chosen = clip_text(clean_code(pair.get("chosen")), code_chars)
    rejected = clip_text(clean_code(pair.get("rejected")), code_chars)
    code_a = chosen if chosen_is_a else rejected
    code_b = rejected if chosen_is_a else chosen
    winner = "A" if chosen_is_a else "B"
    if answer_first:
        return_instruction = "Return `Winner: A` or `Winner: B` as the first line, then a rubric comparison."
    else:
        return_instruction = "Return a rubric comparison and a final line `Winner: A` or `Winner: B`."
    prompt = (
        "You are the same Python model acting as a pairwise rubric judge.\n"
        "Compare two candidate solutions for the public task. Use only the task and the visible code.\n"
        f"{return_instruction}\n\n"
        "Public task:\n"
        f"{clip_text(pair.get('prompt'), task_chars)}\n\n"
        "Candidate A:\n"
        "```python\n"
        f"{code_a}\n"
        "```\n\n"
        "Candidate B:\n"
        "```python\n"
        f"{code_b}\n"
        "```"
    )
    return prompt, winner


def pair_judge_completion(pair: dict[str, Any], winner: str, answer_first: bool) -> str:
    loser = "B" if winner == "A" else "A"
    original_failure = str(pair.get("original_failure_type") or "failed_verifier")
    comparison = (
        f"Candidate {winner} is preferred because it better satisfies the public task and code-quality rubric. "
        f"Candidate {loser} shows the risk pattern associated with {original_failure} in this training pair."
    )
    if answer_first:
        return (
            f"Winner: {winner}\n"
            f"{rubric_block()}\n"
            "Comparison:\n"
            f"{comparison}"
        )
    return (
        f"{rubric_block()}\n"
        "Comparison:\n"
        f"{comparison}\n"
        f"Winner: {winner}"
    )


def solve_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or "").strip()


def make_record(
    *,
    row_id: str,
    split: str,
    task_type: str,
    prompt: str,
    completion: str,
    source: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row_id,
        "split": split,
        "task_type": task_type,
        "prompt": prompt.strip(),
        "completion": completion.strip(),
        "source": source,
        "metadata": metadata,
    }


def task_type_allowed(record: dict[str, Any], args: argparse.Namespace) -> bool:
    split = str(record.get("split") or "")
    task_type = str(record.get("task_type") or "")
    if split == "train" and args.train_task_types and task_type not in set(args.train_task_types):
        return False
    if split != "train" and args.heldout_task_types and task_type not in set(args.heldout_task_types):
        return False
    return True


def train_repeat_multiplier(record: dict[str, Any], args: argparse.Namespace) -> float:
    if str(record.get("split") or "") != "train":
        return 1.0
    task_type = str(record.get("task_type") or "")
    multiplier = args.train_task_repeat_rates.get(task_type, 1.0)
    if task_type == "judge_single":
        metadata = record.get("metadata") or {}
        if bool(metadata.get("passed")):
            multiplier *= args.train_pass_repeat
        else:
            multiplier *= args.train_fail_repeat
        primary_error = str(metadata.get("primary_error") or "")
        multiplier *= args.train_primary_error_repeat_rates.get(primary_error, 1.0)
    return multiplier


def expand_record(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not task_type_allowed(record, args):
        return []
    count = repeat_count(train_repeat_multiplier(record, args), f"{args.sampling_seed}:{record['id']}")
    if count <= 0:
        return []
    if count == 1:
        return [record]
    expanded: list[dict[str, Any]] = []
    for repeat_index in range(count):
        new_record = dict(record)
        metadata = dict(record.get("metadata") or {})
        metadata["repeat_index"] = repeat_index
        metadata["repeat_count"] = count
        new_record["metadata"] = metadata
        if repeat_index > 0:
            new_record["id"] = f"{record['id']}__repeat{repeat_index}"
        expanded.append(new_record)
    return expanded


def add_record(records: list[dict[str, Any]], record: dict[str, Any], args: argparse.Namespace) -> None:
    records.extend(expand_record(record, args))


def hard_case_record_for_row(
    row: dict[str, Any],
    rid: str,
    split: str,
    label: str,
    label_source: str,
    confidence: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.add_hard_case_records or split != "train":
        return None
    passed = bool(row.get("passed"))
    if passed:
        if not has_hard_positive_features(row):
            return None
        task_type = "judge_single_hard_positive"
        source = "verifier_passing_visible_edge_case_positive"
    elif has_visible_negative_features(row, label, args.hard_logic_other):
        task_type = "judge_single_hard_negative"
        source = "visible_evidence_hard_negative"
    else:
        return None

    features = code_feature_summary(clean_code(row.get("extracted_code") or row.get("generated_code")))
    return make_record(
        row_id=f"{rid}__{task_type}",
        split=split,
        task_type=task_type,
        prompt=single_judge_prompt(
            row,
            args.task_chars,
            args.code_chars,
            args.answer_first_judge,
            args.calibrated_extra_content_policy,
        ),
        completion=single_judge_completion(
            row,
            args.answer_first_judge,
            evidence_aware=True,
            calibrated_extra_content_policy=args.calibrated_extra_content_policy,
        ),
        source=source,
        metadata={
            "response_id": rid,
            "problem_id": row.get("id"),
            "passed": passed,
            "primary_error": label,
            "label_source": label_source,
            "confidence": confidence,
            "visible_feature_tags": visible_feature_tags(features),
            "repair_role": "hard_positive" if passed else "hard_negative",
        },
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    evaluator_rows = read_jsonl(args.evaluator_rows)
    repair_pairs = read_jsonl(args.repair_pairs)
    records: list[dict[str, Any]] = []

    for row in evaluator_rows:
        split = source_split(row)
        label, label_source, confidence = error_label(row)
        if args.high_confidence_failures_only and not row.get("passed") and confidence == "low":
            continue
        rid = str(row.get("response_id") or f"{row.get('id')}__sample{row.get('sample_id', 0)}")
        add_record(
            records,
            make_record(
                row_id=f"{rid}__judge_single",
                split=split,
                task_type="judge_single",
                prompt=single_judge_prompt(
                    row,
                    args.task_chars,
                    args.code_chars,
                    args.answer_first_judge,
                    args.calibrated_extra_content_policy,
                ),
                completion=single_judge_completion(
                    row,
                    args.answer_first_judge,
                    args.evidence_aware_judge,
                    args.calibrated_extra_content_policy,
                ),
                source="verifier_gated_single_judge",
                metadata={
                    "response_id": rid,
                    "problem_id": row.get("id"),
                    "passed": bool(row.get("passed")),
                    "primary_error": label,
                    "label_source": label_source,
                    "confidence": confidence,
                },
            ),
            args,
        )
        hard_record = hard_case_record_for_row(row, rid, split, label, label_source, confidence, args)
        if hard_record:
            add_record(records, hard_record, args)
        if bool(row.get("passed")):
            code = clean_code(row.get("extracted_code") or row.get("generated_code"))
            if code:
                add_record(
                    records,
                    make_record(
                        row_id=f"{rid}__solve_verified_pass",
                        split=split,
                        task_type="solve_verified_pass",
                        prompt=solve_prompt(row),
                        completion=code,
                        source="verifier_passing_base_solution",
                        metadata={"response_id": rid, "problem_id": row.get("id")},
                    ),
                    args,
                )

    for pair in repair_pairs:
        if str(pair.get("split") or pair.get("source_split") or "train") != "train":
            continue
        pair_id = str(pair.get("pair_id") or f"{pair.get('id')}__pair")
        chosen = clean_code(pair.get("chosen"))
        if chosen:
            add_record(
                records,
                make_record(
                    row_id=f"{pair_id}__solve_repair_chosen",
                    split="train",
                    task_type="solve_repair_chosen",
                    prompt=str(pair.get("prompt") or "").strip(),
                    completion=chosen,
                    source="verifier_passing_repair_solution",
                    metadata={
                        "pair_id": pair_id,
                        "problem_id": pair.get("id"),
                        "chosen_response_id": pair.get("chosen_response_id"),
                        "rejected_response_id": pair.get("rejected_response_id"),
                    },
                ),
                args,
            )
        prompt, winner = pair_judge_prompt(pair, args.task_chars, args.code_chars, args.answer_first_judge)
        add_record(
            records,
            make_record(
                row_id=f"{pair_id}__judge_pair",
                split="train",
                task_type="judge_pair",
                prompt=prompt,
                completion=pair_judge_completion(pair, winner, args.answer_first_judge),
                source="verifier_gated_pairwise_judge",
                metadata={
                    "pair_id": pair_id,
                    "problem_id": pair.get("id"),
                    "winner": winner,
                    "preference_source": pair.get("preference_source"),
                },
            ),
            args,
        )

    return records


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "total_rows": len(records),
        "split_counts": dict(Counter(str(row.get("split")) for row in records)),
        "task_type_counts": dict(Counter(str(row.get("task_type")) for row in records)),
        "split_task_type_counts": {
            f"{split}:{task_type}": count
            for (split, task_type), count in Counter((str(row.get("split")), str(row.get("task_type"))) for row in records).items()
        },
        "source_counts": dict(Counter(str(row.get("source")) for row in records)),
        "primary_error_counts": dict(Counter(str((row.get("metadata") or {}).get("primary_error")) for row in records if row.get("task_type", "").startswith("judge_single"))),
        "repair_role_counts": dict(Counter(str((row.get("metadata") or {}).get("repair_role")) for row in records if (row.get("metadata") or {}).get("repair_role"))),
        "visible_feature_tag_counts": dict(
            Counter(
                tag
                for row in records
                for tag in ((row.get("metadata") or {}).get("visible_feature_tags") or [])
            )
        ),
        "inputs": {
            "evaluator_rows": str(args.evaluator_rows),
            "evaluator_rows_sha256": sha256_file(args.evaluator_rows),
            "repair_pairs": str(args.repair_pairs),
            "repair_pairs_sha256": sha256_file(args.repair_pairs),
        },
        "policy": {
            "model_form": "CausalLM SFT, not sequence classification",
            "prompt_inputs": "public task/interface/code only",
            "target_source": "external verifier labels and verifier-confirmed repair pairs",
            "purpose": "teach one generator to solve, critique, and judge before self-generated reward is trusted",
            "judge_output_format": "answer_first" if args.answer_first_judge else "rubric_first",
            "train_task_types": args.train_task_types,
            "heldout_task_types": args.heldout_task_types,
            "train_pass_repeat": args.train_pass_repeat,
            "train_fail_repeat": args.train_fail_repeat,
            "train_task_repeat_rates": args.train_task_repeat_rates,
            "train_primary_error_repeat_rates": args.train_primary_error_repeat_rates,
            "evidence_aware_judge": args.evidence_aware_judge,
            "calibrated_extra_content_policy": args.calibrated_extra_content_policy,
            "add_hard_case_records": args.add_hard_case_records,
            "hard_logic_other": args.hard_logic_other,
            "sampling_seed": args.sampling_seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 1 generative self-evaluator SFT data.")
    parser.add_argument("--evaluator-rows", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"))
    parser.add_argument("--repair-pairs", type=Path, default=Path("data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/sft/apps_simple_method1_generative_self_evaluator_v1.jsonl"))
    parser.add_argument("--split-output-dir", type=Path, default=Path("data/sft/apps_simple_method1_generative_self_evaluator_v1"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/sft/apps_simple_method1_generative_self_evaluator_v1_summary.json"))
    parser.add_argument("--task-chars", type=int, default=4500)
    parser.add_argument("--code-chars", type=int, default=4500)
    parser.add_argument("--high-confidence-failures-only", action="store_true")
    parser.add_argument(
        "--answer-first-judge",
        action="store_true",
        help="Place Verdict/Winner before rubric text for generative judge targets.",
    )
    parser.add_argument(
        "--evidence-aware-judge",
        action="store_true",
        help="Use visible code features in judge analysis targets instead of label-only templates.",
    )
    parser.add_argument(
        "--calibrated-extra-content-policy",
        action="store_true",
        help="Calibrate rubric/prompt so harmless comments and example calls are not automatic failures.",
    )
    parser.add_argument(
        "--add-hard-case-records",
        action="store_true",
        help="Add train-only hard-positive and hard-negative judge records based on visible code features.",
    )
    parser.add_argument(
        "--hard-logic-other",
        action="store_true",
        help="Allow logic_other_or_unknown examples with visible feature evidence to become hard negatives.",
    )
    parser.add_argument(
        "--train-task-types",
        nargs="+",
        default=None,
        help="Optional task_type allowlist for train split only.",
    )
    parser.add_argument(
        "--heldout-task-types",
        nargs="+",
        default=None,
        help="Optional task_type allowlist for non-train splits only.",
    )
    parser.add_argument("--train-pass-repeat", type=float, default=1.0)
    parser.add_argument("--train-fail-repeat", type=float, default=1.0)
    parser.add_argument(
        "--train-task-repeat",
        action="append",
        default=[],
        help="Train-only task_type repeat/downsample rate, e.g. judge_pair=0.5.",
    )
    parser.add_argument(
        "--train-primary-error-repeat",
        action="append",
        default=[],
        help="Train-only judge_single primary_error repeat/downsample rate, e.g. logic_other_or_unknown=0.5.",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_pass_repeat < 0 or args.train_fail_repeat < 0:
        raise ValueError("train repeat rates must be non-negative")
    args.train_task_repeat_rates = parse_rate_assignments(args.train_task_repeat, "--train-task-repeat")
    args.train_primary_error_repeat_rates = parse_rate_assignments(
        args.train_primary_error_repeat,
        "--train-primary-error-repeat",
    )

    records = build_rows(args)
    if not records:
        raise SystemExit("no SFT records were built")
    records = sorted(records, key=lambda row: (str(row.get("split")), str(row.get("task_type")), str(row.get("id"))))
    write_jsonl(args.output, records)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_split.setdefault(str(record.get("split")), []).append(record)
    for split, split_rows in by_split.items():
        write_jsonl(args.split_output_dir / f"{split}.jsonl", split_rows)
    write_json(args.summary_output, summarize(records, args))
    print(json.dumps(summarize(records, args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
