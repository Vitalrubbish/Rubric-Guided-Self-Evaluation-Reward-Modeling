#!/usr/bin/env python3
"""Consolidate raw discovered clusters into a rubric-ready taxonomy.

This stage is intentionally automatic:
  1. Summarize raw cluster evidence from the Phase 1 taxonomy YAML.
  2. Ask an LLM to merge the raw clusters into 6-8 higher-level categories.
  3. Deterministically audit and repair cluster coverage.
  4. Write a consolidated taxonomy, audit JSON, cluster mapping, and optional
     response-level category assignments.

The LLM performs semantic normalization; the script enforces coverage and
schema constraints. No human approval is required for the output artifact.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from vllm import LLM, SamplingParams


PRIVATE_TOKENS = {
    "test_list",
    "test_setup_code",
    "private_diagnostics",
    "assert ",
    "assert(",
}

BANNED_CATEGORY_IDS = {
    "logic_errors",
    "function_errors",
    "type_errors",
    "runtime_errors",
    "syntax_errors",
    "mixed_or_low_confidence",
    "other_errors",
    "general_errors",
}

BANNED_CATEGORY_NAMES = {
    "logic errors",
    "function errors",
    "type errors",
    "runtime errors",
    "syntax errors",
    "mixed or low-confidence errors",
    "other errors",
    "general errors",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return slug[:64] or fallback


def compact_dict(counter: dict | None, limit: int = 6) -> dict:
    if not counter:
        return {}
    return dict(Counter(counter).most_common(limit))


def cluster_brief(cluster: dict, max_examples: int = 4) -> dict:
    examples = []
    for example in (cluster.get("examples") or [])[:max_examples]:
        examples.append(
            {
                "response_id": example.get("response_id"),
                "failure_type": example.get("failure_type"),
                "summary": example.get("summary"),
            }
        )
    return {
        "cluster_id": cluster.get("cluster_id"),
        "raw_name": cluster.get("name"),
        "response_count": cluster.get("response_count", cluster.get("size")),
        "unique_task_count": cluster.get("unique_task_count"),
        "task_ratio": cluster.get("task_ratio"),
        "failure_types": compact_dict(cluster.get("failure_types")),
        "error_patterns": compact_dict(cluster.get("error_patterns") or cluster.get("rule_patterns")),
        "top_terms": (cluster.get("top_terms") or [])[:8],
        "example_summaries": examples,
    }


def build_consolidation_prompt(taxonomy: dict, min_categories: int, max_categories: int) -> str:
    clusters = [cluster_brief(cluster) for cluster in taxonomy.get("clusters") or taxonomy.get("patterns") or []]
    cluster_ids = [cluster["cluster_id"] for cluster in clusters]
    evidence = json.dumps(clusters, ensure_ascii=False, indent=2)
    return f"""You are consolidating an automatically discovered coding-error taxonomy into rubric dimensions.

Input: raw clusters produced by LLM root-cause summaries plus algorithmic clustering.
Task: merge these raw clusters into {min_categories}-{max_categories} higher-level, rubric-operational error categories.

Hard constraints:
- Use every raw cluster exactly once.
- Do not invent cluster ids. Valid raw cluster ids are: {cluster_ids}
- Do not cite hidden tests, assert statements, exact expected values, or private verifier details.
- Do not make task-specific categories; categories must be general enough to score new code.
- Every category must be directly usable as a rubric dimension with 1-5 scoring criteria.
- Avoid broad labels such as "logic errors", "function errors", "type errors", "runtime errors", or "syntax errors".
- Prefer operational labels such as "algorithmic wrong value", "output type or container shape", "edge-case handling", "interface/name/signature mismatch", "runtime API/type misuse", "syntax parseability/truncation", and "string/regex pattern logic".
- If a raw cluster is mixed, assign it to the closest category and mention low confidence.

Return ONLY valid JSON with this schema:
{{
  "taxonomy_name": "mbpp_consolidated_error_taxonomy_v1",
  "categories": [
    {{
      "id": "snake_case_id",
      "name": "Short Human Readable Name",
      "description": "One or two sentences.",
      "linked_clusters": ["cluster_00"],
      "common_failure_signals": ["short signal"],
      "rubric_hint": "How a rubric judge should evaluate this category.",
      "score_focus": "What a 1-5 rubric scale should distinguish.",
      "confidence": "high"
    }}
  ],
  "notes": "optional short note"
}}

Raw cluster evidence:
{evidence}
"""


def parse_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:index + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def dominant_pattern(cluster: dict) -> str:
    patterns = Counter(cluster.get("error_patterns") or cluster.get("rule_patterns") or {})
    if patterns:
        return patterns.most_common(1)[0][0]
    failures = Counter(cluster.get("failure_types") or {})
    return failures.most_common(1)[0][0] if failures else "unknown"


def fallback_category_for(cluster: dict) -> tuple[str, str]:
    pattern = dominant_pattern(cluster)
    name = str(cluster.get("name", "")).lower()
    terms = " ".join(cluster.get("top_terms") or []).lower()
    text = " ".join([pattern, name, terms])

    if "syntax" in text:
        if "duplicate" in text or "markdown" in text or "format" in text:
            return "format_or_duplicate_code_output", "Format or Duplicate Code Output"
        return "syntax_parseability_or_truncation", "Syntax Parseability or Truncation"
    if "missing_required_interface" in text or "interface" in text or "nameerror" in text:
        return "interface_name_signature_mismatch", "Interface, Name, or Signature Mismatch"
    if "runtime" in text or "typeerror" in text:
        return "runtime_api_type_misuse", "Runtime API or Type Misuse"
    if "timeout" in text:
        return "termination_or_complexity_control", "Termination or Complexity Control"
    if "regex" in text or "pattern" in text or "string" in text:
        return "string_regex_pattern_logic", "String or Regex Pattern Logic"
    if any(token in text for token in ("edge", "empty", "duplicate", "boundary", "does handle", "handle cases")):
        return "edge_case_boundary_handling", "Edge Case or Boundary Handling"
    if any(token in text for token in ("number", "numeric", "formula", "ceil", "floor", "triangle", "integer")):
        return "numeric_formula_correctness", "Numeric or Formula Correctness"
    if "wrong_output_type" in text or "wrong_output_length" in text or "tuple" in text or "list" in text:
        return "output_type_container_shape", "Output Type or Container Shape"
    return "algorithmic_wrong_value", "Algorithmic Wrong Value"


def deterministic_fallback(raw_clusters: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    for cluster in raw_clusters:
        category_id, name = fallback_category_for(cluster)
        grouped.setdefault(
            category_id,
            {
                "id": category_id,
                "name": name,
                "description": "Automatically grouped from dominant failure signals when LLM consolidation was unavailable or invalid.",
                "linked_clusters": [],
                "common_failure_signals": [],
                "rubric_hint": "Check whether the generated code avoids this class of failure without relying on hidden tests.",
                "score_focus": "Distinguish severe, partial, and minor instances of this failure mode.",
                "confidence": "medium",
            },
        )
        grouped[category_id]["linked_clusters"].append(cluster["cluster_id"])
        grouped[category_id]["common_failure_signals"].append(dominant_pattern(cluster))
    return {
        "taxonomy_name": "mbpp_consolidated_error_taxonomy_v1",
        "categories": list(grouped.values()),
        "notes": "deterministic fallback used",
    }


def normalize_categories(candidate: dict, raw_clusters: list[dict]) -> tuple[list[dict], dict]:
    raw_by_id = {cluster["cluster_id"]: cluster for cluster in raw_clusters}
    valid_ids = set(raw_by_id)
    seen: set[str] = set()
    duplicate_clusters = []
    unknown_clusters = []
    categories = []
    used_category_ids: set[str] = set()

    for index, category in enumerate(candidate.get("categories") or []):
        if not isinstance(category, dict):
            continue
        category_id = slugify(str(category.get("id") or category.get("name") or ""), f"category_{index:02d}")
        if category_id in used_category_ids:
            suffix = 2
            base = category_id
            while f"{base}_{suffix}" in used_category_ids:
                suffix += 1
            category_id = f"{base}_{suffix}"
        used_category_ids.add(category_id)

        linked = []
        for cid in category.get("linked_clusters") or []:
            cid = str(cid)
            if cid not in valid_ids:
                unknown_clusters.append(cid)
                continue
            if cid in seen:
                duplicate_clusters.append(cid)
                continue
            seen.add(cid)
            linked.append(cid)
        if not linked:
            continue
        categories.append(
            {
                "id": category_id,
                "name": str(category.get("name") or category_id.replace("_", " ").title()),
                "description": str(category.get("description") or "").strip(),
                "linked_clusters": linked,
                "common_failure_signals": [str(item) for item in (category.get("common_failure_signals") or [])][:8],
                "rubric_hint": str(category.get("rubric_hint") or "").strip(),
                "score_focus": str(category.get("score_focus") or "").strip(),
                "confidence": str(category.get("confidence") or "medium").lower(),
            }
        )

    missing = sorted(valid_ids - seen)
    if missing:
        for cid in missing:
            cluster = raw_by_id[cid]
            fallback_id, fallback_name = fallback_category_for(cluster)
            target = next((cat for cat in categories if cat["id"] == fallback_id), None)
            if target is None:
                target = {
                    "id": fallback_id,
                    "name": fallback_name,
                    "description": "Automatically created for raw clusters omitted by the LLM consolidation, using dominant raw-cluster failure signals.",
                    "linked_clusters": [],
                    "common_failure_signals": [],
                    "rubric_hint": "Check this failure mode without relying on hidden tests.",
                    "score_focus": "Distinguish severe, partial, and minor instances of this failure mode.",
                    "confidence": "medium",
                }
                categories.append(target)
            target["linked_clusters"].append(cid)
            target["common_failure_signals"].append(dominant_pattern(cluster))
            seen.add(cid)

    audit = {
        "unknown_clusters_removed": sorted(set(unknown_clusters)),
        "duplicate_clusters_removed": sorted(set(duplicate_clusters)),
        "missing_clusters_repaired": missing,
    }
    return categories, audit


def is_broad_category(category: dict) -> bool:
    category_id = str(category.get("id", "")).lower()
    name = str(category.get("name", "")).strip().lower()
    return category_id in BANNED_CATEGORY_IDS or name in BANNED_CATEGORY_NAMES


def operationalize_broad_categories(categories: list[dict], raw_clusters: list[dict]) -> tuple[list[dict], list[dict]]:
    """Replace broad LLM categories with rubric-operational fallback categories."""
    raw_by_id = {cluster["cluster_id"]: cluster for cluster in raw_clusters}
    result: list[dict] = []
    by_id: dict[str, dict] = {}
    replacements = []

    def add_to_result(category: dict) -> None:
        existing = by_id.get(category["id"])
        if existing is None:
            by_id[category["id"]] = category
            result.append(category)
            return
        existing["linked_clusters"].extend(category.get("linked_clusters") or [])
        existing["common_failure_signals"].extend(category.get("common_failure_signals") or [])

    for category in categories:
        if not is_broad_category(category):
            add_to_result(category)
            continue

        grouped: dict[str, dict] = {}
        for cid in category.get("linked_clusters") or []:
            cluster = raw_by_id.get(cid)
            if not cluster:
                continue
            fallback_id, fallback_name = fallback_category_for(cluster)
            grouped.setdefault(
                fallback_id,
                {
                    "id": fallback_id,
                    "name": fallback_name,
                    "description": (
                        f"Automatically operationalized from broad LLM category '{category.get('name')}' "
                        "using raw-cluster failure signals."
                    ),
                    "linked_clusters": [],
                    "common_failure_signals": [],
                    "rubric_hint": category.get("rubric_hint") or "Check this failure mode without relying on hidden tests.",
                    "score_focus": category.get("score_focus") or "Distinguish severe, partial, and minor instances of this failure mode.",
                    "confidence": "medium",
                },
            )
            grouped[fallback_id]["linked_clusters"].append(cid)
            grouped[fallback_id]["common_failure_signals"].append(dominant_pattern(cluster))
        for replacement in grouped.values():
            replacement["common_failure_signals"] = sorted(set(replacement["common_failure_signals"]))[:8]
            add_to_result(replacement)
        replacements.append(
            {
                "source_category": category.get("id"),
                "replacement_categories": sorted(grouped),
            }
        )

    for category in result:
        category["linked_clusters"] = sorted(set(category.get("linked_clusters") or []))
        category["common_failure_signals"] = sorted(set(category.get("common_failure_signals") or []))[:8]
    return [category for category in result if category["linked_clusters"]], replacements


def extract_specific_categories(
    categories: list[dict], raw_clusters: list[dict], max_categories: int
) -> tuple[list[dict], list[dict]]:
    """Move distinct fallback groups out of overly broad LLM categories."""
    raw_by_id = {cluster["cluster_id"]: cluster for cluster in raw_clusters}
    broad_ids = set(BANNED_CATEGORY_IDS)
    priority = [
        "interface_name_signature_mismatch",
        "runtime_api_type_misuse",
        "string_regex_pattern_logic",
        "output_type_container_shape",
        "edge_case_boundary_handling",
        "numeric_formula_correctness",
        "algorithmic_wrong_value",
        "termination_or_complexity_control",
        "syntax_parseability_or_truncation",
        "format_or_duplicate_code_output",
    ]
    extractions = []
    priority_rank = {category_id: index for index, category_id in enumerate(priority)}

    while len(categories) < max_categories:
        candidates = []
        for category_index, category in enumerate(categories):
            if category["id"] not in broad_ids or len(category["linked_clusters"]) <= 1:
                continue
            groups: dict[str, tuple[str, list[str]]] = {}
            for cid in category["linked_clusters"]:
                cluster = raw_by_id.get(cid)
                if not cluster:
                    continue
                fallback_id, fallback_name = fallback_category_for(cluster)
                groups.setdefault(fallback_id, (fallback_name, []))[1].append(cid)
            for fallback_id, (fallback_name, linked) in groups.items():
                if fallback_id == category["id"] or not linked or len(linked) == len(category["linked_clusters"]):
                    continue
                candidates.append(
                    (
                        priority_rank.get(fallback_id, len(priority)),
                        -len(linked),
                        category_index,
                        fallback_id,
                        fallback_name,
                        linked,
                    )
                )
        if not candidates:
            break

        _, _, category_index, fallback_id, fallback_name, linked = sorted(candidates)[0]
        category = categories[category_index]
        linked_set = set(linked)
        category["linked_clusters"] = [cid for cid in category["linked_clusters"] if cid not in linked_set]
        signals = [dominant_pattern(raw_by_id[cid]) for cid in linked if cid in raw_by_id]
        existing = next((cat for cat in categories if cat["id"] == fallback_id), None)
        if existing is None:
            categories.append(
                {
                    "id": fallback_id,
                    "name": fallback_name,
                    "description": (
                        f"Automatically extracted from broad LLM category '{category['name']}' "
                        "using dominant raw-cluster failure signals."
                    ),
                    "linked_clusters": linked,
                    "common_failure_signals": sorted(set(signals))[:8],
                    "rubric_hint": category.get("rubric_hint") or "Check this failure mode without relying on hidden tests.",
                    "score_focus": category.get("score_focus") or "Distinguish severe, partial, and minor instances of this failure mode.",
                    "confidence": "medium",
                }
            )
        else:
            existing["linked_clusters"].extend(linked)
            existing["common_failure_signals"].extend(signals)
        extractions.append(
            {
                "source_category": category["id"],
                "new_or_existing_category": fallback_id,
                "moved_clusters": linked,
            }
        )

    categories = [category for category in categories if category["linked_clusters"]]
    return categories, extractions


def enforce_min_categories(categories: list[dict], raw_clusters: list[dict], min_categories: int) -> tuple[list[dict], list[dict]]:
    """Split overly broad LLM categories by deterministic raw-cluster signals."""
    raw_by_id = {cluster["cluster_id"]: cluster for cluster in raw_clusters}
    splits = []
    while len(categories) < min_categories:
        split_index = None
        split_groups: dict[tuple[str, str], list[str]] = {}
        for index, category in enumerate(categories):
            groups: dict[tuple[str, str], list[str]] = defaultdict(list)
            for cid in category["linked_clusters"]:
                cluster = raw_by_id.get(cid)
                if not cluster:
                    continue
                fallback_id, fallback_name = fallback_category_for(cluster)
                groups[(fallback_id, fallback_name)].append(cid)
            if len(groups) > 1 and len(category["linked_clusters"]) > 1:
                if split_index is None or len(category["linked_clusters"]) > len(categories[split_index]["linked_clusters"]):
                    split_index = index
                    split_groups = groups
        if split_index is None:
            break

        source = categories.pop(split_index)
        new_categories = []
        for (fallback_id, fallback_name), linked in sorted(split_groups.items(), key=lambda item: (-len(item[1]), item[0][0])):
            signals = [dominant_pattern(raw_by_id[cid]) for cid in linked if cid in raw_by_id]
            new_categories.append(
                {
                    "id": fallback_id,
                    "name": fallback_name,
                    "description": (
                        f"Automatically split from broad LLM category '{source['name']}' "
                        "using dominant raw-cluster failure signals."
                    ),
                    "linked_clusters": linked,
                    "common_failure_signals": sorted(set(signals))[:8],
                    "rubric_hint": source.get("rubric_hint") or "Check this failure mode without relying on hidden tests.",
                    "score_focus": source.get("score_focus") or "Distinguish severe, partial, and minor instances of this failure mode.",
                    "confidence": "medium" if source.get("confidence") == "high" else source.get("confidence", "medium"),
                }
            )
        categories.extend(new_categories)
        splits.append(
            {
                "source_category": source["id"],
                "new_categories": [category["id"] for category in new_categories],
            }
        )

    return categories, splits


def enforce_max_categories(categories: list[dict], max_categories: int) -> tuple[list[dict], list[dict]]:
    """Merge closest operational categories when automatic repair creates too many."""
    merge_plan = [
        (
            "format_or_duplicate_code_output",
            "syntax_parseability_or_truncation",
            "syntax_parseability_or_output_format",
            "Syntax Parseability or Output Format",
            "Evaluate whether the answer is valid, complete Python code without truncation, duplicated definitions, or formatting artifacts.",
        ),
        (
            "numeric_formula_correctness",
            "algorithmic_wrong_value",
            "algorithmic_or_numeric_correctness",
            "Algorithmic or Numeric Correctness",
            "Evaluate whether the implementation uses the correct algorithm, formula, and computations to produce correct values.",
        ),
        (
            "edge_case_boundary_handling",
            "algorithmic_wrong_value",
            "algorithmic_edge_case_correctness",
            "Algorithmic and Edge-Case Correctness",
            "Evaluate whether the implementation is correct for normal cases and boundary conditions.",
        ),
    ]
    merges = []

    def find(category_id: str) -> dict | None:
        return next((category for category in categories if category["id"] == category_id), None)

    for source_id, target_id, merged_id, merged_name, merged_description in merge_plan:
        if len(categories) <= max_categories:
            break
        source = find(source_id)
        target = find(target_id)
        if not source or not target:
            continue
        target["id"] = merged_id
        target["name"] = merged_name
        target["description"] = merged_description
        target["linked_clusters"].extend(source.get("linked_clusters") or [])
        target["common_failure_signals"].extend(source.get("common_failure_signals") or [])
        target["common_failure_signals"] = sorted(set(target["common_failure_signals"]))[:8]
        target["rubric_hint"] = target.get("rubric_hint") or source.get("rubric_hint") or "Check this failure mode without relying on hidden tests."
        target["score_focus"] = (
            target.get("score_focus")
            or source.get("score_focus")
            or "Distinguish invalid, partially usable, and clean correct code."
        )
        categories = [category for category in categories if category is not source]
        merges.append(
            {
                "source_category": source_id,
                "target_category": target_id,
                "merged_category": merged_id,
            }
        )

    return categories, merges


def category_stats(categories: list[dict], raw_clusters: list[dict]) -> list[dict]:
    raw_by_id = {cluster["cluster_id"]: cluster for cluster in raw_clusters}
    enriched = []
    for category in categories:
        clusters = [raw_by_id[cid] for cid in category["linked_clusters"] if cid in raw_by_id]
        failure_types = Counter()
        error_patterns = Counter()
        response_count = 0
        unique_task_count = 0
        for cluster in clusters:
            failure_types.update(cluster.get("failure_types") or {})
            error_patterns.update(cluster.get("error_patterns") or cluster.get("rule_patterns") or {})
            response_count += int(cluster.get("response_count", cluster.get("size", 0)) or 0)
            unique_task_count += int(cluster.get("unique_task_count", 0) or 0)
        enriched.append(
            {
                **category,
                "response_count": response_count,
                "unique_task_count_upper_bound": unique_task_count,
                "failure_types": dict(failure_types.most_common()),
                "error_patterns": dict(error_patterns.most_common()),
            }
        )
    return enriched


def leakage_flags(obj: Any) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False).lower()
    return sorted(token for token in PRIVATE_TOKENS if token in text)


def build_cluster_mapping(categories: list[dict], raw_clusters: list[dict]) -> list[dict]:
    raw_by_id = {cluster["cluster_id"]: cluster for cluster in raw_clusters}
    rows = []
    for category in categories:
        for cid in category["linked_clusters"]:
            cluster = raw_by_id.get(cid, {})
            rows.append(
                {
                    "cluster_id": cid,
                    "cluster_name": cluster.get("name"),
                    "category_id": category["id"],
                    "category_name": category["name"],
                    "response_count": cluster.get("response_count", cluster.get("size")),
                    "unique_task_count": cluster.get("unique_task_count"),
                    "dominant_error_pattern": dominant_pattern(cluster),
                }
            )
    return sorted(rows, key=lambda row: row["cluster_id"])


def build_response_assignments(assignments_path: Path, cluster_mapping: list[dict]) -> list[dict]:
    rows = read_jsonl(assignments_path)
    by_cluster = {row["cluster_id"]: row for row in cluster_mapping}
    output = []
    for row in rows:
        mapped = by_cluster.get(row.get("cluster_id"))
        if not mapped:
            continue
        output.append(
            {
                **row,
                "taxonomy_category_id": mapped["category_id"],
                "taxonomy_category_name": mapped["category_name"],
            }
        )
    return output


def run_llm(args: argparse.Namespace, prompt: str) -> tuple[dict | None, str]:
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
    )
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=42,
    )
    output = llm.generate([prompt], sampling)[0].outputs[0].text
    return parse_json_object(output), output


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically consolidate raw discovered taxonomy clusters.")
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--raw-assignments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--cluster-mapping-output", type=Path, required=True)
    parser.add_argument("--response-assignments-output", type=Path)
    parser.add_argument("--raw-llm-output", type=Path)
    parser.add_argument("--existing-llm-output", type=Path, help="Parse an existing raw LLM response instead of calling vLLM.")
    parser.add_argument("--model", type=str, default="models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-categories", type=int, default=6)
    parser.add_argument("--max-categories", type=int, default=8)
    parser.add_argument("--skip-llm", action="store_true", help="Use deterministic fallback only.")
    args = parser.parse_args()

    raw_taxonomy = load_yaml(args.taxonomy)
    raw_clusters = raw_taxonomy.get("clusters") or raw_taxonomy.get("patterns") or []
    raw_clusters = [cluster for cluster in raw_clusters if cluster.get("cluster_id")]
    if not raw_clusters:
        raise ValueError(f"No clusters found in {args.taxonomy}")

    raw_llm_text = ""
    parsed: dict | None = None
    used_fallback = False
    if args.existing_llm_output:
        raw_llm_text = args.existing_llm_output.read_text(encoding="utf-8")
        parsed = parse_json_object(raw_llm_text)
    elif not args.skip_llm:
        prompt = build_consolidation_prompt(raw_taxonomy, args.min_categories, args.max_categories)
        parsed, raw_llm_text = run_llm(args, prompt)
    if parsed is None:
        parsed = deterministic_fallback(raw_clusters)
        used_fallback = True

    categories, repair_audit = normalize_categories(parsed, raw_clusters)
    categories, broad_replacements = operationalize_broad_categories(categories, raw_clusters)
    repair_audit["broad_category_replacements"] = broad_replacements
    categories, category_extractions = extract_specific_categories(categories, raw_clusters, args.max_categories)
    repair_audit["category_extractions"] = category_extractions
    categories, category_splits = enforce_min_categories(categories, raw_clusters, args.min_categories)
    repair_audit["category_splits"] = category_splits
    categories, category_merges = enforce_max_categories(categories, args.max_categories)
    repair_audit["category_merges"] = category_merges
    categories = category_stats(categories, raw_clusters)
    cluster_mapping = build_cluster_mapping(categories, raw_clusters)

    covered = [cid for category in categories for cid in category["linked_clusters"]]
    counts = Counter(covered)
    missing = sorted({cluster["cluster_id"] for cluster in raw_clusters} - set(covered))
    duplicates = sorted(cid for cid, count in counts.items() if count > 1)
    unknown = sorted(set(covered) - {cluster["cluster_id"] for cluster in raw_clusters})
    leak_flags = leakage_flags(categories)
    broad_flags = [
        {"id": category.get("id"), "name": category.get("name")}
        for category in categories
        if is_broad_category(category)
    ]
    category_count = len(categories)

    audit = {
        "source_taxonomy": str(args.taxonomy),
        "method": "llm_cluster_consolidation_with_deterministic_coverage_audit",
        "model": None if args.skip_llm else args.model,
        "used_deterministic_fallback": used_fallback,
        "raw_cluster_count": len(raw_clusters),
        "category_count": category_count,
        "covered_cluster_count": len(set(covered)),
        "missing_clusters": missing,
        "duplicate_clusters": duplicates,
        "unknown_clusters": unknown,
        "private_leakage_flags": leak_flags,
        "broad_category_flags": broad_flags,
        "repair_audit": repair_audit,
        "valid": (
            not missing
            and not duplicates
            and not unknown
            and not leak_flags
            and not broad_flags
            and args.min_categories <= category_count <= args.max_categories
        ),
    }

    output_taxonomy = {
        "name": parsed.get("taxonomy_name") or "mbpp_consolidated_error_taxonomy_v1",
        "source_taxonomy": str(args.taxonomy),
        "source_method": raw_taxonomy.get("method"),
        "method": "LLM semantic consolidation of raw clusters with deterministic audit",
        "total_raw_clusters": len(raw_clusters),
        "total_failures": raw_taxonomy.get("total_failures"),
        "total_tasks": raw_taxonomy.get("total_tasks"),
        "num_categories": len(categories),
        "categories": categories,
        "cluster_mapping": cluster_mapping,
        "audit": audit,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output_taxonomy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.cluster_mapping_output, cluster_mapping)
    if args.raw_llm_output:
        args.raw_llm_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_llm_output.write_text(raw_llm_text, encoding="utf-8")
    if args.raw_assignments and args.response_assignments_output:
        write_jsonl(args.response_assignments_output, build_response_assignments(args.raw_assignments, cluster_mapping))

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
