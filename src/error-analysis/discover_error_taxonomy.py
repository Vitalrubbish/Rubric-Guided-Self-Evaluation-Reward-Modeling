#!/usr/bin/env python3
"""Two-stage error taxonomy discovery: LLM summarization + ML clustering.

Stage 1 -- LLM free-form summarization:
  Each failure is shown to an LLM with the task, generated code, and error info.
  The LLM outputs a one-sentence root cause summary (NO predefined taxonomy).
  This produces a JSONL with an added "llm_summary" field.

Stage 2 -- TF-IDF embedding + recursive clustering:
  Summaries are vectorized with TF-IDF (unigrams + bigrams), reduced via
  TruncatedSVD, then clustered with HDBSCAN (KMeans fallback). Clusters
  exceeding --max-cluster-ratio (default 25%) are recursively sub-clustered
  to reveal finer-grained error categories. Cluster labels are derived
  entirely from LLM summary keywords, not from rule-based error_pattern.

Outputs:
  - Stage 1: failure JSONL with llm_summary field
  - Stage 2: cluster assignments JSONL, taxonomy YAML, summary JSON
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import hdbscan
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from vllm import LLM, SamplingParams


PRIVATE_FIELDS = {"test_list", "test", "test_setup_code", "private_diagnostics"}

# ---------------------------------------------------------------------------
# Prompt for LLM summarization (free-form, no predefined categories)
# ---------------------------------------------------------------------------
SUMMARIZE_SYSTEM = (
    "You are an expert code reviewer. Your task is to identify the root cause "
    "of a code failure in ONE concise sentence. Focus on what went wrong in "
    "the generated code, NOT on what the correct solution should be. "
    "Be specific and concrete."
)


def build_summarize_prompt(row: dict) -> str:
    """Construct a prompt asking for a free-form one-sentence root cause summary."""
    prompt_text = row.get("prompt", "")
    task = prompt_text
    task_match = re.search(r"Task:\s*(.+?)(?:\n\n|$)", prompt_text, re.DOTALL)
    if task_match:
        task = task_match.group(1).strip()

    code = row.get("extracted_code") or row.get("generated_code") or ""
    if len(code) > 1800:
        code = code[:1800] + "\n# ... [truncated]"

    failure_type = row.get("failure_type", "unknown")
    error_msg = (row.get("error") or "").strip()
    safe_diagnostics = row.get("safe_diagnostics") or {}
    diagnostics_text = json.dumps(safe_diagnostics, ensure_ascii=False, sort_keys=True)
    if len(diagnostics_text) > 1200:
        diagnostics_text = diagnostics_text[:1197] + "..."

    return (
        f"{SUMMARIZE_SYSTEM}\n\n"
        f"## Task\n{task}\n\n"
        f"## Generated Code\n```python\n{code}\n```\n\n"
        f"## Execution Result\n"
        f"- Failure type: {failure_type}\n"
        f"- Error message: {error_msg or '(none — likely wrong output)'}\n\n"
        f"## Safe Verifier Diagnostics\n"
        f"{diagnostics_text or '{}'}\n\n"
        f"## Instructions\n"
        f"Describe the root cause of this failure in ONE concise sentence. "
        f"Focus on what the code did wrong, not what it should have done.\n\n"
        f'Respond with: {{"summary": "<one sentence>"}}'
    )


def parse_summary(text: str) -> str:
    """Extract the summary string from LLM output."""
    try:
        data = json.loads(text.strip())
        s = str(data.get("summary", "")).strip()
        if s:
            return s
    except (json.JSONDecodeError, AttributeError):
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1))
            s = str(data.get("summary", "")).strip()
            if s:
                return s
        except (json.JSONDecodeError, AttributeError):
            pass
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group())
            s = str(data.get("summary", "")).strip()
            if s:
                return s
        except json.JSONDecodeError:
            continue
    return text.strip()[:300]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def private_fields_present(row: dict) -> list[str]:
    return sorted(key for key in PRIVATE_FIELDS if key in row and row.get(key) not in (None, [], ""))


def strip_private_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in PRIVATE_FIELDS}


def response_id(row: dict, fallback_index: int | None = None) -> str:
    if row.get("response_id"):
        return str(row["response_id"])
    problem_id = row.get("id", str(fallback_index) if fallback_index is not None else "unknown")
    return f"{problem_id}__sample{row.get('sample_id', 0)}"


# ---------------------------------------------------------------------------
# Stage 2: TF-IDF + clustering helpers
# ---------------------------------------------------------------------------
def build_text_features(rows: list[dict]) -> list[str]:
    """Encode each row as text for TF-IDF, centered on the model-written cause."""
    texts = []
    for row in rows:
        parts = [
            row.get("llm_summary", ""),
            f"[failure_type: {row.get('failure_type', 'unknown')}]",
            f"[error: {(row.get('error') or '')[:200]}]",
        ]
        texts.append(" ".join(parts))
    return texts


def fit_vectorizer(texts: list[str]) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """TF-IDF -> TruncatedSVD -> normalize. Returns (vectorizer, tfidf, features)."""
    vectorizer = TfidfVectorizer(
        max_features=5000, min_df=2, ngram_range=(1, 2), stop_words="english"
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(64, max(2, tfidf.shape[1] - 1), max(2, len(texts) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    features = svd.fit_transform(tfidf)
    features = normalize(features)
    print(f"  TF-IDF: {tfidf.shape[1]} terms -> SVD {n_components}d")
    return vectorizer, tfidf, features


def cluster_features(features: np.ndarray, min_cluster_size: int,
                     min_samples: int) -> tuple[np.ndarray, str]:
    """HDBSCAN with KMeans fallback. Returns (labels, method_description)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(features)
    cluster_count = len({int(l) for l in labels if l != -1})
    noise_ratio = float(np.mean(labels == -1))

    if cluster_count >= 3 and noise_ratio <= 0.55:
        method = f"HDBSCAN(min_cluster={min_cluster_size}, min_samples={min_samples}) -> {cluster_count} clusters, noise={noise_ratio:.1%}"
        print(f"  {method}")
    else:
        k = min(15, max(4, round(math.sqrt(len(labels)))))
        method = f"HDBSCAN gave {cluster_count} clusters ({noise_ratio:.0%} noise), fallback KMeans(k={k})"
        print(f"  {method}")
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(features)

    # Reassign noise points to nearest centroid
    if -1 in labels:
        unique_labels = sorted(set(labels) - {-1})
        centroids = np.array([features[labels == l].mean(axis=0) for l in unique_labels])
        for i in np.where(labels == -1)[0]:
            dists = np.linalg.norm(centroids - features[i], axis=1)
            labels[i] = unique_labels[int(np.argmin(dists))]
        print(f"  Noise points reassigned to nearest centroid")

    return labels, method


def summary_keywords(summaries: list[str], top_n: int = 8) -> list[str]:
    """Extract distinctive keywords from a list of LLM summaries."""
    if not summaries:
        return []
    # Lightweight: use a local TF-IDF on just this cluster's summaries
    vec = TfidfVectorizer(
        max_features=200, min_df=1, ngram_range=(1, 2),
        stop_words="english", sublinear_tf=True,
    )
    try:
        tfidf = vec.fit_transform(summaries)
    except ValueError:
        return []
    centroid = np.asarray(tfidf.mean(axis=0)).ravel()
    names = np.array(vec.get_feature_names_out())
    top = centroid.argsort()[::-1][:top_n]
    return [str(names[i]) for i in top if centroid[i] > 0]


def label_cluster_from_summaries(summaries: list[str]) -> str:
    """Generate a human-readable label purely from LLM summary keywords."""
    kw = summary_keywords(summaries, top_n=8)
    if not kw:
        return "unlabeled"
    # Normalize: TF-IDF bigrams use spaces, unigrams are single words.
    # Normalize all to underscores, then split and deduplicate individual words.
    seen_tokens: set[str] = set()
    deduped: list[str] = []
    for k in kw:
        # Normalize spaces to underscores for consistent splitting
        normalized = k.replace(" ", "_")
        parts = normalized.split("_")
        # Keep only previously unseen words
        unique_parts = [p for p in parts if p and p not in seen_tokens]
        if not unique_parts:
            continue
        deduped.append("_".join(unique_parts))
        seen_tokens.update(parts)
        if len(deduped) >= 3:
            break
    if not deduped:
        return "unlabeled"
    return f"discovered_{'_'.join(deduped)}"


def build_taxonomy_entry(
    cluster_id: str, indices: list[int], rows: list[dict],
    total_responses: int, total_tasks: int, parent_id: str | None,
    vectorizer: TfidfVectorizer | None, tfidf_matrix,
) -> dict:
    """Build a single taxonomy entry for a cluster."""
    cluster_rows = [rows[i] for i in indices]
    summaries = [r.get("llm_summary", "") for r in cluster_rows]
    label_str = label_cluster_from_summaries(summaries)

    failure_types = Counter(r.get("failure_type", "unknown") for r in cluster_rows)
    error_patterns = Counter(r.get("error_pattern", "unknown") for r in cluster_rows)
    datasets = Counter(r.get("dataset", "unknown") for r in cluster_rows)
    task_counts = Counter(r.get("id", "unknown") for r in cluster_rows)
    sample_ids = Counter(str(r.get("sample_id", 0)) for r in cluster_rows)

    # TF-IDF top terms (global vectorizer)
    top_terms = []
    if vectorizer is not None and len(indices) > 0:
        centroid = np.asarray(tfidf_matrix[indices].mean(axis=0)).ravel()
        if centroid.size > 0:
            names = np.array(vectorizer.get_feature_names_out())
            top = centroid.argsort()[::-1][:8]
            top_terms = [str(names[i]) for i in top if centroid[i] > 0]

    # Diverse examples
    examples = []
    seen_fts = set()
    for r in cluster_rows:
        ft = r.get("failure_type", "unknown")
        if ft not in seen_fts or len(examples) < 3:
            seen_fts.add(ft)
            examples.append({
                "response_id": response_id(r),
                "id": r.get("id"),
                "sample_id": r.get("sample_id", 0),
                "dataset": r.get("dataset"),
                "failure_type": ft,
                "error": (r.get("error") or "")[:200],
                "summary": r.get("llm_summary", "")[:200],
                "snippet": (r.get("extracted_code") or r.get("generated_code") or "")[:300],
            })
        if len(examples) >= 5:
            break

    entry = {
        "cluster_id": cluster_id,
        "name": label_str,
        "size": len(cluster_rows),
        "response_count": len(cluster_rows),
        "unique_task_count": len(task_counts),
        "ratio": round(len(cluster_rows) / total_responses, 4) if total_responses else 0,
        "failure_types": dict(failure_types),
        "error_patterns": dict(error_patterns),
        "datasets": dict(datasets),
        "task_ratio": round(len(task_counts) / total_tasks, 4) if total_tasks else 0,
        "tasks_with_multiple_failed_samples": sum(1 for count in task_counts.values() if count > 1),
        "sample_id_distribution": dict(sample_ids),
        "top_terms": top_terms,
        "examples": examples,
        "_indices": list(indices),  # internal: row indices for recursive sub-clustering
    }
    if parent_id:
        entry["parent_cluster"] = parent_id
    return entry


def cluster_single_level(
    rows: list[dict], total: int, total_tasks: int, cluster_id_prefix: str,
    min_cluster_size: int, min_samples: int,
    vectorizer: TfidfVectorizer | None, tfidf_matrix,
    parent_id: str | None,
) -> list[dict]:
    """Run one level of clustering. Returns list of taxonomy entries."""
    texts = build_text_features(rows)
    if vectorizer is None:
        vec, tfidf, features = fit_vectorizer(texts)
    else:
        tfidf = vectorizer.transform(texts)
        n_components = min(64, max(2, tfidf.shape[1] - 1), max(2, len(texts) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        features = normalize(svd.fit_transform(tfidf))
        vec = vectorizer

    labels, _ = cluster_features(
        features,
        max(3, min(min_cluster_size, len(rows) // 6)),
        max(1, min(min_samples, len(rows) // 12)),
    )

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[int(label)].append(idx)

    entries = []
    for new_id, (raw_label, indices) in enumerate(
        sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0]))
    ):
        cid = f"{cluster_id_prefix}_{new_id:02d}" if parent_id else f"cluster_{new_id:02d}"
        entry = build_taxonomy_entry(cid, indices, rows, total, total_tasks, parent_id, vec, tfidf)
        entries.append(entry)
    return entries


def recursive_subcluster(
    entries: list[dict], rows: list[dict], total: int,
    max_ratio: float, min_cluster_size: int, min_samples: int,
    subcluster_min_cluster_size: int | None = None,
    subcluster_min_samples: int | None = None,
    max_depth: int = 3,
) -> list[dict]:
    """Recursively sub-cluster any entry exceeding max_ratio of total."""
    for _ in range(max_depth):
        result = []
        changed = False
        for entry in entries:
            if entry["ratio"] <= max_ratio:
                result.append(entry)
                continue

            sub_indices = entry.get("_indices", [])
            if len(sub_indices) < 10:
                result.append(entry)
                continue

            sub_rows = [rows[i] for i in sub_indices]
            print(f"\n  Sub-clustering '{entry['name']}' ({len(sub_rows)} samples, "
                  f"{entry['ratio']*100:.1f}% of total)...")

            auto_sub_min_cluster = max(
                min_cluster_size,
                min(24, max(8, len(sub_rows) // 30)),
            )
            sub_min_cluster = subcluster_min_cluster_size or auto_sub_min_cluster
            sub_min_samples = subcluster_min_samples or max(min_samples, min(5, max(1, sub_min_cluster // 4)))
            sub_entries = cluster_single_level(
                sub_rows, total, len({row.get("id") for row in rows}), entry["cluster_id"],
                sub_min_cluster,
                sub_min_samples,
                None, None,
                entry["cluster_id"],
            )

            for se in sub_entries:
                se["ratio"] = round(se["size"] / total, 4)
                se["_indices"] = [sub_indices[i] for i in se["_indices"]]
            result.extend(sub_entries)
            changed = True
            print(f"    -> split into {len(sub_entries)} sub-clusters "
                  f"(largest={max(e['size'] for e in sub_entries)}, "
                  f"smallest={min(e['size'] for e in sub_entries)})")

        if not changed:
            break
        entries = result

    return entries


# ---------------------------------------------------------------------------
# Stage 1: LLM summarization
# ---------------------------------------------------------------------------
def run_stage1(args: argparse.Namespace) -> Path:
    """Run LLM summarization and return path to the enriched failure JSONL."""
    rows = list(read_jsonl(args.failures))
    print(f"Stage 1: Loaded {len(rows)} failure samples from {args.failures}")
    if not args.allow_private_fields:
        offenders = [(response_id(row, idx), private_fields_present(row)) for idx, row in enumerate(rows) if private_fields_present(row)]
        if offenders:
            preview = ", ".join(f"{rid}:{fields}" for rid, fields in offenders[:5])
            raise ValueError(
                "Refusing to run attribution on rows with private verifier fields. "
                f"Found {len(offenders)} offending rows, e.g. {preview}. "
                "Use sanitized failure artifacts or pass --allow-private-fields for private/debug analysis only."
            )

    prompts = [build_summarize_prompt(row) for row in rows]

    print(f"Loading summarization model: {args.summarize_model}")
    llm = LLM(
        model=args.summarize_model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=getattr(args, "max_model_len", 8192),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.35),
        max_num_seqs=getattr(args, "batch_size", 64),
    )

    sampling = SamplingParams(
        n=1,
        temperature=getattr(args, "temperature", 0.0),
        max_tokens=getattr(args, "max_tokens", 128),
        seed=42,
    )

    print(f"Running summarization on {len(prompts)} samples ...")
    outputs = llm.generate(prompts, sampling)

    enriched = []
    for row, out in zip(rows, outputs):
        raw = out.outputs[0].text
        summary = parse_summary(raw)
        base = row if args.allow_private_fields else strip_private_fields(row)
        enriched.append({**base, "llm_summary": summary, "llm_raw_summary_response": raw})

    out_path = args.stage1_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Stage 1 done: {len(enriched)} enriched records written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stage 2: Clustering
# ---------------------------------------------------------------------------
def run_stage2(args: argparse.Namespace, stage1_path: Path) -> None:
    """Embed summaries, cluster recursively, and produce taxonomy outputs."""
    rows = list(read_jsonl(stage1_path))
    if not args.allow_private_fields:
        offenders = [(response_id(row, idx), private_fields_present(row)) for idx, row in enumerate(rows) if private_fields_present(row)]
        if offenders:
            preview = ", ".join(f"{rid}:{fields}" for rid, fields in offenders[:5])
            raise ValueError(
                "Refusing to cluster rows with private verifier fields. "
                f"Found {len(offenders)} offending rows, e.g. {preview}. "
                "Use sanitized failure artifacts or pass --allow-private-fields for private/debug analysis only."
            )
    summaries = [r.get("llm_summary", "") for r in rows]
    total = len(rows)
    total_tasks = len({r.get("id") for r in rows})
    print(f"Stage 2: Loaded {total} records, {sum(1 for s in summaries if s)} have summaries")

    min_cs = getattr(args, "min_cluster_size", 8)
    min_s = getattr(args, "min_samples", 3)
    max_ratio = getattr(args, "max_cluster_ratio", 0.25)

    # Level 1: cluster all rows
    texts = build_text_features(rows)
    vectorizer, tfidf, features = fit_vectorizer(texts)
    labels, method_desc = cluster_features(features, min_cs, min_s)

    # Group
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[int(label)].append(idx)

    # Build level-1 entries
    entries = []
    for new_id, (raw_label, indices) in enumerate(
        sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0]))
    ):
        cid = f"cluster_{new_id:02d}"
        entry = build_taxonomy_entry(cid, indices, rows, total, total_tasks, None, vectorizer, tfidf)
        entries.append(entry)

    # Recursively sub-cluster oversized clusters
    max_ratio = getattr(args, "max_cluster_ratio", 0.25)
    print(f"\nRecursive sub-clustering (threshold >{max_ratio*100:.0f}% of total)...")
    oversized = [e for e in entries if e["ratio"] > max_ratio]
    if oversized:
        for e in oversized:
            print(f"  Oversized: {e['cluster_id']} '{e['name']}' "
                  f"({e['size']} samples, {e['ratio']*100:.1f}%)")
        entries = recursive_subcluster(
            entries,
            rows,
            total,
            max_ratio,
            min_cs,
            min_s,
            getattr(args, "subcluster_min_cluster_size", None),
            getattr(args, "subcluster_min_samples", None),
        )
    else:
        print(f"  No clusters exceed {max_ratio*100:.0f}% threshold")

    # Flatten hierarchy for display
    print(f"\nDiscovered taxonomy ({len(entries)} categories):")
    entries.sort(key=lambda e: (-e["size"], e["cluster_id"]))
    for e in entries:
        prefix = "  " if e.get("parent_cluster") else ""
        bar = "█" * max(1, e["size"] * 50 // total)
        parent = f" (sub of {e['parent_cluster']})" if e.get("parent_cluster") else ""
        print(f"  {prefix}{e['cluster_id']:14s} {e['name'][:48]:48s} {e['size']:4d} ({e['ratio']*100:5.1f}%){parent} {bar}")

    # Rebuild assignments from final entries (which carry correct _indices)
    final_assignments = {}
    for e in entries:
        for idx in e.get("_indices", []):
            r = rows[idx]
            uid = response_id(r, idx)
            final_assignments[uid] = {
                "response_id": uid,
                "id": r.get("id"),
                "sample_id": r.get("sample_id", 0),
                "dataset": r.get("dataset"),
                "split": r.get("split"),
                "failure_type": r.get("failure_type"),
                "error_pattern": r.get("error_pattern"),
                "safe_diagnostics": r.get("safe_diagnostics"),
                "llm_summary": r.get("llm_summary"),
                "cluster_id": e["cluster_id"],
                "cluster_name": e["name"],
            }

    # --- Write outputs ---
    assignments = sorted(final_assignments.values(), key=lambda x: x["response_id"])
    args.assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with args.assignments_output.open("w", encoding="utf-8") as f:
        for rec in assignments:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Strip internal _indices field before serializing
    clean_entries = [{k: v for k, v in e.items() if k != "_indices"} for e in entries]

    taxonomy = {
        "name": "coding_error_taxonomy_discovered",
        "source_failures": str(stage1_path),
        "method": (
            "LLM free-form summarization -> TF-IDF (1,2)-grams -> "
            "TruncatedSVD -> HDBSCAN + recursive sub-clustering"
        ),
        "total_failures": total,
        "total_tasks": total_tasks,
        "num_clusters": len(entries),
        "max_cluster_ratio": max_ratio,
        "clusters": clean_entries,
        "patterns": clean_entries,
    }
    args.taxonomy_output.parent.mkdir(parents=True, exist_ok=True)
    args.taxonomy_output.write_text(
        yaml.safe_dump(taxonomy, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    summary = {
        "input": str(stage1_path),
        "method": "llm_summarization + tfidf_svd + hdbscan + recursive_subcluster",
        "total_failures": total,
        "num_clusters": len(entries),
        "cluster_distribution": {
            e["cluster_id"]: {
                "name": e["name"],
                "size": e["size"],
                "response_count": e["response_count"],
                "unique_task_count": e["unique_task_count"],
                "ratio": e["ratio"],
                "task_ratio": e["task_ratio"],
            }
            for e in entries
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nWrote {len(assignments)} assignments to {args.assignments_output}")
    print(f"Wrote taxonomy to {args.taxonomy_output}")
    print(f"Wrote summary to {args.summary_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage error taxonomy discovery (LLM summarization + ML clustering)"
    )
    # Input / output
    parser.add_argument("--failures", type=Path, required=True,
                        help="Input failure JSONL (from build_failure_artifacts.py)")
    parser.add_argument("--stage1-output", type=Path,
                        default=Path("data/analysis/failures_with_llm_summaries.jsonl"),
                        help="Output of stage 1 (failures + llm_summary field)")
    parser.add_argument("--assignments-output", type=Path,
                        default=Path("data/analysis/discovered_taxonomy_assignments.jsonl"))
    parser.add_argument("--taxonomy-output", type=Path,
                        default=Path("data/analysis/discovered_error_taxonomy.yaml"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("data/analysis/discovered_taxonomy_summary.json"))
    # Stage control
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip LLM summarization; input already has llm_summary field")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip clustering; only run LLM summarization")
    parser.add_argument(
        "--allow-private-fields",
        action="store_true",
        help="Private/debug mode: allow input rows that contain tests or exact private diagnostics.",
    )
    # Model config
    parser.add_argument("--summarize-model", type=str,
                        default="models/models--Qwen--Qwen2.5-7B-Instruct/"
                                "snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
                        help="vLLM model for summarization (stage 1)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    # Clustering config
    parser.add_argument("--min-cluster-size", type=int, default=8,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="HDBSCAN min_samples")
    parser.add_argument("--max-cluster-ratio", type=float, default=0.25,
                        help="Recursively sub-cluster clusters above this ratio (default 0.25)")
    parser.add_argument(
        "--subcluster-min-cluster-size",
        type=int,
        default=None,
        help="Override HDBSCAN min_cluster_size for recursive sub-clustering. Defaults to a size scaled by the oversized cluster.",
    )
    parser.add_argument(
        "--subcluster-min-samples",
        type=int,
        default=None,
        help="Override HDBSCAN min_samples for recursive sub-clustering.",
    )

    args = parser.parse_args()

    if args.skip_stage1:
        stage1_path = args.failures
    else:
        stage1_path = run_stage1(args)

    if not args.skip_stage2:
        run_stage2(args, stage1_path)


if __name__ == "__main__":
    main()
