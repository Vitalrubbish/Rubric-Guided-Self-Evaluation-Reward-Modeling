#!/usr/bin/env python3
"""Cluster failure samples and build a refined coding error taxonomy."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import hdbscan
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact(text: str, limit: int) -> str:
    text = (text or "").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text[:limit]


def sample_text(row: dict) -> str:
    return "\n".join(
        [
            f"[dataset] {row.get('dataset')}",
            f"[failure_type] {row.get('failure_type')}",
            f"[rule_pattern] {row.get('error_pattern')}",
            f"[error] {compact(row.get('error', ''), 400)}",
            f"[prompt] {compact(row.get('prompt', ''), 700)}",
            f"[code] {compact(row.get('extracted_code') or row.get('generated_code') or '', 1200)}",
        ]
    )


def top_terms_for_cluster(tfidf_matrix, vectorizer: TfidfVectorizer, indices: list[int], limit: int = 10) -> list[str]:
    if not indices:
        return []
    centroid = np.asarray(tfidf_matrix[indices].mean(axis=0)).ravel()
    if centroid.size == 0:
        return []
    feature_names = np.array(vectorizer.get_feature_names_out())
    top = centroid.argsort()[::-1][:limit]
    return [str(feature_names[i]) for i in top if centroid[i] > 0]


def label_cluster(rows: list[dict], top_terms: list[str]) -> str:
    patterns = Counter(row.get("error_pattern", "unknown") for row in rows)
    failure_types = Counter(row.get("failure_type", "unknown") for row in rows)
    dominant_pattern, pattern_count = patterns.most_common(1)[0]
    dominant_type, type_count = failure_types.most_common(1)[0]
    pattern_ratio = pattern_count / len(rows)
    type_ratio = type_count / len(rows)
    if pattern_ratio >= 0.55:
        return dominant_pattern
    if type_ratio >= 0.65:
        return f"{dominant_type}_mixed_{'_'.join(top_terms[:3])}"
    return f"mixed_{'_'.join(top_terms[:4])}"


def describe_cluster(label: str, rows: list[dict], top_terms: list[str]) -> str:
    patterns = Counter(row.get("error_pattern", "unknown") for row in rows).most_common(3)
    failures = Counter(row.get("failure_type", "unknown") for row in rows).most_common(3)
    pattern_text = ", ".join(f"{name} ({count})" for name, count in patterns)
    failure_text = ", ".join(f"{name} ({count})" for name, count in failures)
    terms = ", ".join(top_terms[:6])
    return f"Cluster dominated by {pattern_text}; failure types: {failure_text}; characteristic terms: {terms}."


def example_for(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "failure_type": row.get("failure_type"),
        "error_pattern": row.get("error_pattern"),
        "error": compact(row.get("error", ""), 260),
        "snippet": compact(row.get("extracted_code") or row.get("generated_code") or "", 360),
    }


def choose_cluster_labels(features, min_cluster_size: int, min_samples: int) -> np.ndarray:
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean").fit_predict(features)
    cluster_count = len({int(label) for label in labels if label != -1})
    noise_ratio = float(np.mean(labels == -1)) if len(labels) else 1.0
    if cluster_count >= 4 and noise_ratio <= 0.55:
        return labels

    # Deterministic fallback: enough clusters to inspect, not too many to be unreadable.
    k = min(18, max(8, round(math.sqrt(len(labels)))))
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--assignments-output", type=Path, default=Path("data/analysis/failure_clusters_qwen25_k1.jsonl"))
    parser.add_argument("--taxonomy-output", type=Path, default=Path("data/analysis/coding_error_taxonomy_refined.yaml"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/analysis/coding_error_taxonomy_refined_summary.json"))
    parser.add_argument("--min-cluster-size", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=4)
    args = parser.parse_args()

    rows = list(read_jsonl(args.failures))
    texts = [sample_text(row) for row in rows]
    vectorizer = TfidfVectorizer(max_features=6000, min_df=2, ngram_range=(1, 2), stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(64, max(2, tfidf.shape[1] - 1), max(2, len(rows) - 1))
    features = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(tfidf)
    features = normalize(features)
    labels = choose_cluster_labels(features, args.min_cluster_size, args.min_samples)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    taxonomy_patterns = []
    assignments = []
    for new_id, (raw_label, indices) in enumerate(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))):
        cluster_rows = [rows[i] for i in indices]
        top_terms = top_terms_for_cluster(tfidf, vectorizer, indices)
        label = label_cluster(cluster_rows, top_terms)
        datasets = Counter(row.get("dataset") for row in cluster_rows)
        failure_types = Counter(row.get("failure_type") for row in cluster_rows)
        rule_patterns = Counter(row.get("error_pattern") for row in cluster_rows)
        examples = [example_for(row) for row in cluster_rows[:5]]
        pattern = {
            "cluster_id": f"cluster_{new_id:02d}",
            "raw_cluster_label": int(raw_label),
            "name": label,
            "description": describe_cluster(label, cluster_rows, top_terms),
            "frequency": len(cluster_rows),
            "ratio_among_failures": round(len(cluster_rows) / len(rows), 4) if rows else 0,
            "datasets": dict(datasets),
            "failure_types": dict(failure_types),
            "rule_patterns": dict(rule_patterns),
            "top_terms": top_terms,
            "example_ids": [row.get("id") for row in cluster_rows[:12]],
            "examples": examples,
        }
        taxonomy_patterns.append(pattern)
        for row in cluster_rows:
            assignments.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "failure_type": row.get("failure_type"),
                    "error_pattern": row.get("error_pattern"),
                    "cluster_id": pattern["cluster_id"],
                    "cluster_name": pattern["name"],
                }
            )

    summary = {
        "input": str(args.failures),
        "total_failures": len(rows),
        "num_clusters": len(taxonomy_patterns),
        "cluster_sizes": {item["cluster_id"]: item["frequency"] for item in taxonomy_patterns},
        "top_clusters": [
            {
                "cluster_id": item["cluster_id"],
                "name": item["name"],
                "frequency": item["frequency"],
                "ratio_among_failures": item["ratio_among_failures"],
            }
            for item in taxonomy_patterns[:10]
        ],
    }
    taxonomy = {
        "name": "coding_error_taxonomy_refined_tfidf_hdbscan",
        "source_failures": str(args.failures),
        "method": "TF-IDF ngrams -> TruncatedSVD -> normalized features -> HDBSCAN with KMeans fallback",
        "total_failures": len(rows),
        "num_clusters": len(taxonomy_patterns),
        "patterns": taxonomy_patterns,
    }

    args.assignments_output.parent.mkdir(parents=True, exist_ok=True)
    args.taxonomy_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.assignments_output.open("w", encoding="utf-8") as f:
        for row in sorted(assignments, key=lambda item: item["id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.taxonomy_output.write_text(yaml.safe_dump(taxonomy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
