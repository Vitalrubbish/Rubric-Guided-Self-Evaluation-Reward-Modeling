#!/usr/bin/env python3
"""Download additional coding datasets from Hugging Face or a mirror.

The default endpoint uses hf-mirror.com because direct Hugging Face access can
be unreliable from the current environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import requests


DEFAULT_ENDPOINT = "https://hf-mirror.com"
BIGCODEBENCH_REPO = "bigcode/bigcodebench"
APPS_REPO = "codeparrot/apps"


def dataset_file_url(endpoint: str, repo: str, path: str) -> str:
    escaped_path = "/".join(quote(part) for part in path.split("/"))
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/main/{escaped_path}"


def download_file(url: str, output: Path, timeout: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    bytes_written = 0
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
    tmp.replace(output)
    return {"path": str(output), "url": url, "bytes": bytes_written, "mode": "full"}


def stream_jsonl_sample(url: str, output: Path, sample_lines: int, timeout: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    lines_written = 0
    bytes_written = 0
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for line in response.iter_lines():
                if not line:
                    continue
                f.write(line + b"\n")
                lines_written += 1
                bytes_written += len(line) + 1
                if lines_written >= sample_lines:
                    break
    tmp.replace(output)
    return {
        "path": str(output),
        "url": url,
        "bytes": bytes_written,
        "lines": lines_written,
        "mode": "jsonl_prefix_sample",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-bigcodebench", action="store_true")
    parser.add_argument("--bigcodebench-version", default="v0.1.4")
    parser.add_argument("--skip-apps", action="store_true")
    parser.add_argument("--apps-train-sample", type=int, default=500)
    parser.add_argument("--apps-test-sample", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/coding_dataset_download_manifest.json"),
    )
    args = parser.parse_args()

    records = []
    if not args.skip_bigcodebench:
        filename = f"data/{args.bigcodebench_version}-00000-of-00001.parquet"
        url = dataset_file_url(args.endpoint, BIGCODEBENCH_REPO, filename)
        output = args.raw_dir / "bigcodebench" / f"{args.bigcodebench_version}.parquet"
        print(f"downloading BigCodeBench {args.bigcodebench_version}: {url}")
        records.append(download_file(url, output, args.timeout))

    if not args.skip_apps:
        if args.apps_train_sample > 0:
            url = dataset_file_url(args.endpoint, APPS_REPO, "train.jsonl")
            output = args.raw_dir / "apps" / f"train_sample{args.apps_train_sample}.jsonl"
            print(f"streaming APPS train sample ({args.apps_train_sample} lines): {url}")
            records.append(stream_jsonl_sample(url, output, args.apps_train_sample, args.timeout))
        if args.apps_test_sample > 0:
            url = dataset_file_url(args.endpoint, APPS_REPO, "test.jsonl")
            output = args.raw_dir / "apps" / f"test_sample{args.apps_test_sample}.jsonl"
            print(f"streaming APPS test sample ({args.apps_test_sample} lines): {url}")
            records.append(stream_jsonl_sample(url, output, args.apps_test_sample, args.timeout))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote manifest to {args.manifest}")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
