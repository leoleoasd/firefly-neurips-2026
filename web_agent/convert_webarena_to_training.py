#!/usr/bin/env python3
"""
Convert WebArena task format to slime training data format.

WebArena format (input):
    Individual JSON files per task with task_id, intent, start_url,
    sites, eval config, etc.

Training format (output):
    {"index": int, "metadata": {<all original WebArena fields>}}
    Preserves all original information without transformation.

Usage:
    python convert_webarena_to_training.py \
        --tasks-dir dataset/train_webarena \
        --output training_data.jsonl \
        --sites shopping

Examples:
    # Convert all tasks
    python convert_webarena_to_training.py \
        --tasks-dir thirdparty/rl_web_agent/dataset/train_webarena \
        --output web_agent/data/train.jsonl

    # Filter by sites
    python convert_webarena_to_training.py \
        --tasks-dir thirdparty/rl_web_agent/dataset/test_webarena_lite \
        --output web_agent/data/test.jsonl \
        --sites shopping,shopping_admin

    # Single-site tasks only
    python convert_webarena_to_training.py \
        --tasks-dir thirdparty/rl_web_agent/dataset/train_webarena \
        --output web_agent/data/train_single_site.jsonl \
        --single-site-only
"""

import argparse
import json
from glob import glob
from pathlib import Path


def load_task_file(task_path: str) -> dict:
    """Load a single WebArena task JSON file."""
    with open(task_path) as f:
        return json.load(f)


def convert_task(task_config: dict, index: int) -> dict:
    """
    Convert a single WebArena task to training format.

    Simply wraps the original task config in {index, metadata} structure,
    preserving all original fields.

    Args:
        task_config: WebArena task configuration dict (preserved as-is)
        index: Unique index for the training sample

    Returns:
        Training format dict: {"index": int, "metadata": {original task config}}
    """
    return {"index": index, "metadata": task_config}


def main():
    parser = argparse.ArgumentParser(
        description="Convert WebArena task format to slime training data format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all training tasks
    python convert_webarena_to_training.py \\
        --tasks-dir thirdparty/rl_web_agent/dataset/train_webarena \\
        --output web_agent/data/train.jsonl

    # Convert test tasks filtered by sites
    python convert_webarena_to_training.py \\
        --tasks-dir thirdparty/rl_web_agent/dataset/test_webarena_lite \\
        --output web_agent/data/test.jsonl \\
        --sites shopping,shopping_admin,gitlab

    # Only single-site tasks
    python convert_webarena_to_training.py \\
        --tasks-dir thirdparty/rl_web_agent/dataset/train_webarena \\
        --output web_agent/data/train_single_site.jsonl \\
        --single-site-only
""",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        required=True,
        help="Directory containing WebArena task JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for training data (.jsonl)",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated list of sites to include (e.g., shopping,gitlab). "
        "If not specified, all sites are included.",
    )
    parser.add_argument(
        "--single-site-only",
        action="store_true",
        help="Only include tasks that use exactly one site",
    )
    parser.add_argument(
        "--exclude-sites",
        type=str,
        default=None,
        help="Comma-separated list of sites to exclude (e.g., map,wikipedia)",
    )

    args = parser.parse_args()

    # Parse site filters
    include_sites = None
    if args.sites:
        include_sites = set(site.strip() for site in args.sites.split(","))
        print(f"Including sites: {include_sites}")

    exclude_sites = set()
    if args.exclude_sites:
        exclude_sites = set(site.strip() for site in args.exclude_sites.split(","))
        print(f"Excluding sites: {exclude_sites}")

    # Load all task files
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    task_files = sorted(glob(str(tasks_dir / "*.json")))
    if not task_files:
        raise ValueError(f"No task JSON files found in {tasks_dir}")

    print(f"Found {len(task_files)} task files in {tasks_dir}")

    # Process tasks
    converted = []
    skipped_sites = 0
    skipped_multi_site = 0
    skipped_excluded = 0

    for task_file in task_files:
        task_config = load_task_file(task_file)
        task_sites = set(task_config["sites"])

        # Filter by single-site-only
        if args.single_site_only and len(task_sites) != 1:
            skipped_multi_site += 1
            continue

        # Filter by excluded sites
        if task_sites & exclude_sites:
            skipped_excluded += 1
            continue

        # Filter by included sites
        if include_sites is not None and not task_sites.issubset(include_sites):
            skipped_sites += 1
            continue

        # Convert task - just wrap in {index, metadata}
        training_sample = convert_task(task_config, len(converted))
        converted.append(training_sample)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in converted:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'=' * 60}")
    print("Conversion complete")
    print(f"{'=' * 60}")
    print(f"  Input tasks:              {len(task_files)}")
    print(f"  Converted:                {len(converted)}")
    if skipped_sites > 0:
        print(f"  Skipped (site filter):    {skipped_sites}")
    if skipped_multi_site > 0:
        print(f"  Skipped (multi-site):     {skipped_multi_site}")
    if skipped_excluded > 0:
        print(f"  Skipped (excluded sites): {skipped_excluded}")
    print(f"  Output: {output_path}")

    # Print site distribution
    site_counts: dict[str, int] = {}
    for sample in converted:
        for site in sample["metadata"]["sites"]:
            site_counts[site] = site_counts.get(site, 0) + 1

    if site_counts:
        print("\nSite distribution:")
        for site, count in sorted(site_counts.items(), key=lambda x: -x[1]):
            print(f"  {site}: {count}")


if __name__ == "__main__":
    main()
