#!/usr/bin/env python3
"""
Convert MCP task format to training data format.

MCP format (input):
    Flat JSON with task_description, answer_schema (placeholder template),
    ground_truth_trajectory, all_dag_tool_calls, etc.
    Tool definitions are NOT inline - they come from mcp_servers_joined.json.

Training format (output):
    {"index": int, "metadata": {task_description, answer_schema, tools, ...}}
    Tools are in OpenAI function-calling format inline.

Supports three tool modes (matching evaluator.py):
    - simple: Only ground truth trajectory tools
    - dag: Ground truth + all DAG tools
    - random: Ground truth + random tools from the full registry

Usage:
    python tool_call_agent/convert_mcp_to_training.py \
        --tasks /tmp/instance_storage/data_batches/3/train_tasks_filtered.json \
        --servers /tmp/instance_storage/mcp_servers_joined.json \
        --output /tmp/instance_storage/data_batches/3/training_data.jsonl \
        --tool-mode dag
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Literal

# Tool mode type - matches evaluator.py
ToolMode = Literal["simple", "random", "dag"]


def normalize_tool_name(tool_id: str) -> str:
    """Convert tool_id to normalized tool name.

    EXACTLY matches the format used in evaluator.py / batch_callbacks.py / dag_tool_callback.py:
    - Format: {normalized_server}__{tool_name}
    - Server: normalized (@ removed, / and - replaced with _)
    - Tool name: kept AS-IS (NOT normalized)

    Example:
        "@BarnacleLabs/chimera-mcp-smithery::search_papers"
        -> "BarnacleLabs_chimera_mcp_smithery__search_papers"
    """
    if "::" in tool_id:
        server_part, tool_name = tool_id.split("::", 1)
    else:
        server_part = ""
        tool_name = tool_id

    # Normalize server name ONLY (convert @, /, - to _)
    normalized_server = server_part.replace("@", "").replace("/", "_").replace("-", "_")

    return f"{normalized_server}__{tool_name}"


def validate_tool_name(name: str) -> bool:
    """Check if tool name is valid for OpenAI/Bedrock function calling format.

    Pattern: ^[a-zA-Z0-9_-]{1,128}$
    """
    return bool(re.match(r"^[a-zA-Z0-9_-]{1,128}$", name))


def load_tool_registry(servers_path: str) -> dict[str, dict]:
    """
    Build a registry mapping tool_id -> OpenAI-format tool definition.

    Tool names are normalized using the same convention as evaluator.py:
    {normalized_server}__{tool_name}

    Args:
        servers_path: Path to mcp_servers_joined.json

    Returns:
        Dict mapping tool_id (e.g. "@org/server::tool_name") to OpenAI tool def.
    """
    with open(servers_path) as f:
        servers = json.load(f)

    registry: dict[str, dict] = {}

    for server in servers:
        server_name = server["qualifiedName"]
        tools = server["tools"] if server["tools"] else []

        for tool in tools:
            tool_name = tool["name"]
            tool_id = f"{server_name}::{tool_name}"

            # Build input schema - ensure it's valid JSON Schema
            input_schema = tool["inputSchema"] if tool["inputSchema"] else {}
            if not isinstance(input_schema, dict) or "type" not in input_schema:
                input_schema = {"type": "object", "properties": {}}

            # Remove $schema key if present (not needed for OpenAI format)
            input_schema.pop("$schema", None)

            description = tool.get("description") or f"Tool: {tool_name}"

            # Use normalized name matching evaluator.py convention
            api_name = normalize_tool_name(tool_id)

            registry[tool_id] = {
                "type": "function",
                "function": {
                    "name": api_name,
                    "description": description,
                    "parameters": input_schema,
                },
            }

    return registry


def _synthesize_tool_def(tool_id: str, tool_input: dict) -> dict:
    """Synthesize a minimal tool definition from trajectory data when not found in registry.

    Uses the same normalized naming convention as evaluator.py.
    """
    api_name = normalize_tool_name(tool_id)

    # Infer parameter schema from the actual input
    properties = {}
    for key, value in tool_input.items():
        if isinstance(value, bool):
            properties[key] = {"type": "boolean"}
        elif isinstance(value, int):
            properties[key] = {"type": "integer"}
        elif isinstance(value, float):
            properties[key] = {"type": "number"}
        elif isinstance(value, str):
            properties[key] = {"type": "string"}
        elif isinstance(value, list):
            properties[key] = {"type": "array"}
        elif isinstance(value, dict):
            properties[key] = {"type": "object"}
        else:
            properties[key] = {"type": "string"}

    return {
        "type": "function",
        "function": {
            "name": api_name,
            "description": f"Tool: {api_name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(tool_input.keys()),
            },
        },
    }


def extract_tools_for_task(
    task: dict,
    tool_registry: dict[str, dict],
    tool_mode: ToolMode = "simple",
    random_tool_count: int = 20,
    all_tool_ids: list[str] | None = None,
) -> tuple[list[dict], int, int]:
    """
    Extract OpenAI-format tool definitions for a task based on tool_mode.

    Modes (matching evaluator.py):
        - simple: Only ground truth trajectory tools
        - dag: Ground truth + all tools from the DAG (all_dag_tool_calls)
        - random: Ground truth + random tools from the full registry

    Args:
        task: MCP format task dict
        tool_registry: tool_id -> OpenAI tool def mapping
        tool_mode: One of "simple", "dag", "random"
        random_tool_count: Number of random tools to add in "random" mode
        all_tool_ids: List of all available tool_ids for random mode

    Returns:
        Tuple of (tools list, registry_hits, registry_misses)
    """
    tools = []
    seen_tool_ids: set[str] = set()
    registry_hits = 0
    registry_misses = 0

    # Always include ground truth trajectory tools
    for step in task["ground_truth_trajectory"]:
        tool_id = step["tool_id"]
        if not tool_id or tool_id in seen_tool_ids:
            continue
        seen_tool_ids.add(tool_id)

        if tool_id in tool_registry:
            tool_def = tool_registry[tool_id]
            if validate_tool_name(tool_def["function"]["name"]):
                tools.append(tool_def)
                registry_hits += 1
        else:
            # Synthesize from trajectory data
            tool_def = _synthesize_tool_def(tool_id, step["tool_input"])
            if validate_tool_name(tool_def["function"]["name"]):
                tools.append(tool_def)
                registry_misses += 1

    if tool_mode == "dag":
        # Add all tools from the DAG
        for step in task["all_dag_tool_calls"]:
            tool_id = step["tool_id"]
            if not tool_id or tool_id in seen_tool_ids:
                continue
            seen_tool_ids.add(tool_id)

            if tool_id in tool_registry:
                tool_def = tool_registry[tool_id]
                if validate_tool_name(tool_def["function"]["name"]):
                    tools.append(tool_def)
                    registry_hits += 1
            else:
                tool_def = _synthesize_tool_def(tool_id, step["tool_input"])
                if validate_tool_name(tool_def["function"]["name"]):
                    tools.append(tool_def)
                    registry_misses += 1

    elif tool_mode == "random":
        # Add random tools from the full registry
        assert all_tool_ids is not None, "all_tool_ids required for random mode"

        available = [tid for tid in all_tool_ids if tid not in seen_tool_ids]
        num_to_add = min(random_tool_count, len(available))
        random_tool_ids = random.sample(available, num_to_add)

        for tool_id in random_tool_ids:
            seen_tool_ids.add(tool_id)
            tool_def = tool_registry[tool_id]
            if validate_tool_name(tool_def["function"]["name"]):
                tools.append(tool_def)
                registry_hits += 1

    return tools, registry_hits, registry_misses


def convert_task(
    task: dict,
    index: int,
    tool_registry: dict[str, dict],
    tool_mode: ToolMode = "simple",
    random_tool_count: int = 20,
    all_tool_ids: list[str] | None = None,
) -> tuple[dict, int, int]:
    """
    Convert a single MCP task to training format.

    Args:
        task: MCP format task dict
        index: Unique index for the training sample
        tool_registry: tool_id -> OpenAI tool def mapping
        tool_mode: Tool availability mode
        random_tool_count: Number of random tools for "random" mode
        all_tool_ids: All available tool_ids for "random" mode

    Returns:
        Tuple of (training format dict, registry_hits, registry_misses)
    """
    tools, hits, misses = extract_tools_for_task(task, tool_registry, tool_mode, random_tool_count, all_tool_ids)

    metadata = {
        # Required fields
        "task_description": task["task_description"],
        "tools": tools,
        # answer_schema is already in placeholder template format
        "answer_schema": task["answer_schema"],
        # Evaluation fields
        "expected_answer": task["expected_answer"],
        "ground_truth_trajectory": task["ground_truth_trajectory"],
        "difficulty": task["difficulty"],
    }

    # Optional fields - only include if present
    if "answer_template" in task:
        metadata["answer_template"] = task["answer_template"]

    if "task_id" in task:
        metadata["task_id"] = task["task_id"]

    # Store batch_id from env var and source_request_id for ground truth prioritization
    batch_id = os.environ.get("BATCH_ID", "")
    if batch_id:
        metadata["batch_id"] = batch_id
    if "source_request_id" in task:
        metadata["task_id"] = str(task["source_request_id"])

    return {"index": index, "metadata": metadata}, hits, misses


def main():
    parser = argparse.ArgumentParser(description="Convert MCP task format to training data format")
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Path to MCP tasks JSON file (e.g. tasks.json)",
    )
    parser.add_argument(
        "--servers",
        type=str,
        default="/tmp/instance_storage/mcp_servers_joined.json",
        help="Path to mcp_servers_joined.json for tool definitions",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for training data (.jsonl)",
    )
    parser.add_argument(
        "--tool-mode",
        type=str,
        choices=["simple", "random", "dag"],
        default="simple",
        help="Tool availability mode: simple (ground truth only), random (ground truth + random), dag (all DAG tools)",
    )
    parser.add_argument(
        "--random-tool-count",
        type=int,
        default=20,
        help="Number of random tools to add in 'random' mode (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 'random' mode (default: 42)",
    )
    parser.add_argument(
        "--filter-success",
        action="store_true",
        default=True,
        help="Only include tasks with status='success' (default: True)",
    )
    parser.add_argument(
        "--no-filter-success",
        action="store_true",
        help="Include all tasks regardless of status",
    )
    args = parser.parse_args()

    filter_success = not args.no_filter_success

    # Load tool registry
    print(f"Loading tool registry from {args.servers}...")
    tool_registry = load_tool_registry(args.servers)
    print(f"  Loaded {len(tool_registry)} tool definitions")

    # Load tasks
    print(f"Loading tasks from {args.tasks}...")
    with open(args.tasks) as f:
        tasks = json.load(f)
    print(f"  Loaded {len(tasks)} tasks")

    # Print mode info
    print(f"  Tool mode: {args.tool_mode}", end="")
    if args.tool_mode == "random":
        print(f" (+{args.random_tool_count} random tools, seed={args.seed})")
        random.seed(args.seed)
    else:
        print()

    # Precompute all valid tool_ids for random mode
    all_tool_ids: list[str] | None = None
    if args.tool_mode == "random":
        all_tool_ids = [tid for tid, tdef in tool_registry.items() if validate_tool_name(tdef["function"]["name"])]
        print(f"  Available tools for random sampling: {len(all_tool_ids)}")

    # Convert
    converted = []
    skipped_status = 0
    skipped_no_tools = 0
    skipped_no_trajectory = 0
    total_registry_hits = 0
    total_registry_misses = 0

    for _i, task in enumerate(tasks):
        # Filter by status
        if filter_success and task["status"] != "success":
            skipped_status += 1
            continue

        # Skip tasks with no ground truth trajectory
        if not task["ground_truth_trajectory"]:
            skipped_no_trajectory += 1
            continue

        training_sample, hits, misses = convert_task(
            task,
            len(converted),
            tool_registry,
            tool_mode=args.tool_mode,
            random_tool_count=args.random_tool_count,
            all_tool_ids=all_tool_ids,
        )
        total_registry_hits += hits
        total_registry_misses += misses

        # Skip if no valid tools after filtering
        if not training_sample["metadata"]["tools"]:
            skipped_no_tools += 1
            continue

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
    print(f"  Input tasks:            {len(tasks)}")
    print(f"  Converted:              {len(converted)}")
    print(f"  Skipped (status):       {skipped_status}")
    print(f"  Skipped (no trajectory):{skipped_no_trajectory}")
    print(f"  Skipped (no tools):     {skipped_no_tools}")
    print(f"  Tool registry hits:     {total_registry_hits}")
    print(f"  Tool registry misses:   {total_registry_misses} (synthesized from trajectory)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
