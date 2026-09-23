#!/usr/bin/env python3
"""
Tool Call Simulator - RAG-based tool call simulation system.

This system loads historical tool calls from trajectory exploration data,
indexes them by tool_id, and can simulate tool calls by:
1. Finding exact matches and returning cached results
2. Finding similar tool calls via fuzzy matching and using LLM to generate simulated results

Usage:
    from batch_inference import BatchInferenceClient
    from tool_call_simulator import ToolCallSimulator

    client = BatchInferenceClient()
    await client.setup(...)

    simulator = await ToolCallSimulator.create(
        data_batches_path="data_batches/",
        servers_path="03_exploration/mcp_servers_joined.json",
        llm_client=client,
    )
    result = await simulator.call_tool("@org/server::tool_name", {"param": "value"})
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from shared.global_counter import counter_scope

# Optional import for batch inference client
try:
    from batch_inference import BatchInferenceClient
except ImportError:
    BatchInferenceClient = None  # type: ignore


@dataclass
class ToolInfo:
    """Information about a tool from MCP servers."""

    tool_id: str
    tool_name: str
    server_name: str
    description: str
    input_schema: dict[str, Any]

    def to_prompt_string(self) -> str:
        """Format tool info for LLM prompt."""
        schema_str = json.dumps(self.input_schema, indent=2)
        return f"""Tool: {self.tool_name}
Server: {self.server_name}
Tool ID: {self.tool_id}
Description: {self.description}
Input Schema:
{schema_str}"""


@dataclass
class ToolCall:
    """Represents a historical tool call."""

    tool_id: str
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    is_error: bool
    input_hash: str = field(default="")
    batch_id: str = field(default="")
    task_id: str = field(default="")

    def __post_init__(self):
        if not self.input_hash:
            self.input_hash = self._hash_input(self.tool_input)

    @staticmethod
    def _hash_input(tool_input: dict[str, Any]) -> str:
        """Create a hash of the tool input for exact matching."""
        # Sort keys for consistent hashing
        normalized = json.dumps(tool_input, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def to_example_string(self) -> str:
        """Format as an example for LLM prompt."""
        input_str = json.dumps(self.tool_input, indent=2)
        output_str = self.tool_output
        status = "ERROR" if self.is_error else "SUCCESS"
        return f"""Input:
{input_str}

Output ({status}):
{output_str}"""


@dataclass
class SimulationResult:
    """Result of a simulated tool call."""

    tool_id: str
    tool_input: dict[str, Any]
    tool_output: str
    is_error: bool
    match_type: str  # "exact", "generated", "no_data"
    confidence: float  # 0.0 to 1.0
    similar_calls: list[ToolCall] = field(default_factory=list)


class ToolCallSimulator:
    """
    RAG-based tool call simulation system.

    Loads historical tool calls and provides simulation via:
    - Exact match: Return cached result if input matches exactly
    - RAG generation: Find similar inputs and use LLM to generate result
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    ):
        """
        Initialize the simulator.

        Args:
            llm_client: BatchInferenceClient instance for LLM calls.
                If None, generation will return errors for non-exact matches.
            model_id: Model ID to use for LLM calls (used if llm_client is in debug mode).
        """
        self.llm_client = llm_client
        self.model_id = model_id

        # tool_id -> list of ToolCall
        self.tool_calls: dict[str, list[ToolCall]] = {}
        # tool_id -> {input_hash -> list[ToolCall]} for exact matching (multiple calls with same input)
        self.exact_match_index: dict[str, dict[str, list[ToolCall]]] = {}
        # tool_id -> ToolInfo
        self.tool_info: dict[str, ToolInfo] = {}

        self.loaded = False

    @classmethod
    async def create(
        cls,
        data_batches_path: str,
        servers_path: str = "03_exploration/mcp_servers_joined.json",
        llm_client: Any | None = None,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    ) -> "ToolCallSimulator":
        """Factory method to create and initialize a simulator."""
        simulator = cls(llm_client=llm_client, model_id=model_id)
        await simulator.load_tool_info(servers_path)
        await simulator.load_trajectories(data_batches_path)
        return simulator

    async def load_tool_info(self, servers_path: str) -> None:
        """Load tool information from MCP servers JSON."""
        print(f"Loading tool info from {servers_path}...")

        with open(servers_path) as f:
            servers = json.load(f)

        for server in servers:
            server_name = server.get("qualifiedName", "")
            tools = server.get("tools", []) or []

            for tool in tools:
                tool_name = tool.get("name", "")
                tool_id = f"{server_name}::{tool_name}"

                self.tool_info[tool_id] = ToolInfo(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    server_name=server_name,
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                )

        print(f"  Loaded info for {len(self.tool_info)} tools")

    async def load_trajectories(self, data_batches_path: str) -> None:
        """Load all trajectory data from data batches directory.

        Supports both:
        - A parent directory with multiple batch subdirs (e.g., data_batches/)
        - A single batch directory (e.g., data_batches/0/)
        """
        base_path = Path(data_batches_path)

        # Check if this is a single batch directory or parent of multiple batches
        direct_file = base_path / "dag_trajectories.json"
        if direct_file.exists():
            # Single batch directory
            trajectory_files = [direct_file]
        else:
            # Parent directory with multiple batches
            trajectory_files = list(base_path.glob("*/dag_trajectories.json"))

        if not trajectory_files:
            raise FileNotFoundError(f"No dag_trajectories.json files found in {base_path}")

        print(f"Loading trajectories from {len(trajectory_files)} batch(es)...")

        total_calls = 0
        for traj_file in trajectory_files:
            calls_loaded = await self._load_trajectory_file(traj_file)
            total_calls += calls_loaded
            print(f"  {traj_file.parent.name}: {calls_loaded} tool calls")

        # Build exact match index (store all calls with same input hash)
        for tool_id, calls in self.tool_calls.items():
            self.exact_match_index[tool_id] = {}
            for call in calls:
                if call.input_hash not in self.exact_match_index[tool_id]:
                    self.exact_match_index[tool_id][call.input_hash] = []
                self.exact_match_index[tool_id][call.input_hash].append(call)

        self.loaded = True
        print(f"\nLoaded {total_calls} total tool calls across {len(self.tool_calls)} unique tools")
        print(f"Unique input combinations: {sum(len(idx) for idx in self.exact_match_index.values())}")

    async def _load_trajectory_file(self, filepath: Path) -> int:
        """Load tool calls from a single trajectory file."""
        with open(filepath) as f:
            trajectories = json.load(f)

        # Extract batch_id from filepath (e.g., data_batches/3/dag_trajectories.json -> "3")
        batch_id = filepath.parent.name

        calls_loaded = 0
        for traj in trajectories:
            # Get task_id from trajectory (request_id)
            task_id = str(traj.get("request_id", ""))

            dag = traj.get("dag", {})
            nodes = dag.get("nodes", {})

            for node in nodes.values():
                tool_id = node.get("tool_id", "")
                if not tool_id:
                    continue

                tool_call = ToolCall(
                    tool_id=tool_id,
                    tool_name=node.get("tool_name", ""),
                    tool_input=node.get("tool_input", {}),
                    tool_output=node.get("tool_output", "") or "",
                    is_error=node.get("is_error", False),
                    batch_id=batch_id,
                    task_id=task_id,
                )

                if tool_id not in self.tool_calls:
                    self.tool_calls[tool_id] = []
                self.tool_calls[tool_id].append(tool_call)
                calls_loaded += 1

        return calls_loaded

    async def call_tool(
        self,
        tool_id: str,
        tool_input: dict[str, Any],
        batch_id: str = "",
        task_id: str = "",
    ) -> SimulationResult:
        """
        Simulate a tool call.

        Args:
            tool_id: The tool identifier (e.g., "@org/server::tool_name")
            tool_input: The input parameters for the tool
            batch_id: Optional batch ID to prioritize ground truth trajectory
            task_id: Optional task ID to prioritize ground truth trajectory

        Returns:
            SimulationResult with the simulated output
        """
        if not self.loaded:
            raise RuntimeError("Simulator not loaded. Call load_trajectories() first.")

        # Check for exact match first
        input_hash = ToolCall._hash_input(tool_input)

        if tool_id in self.exact_match_index and input_hash in self.exact_match_index[tool_id]:
            matching_calls = self.exact_match_index[tool_id][input_hash]

            # If batch_id and task_id provided, prioritize ground truth trajectory
            matched_call = matching_calls[0]  # Default to first
            if batch_id and task_id:
                for call in matching_calls:
                    if call.batch_id == batch_id and call.task_id == task_id:
                        matched_call = call
                        break

            return SimulationResult(
                tool_id=tool_id,
                tool_input=tool_input,
                tool_output=matched_call.tool_output,
                is_error=matched_call.is_error,
                match_type="exact",
                confidence=1.0,
            )

        # Find similar tool calls (with ground truth prioritization)
        similar_calls = await self._find_similar_calls(tool_id, tool_input, top_k=5, batch_id=batch_id, task_id=task_id)

        if not similar_calls:
            # No similar calls found - return error
            return SimulationResult(
                tool_id=tool_id,
                tool_input=tool_input,
                tool_output=json.dumps(
                    {
                        "error": f"No historical data available for tool {tool_id}",
                        "tool_id": tool_id,
                    }
                ),
                is_error=True,
                match_type="no_data",
                confidence=0.0,
            )

        # Generate simulated result using LLM
        generated_output, is_error = await self._generate_simulated_result(tool_id, tool_input, similar_calls)

        return SimulationResult(
            tool_id=tool_id,
            tool_input=tool_input,
            tool_output=generated_output,
            is_error=is_error,
            match_type="generated",
            confidence=0.7,  # Medium confidence for generated results
            similar_calls=similar_calls,
        )

    async def _find_similar_calls(
        self,
        tool_id: str,
        tool_input: dict[str, Any],
        top_k: int = 5,
        batch_id: str = "",
        task_id: str = "",
    ) -> list[ToolCall]:
        """Find the most similar historical tool calls using fuzzy matching.

        If batch_id and task_id are provided, prioritizes tool calls from the
        ground truth trajectory (same batch and task).
        """
        if tool_id not in self.tool_calls:
            return []

        candidates = self.tool_calls[tool_id]
        if not candidates:
            return []

        # Score each candidate
        scored = []
        for call in candidates:
            score = self._compute_similarity(tool_input, call.tool_input)

            # Prioritize ground truth trajectory: add bonus for matching batch_id and task_id
            is_ground_truth = batch_id and task_id and call.batch_id == batch_id and call.task_id == task_id

            scored.append((score, is_ground_truth, call))

        # Sort by: ground truth first, then by score descending
        scored.sort(key=lambda x: (-x[1], -x[0]))

        # Return top_k
        return [call for _, _, call in scored[:top_k]]

    def _compute_similarity(
        self,
        input_a: dict[str, Any],
        input_b: dict[str, Any],
    ) -> float:
        """Compute similarity score between two tool inputs.

        The scoring prioritizes:
        1. Exact matches on keys present in input_a (the query)
        2. Missing keys in input_b that exist in input_a are penalized
        3. Extra keys in input_b that don't exist in input_a are lightly penalized
        4. "Config" keys (format, language, limit, etc.) are weighted less
        """
        keys_a = set(input_a.keys())
        keys_b = set(input_b.keys())

        if not keys_a and not keys_b:
            return 1.0
        if not keys_a:
            # Query has no keys, candidate has keys - slight penalty
            return 0.8
        if not keys_b:
            # Query has keys, candidate has none - big penalty
            return 0.2

        # Focus on keys in the query (input_a)
        matched_keys = keys_a & keys_b
        extra_in_b = keys_b - keys_a  # Keys in candidate but not in query

        if not matched_keys:
            # No overlapping keys - very low score
            return 0.1

        # Config keys are less important for similarity
        config_keys = {
            "format",
            "language",
            "lang",
            "limit",
            "max_results",
            "num_results",
            "offset",
            "page",
            "page_size",
            "sort",
            "order",
            "verbose",
            "debug",
            "output_format",
            "response_format",
            "locale",
            "timezone",
            "tz",
        }

        # Compute weighted value similarity for matched keys
        total_weight = 0.0
        weighted_score = 0.0

        for key in matched_keys:
            val_a = input_a[key]
            val_b = input_b[key]
            sim = self._value_similarity(val_a, val_b)

            # Config keys get lower weight
            key_lower = key.lower()
            if key_lower in config_keys or key_lower.endswith("_format") or key_lower.endswith("_limit"):
                weight = 0.3
            else:
                weight = 1.0

            weighted_score += sim * weight
            total_weight += weight

        avg_value_similarity = weighted_score / total_weight if total_weight > 0 else 0.0

        # Coverage: what fraction of query keys are matched?
        coverage = len(matched_keys) / len(keys_a)

        # Penalty for extra keys in candidate (mild)
        extra_penalty = 0.02 * len(extra_in_b)  # 2% per extra key

        # Final score:
        # - 80% weight on value similarity of matched keys
        # - 20% weight on coverage of query keys
        # - Small penalty for extra keys
        score = 0.8 * avg_value_similarity + 0.2 * coverage - extra_penalty

        return max(0.0, min(1.0, score))

    def _value_similarity(self, val_a: Any, val_b: Any) -> float:
        """Compute similarity between two values using fuzzy matching."""
        if val_a == val_b:
            return 1.0

        if isinstance(val_a, str) and isinstance(val_b, str):
            if not val_a or not val_b:
                return 0.0
            return fuzz.token_sort_ratio(val_a, val_b) / 100.0

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            if val_a == 0 and val_b == 0:
                return 1.0
            max_val = max(abs(val_a), abs(val_b))
            if max_val == 0:
                return 1.0
            diff = abs(val_a - val_b) / max_val
            return max(0.0, 1.0 - diff)

        if isinstance(val_a, list) and isinstance(val_b, list):
            if not val_a and not val_b:
                return 1.0
            if not val_a or not val_b:
                return 0.0
            str_a = json.dumps(val_a, sort_keys=True, default=str)
            str_b = json.dumps(val_b, sort_keys=True, default=str)
            return fuzz.token_sort_ratio(str_a, str_b) / 100.0

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            return self._compute_similarity(val_a, val_b)

        str_a = str(val_a)
        str_b = str(val_b)
        return fuzz.ratio(str_a, str_b) / 100.0

    # Threshold: if any RAG-ed example output is >= this length, use the
    # constrained "pick or error" path instead of free generation.
    LONG_OUTPUT_THRESHOLD = 2000

    async def _generate_simulated_result(
        self,
        tool_id: str,
        tool_input: dict[str, Any],
        similar_calls: list[ToolCall],
    ) -> tuple[str, bool]:
        """
        Generate a simulated result using LLM based on similar historical calls.

        Two modes:
        - Short outputs (all examples < LONG_OUTPUT_THRESHOLD chars):
          LLM may freely generate a new output or reuse an example.
        - Long outputs (any example >= LONG_OUTPUT_THRESHOLD chars):
          LLM must pick one existing example or return an error.
          This prevents hallucinated large payloads for search/retrieval tools.

        Returns:
            Tuple of (output_string, is_error)
        """
        if self.llm_client is None:
            # No LLM client - return the most similar call's output
            best_match = similar_calls[0]
            return best_match.tool_output, best_match.is_error

        has_long = any(len(call.tool_output) >= self.LONG_OUTPUT_THRESHOLD for call in similar_calls)

        if has_long:
            return await self._generate_pick_or_error(tool_id, tool_input, similar_calls)
        else:
            return await self._generate_free(tool_id, tool_input, similar_calls)

    async def _generate_free(
        self,
        tool_id: str,
        tool_input: dict[str, Any],
        similar_calls: list[ToolCall],
    ) -> tuple[str, bool]:
        """Free-generation mode for tools with short outputs.

        The LLM can either:
        1. Return "USE_EXAMPLE:<id>" to use an example's output directly
        2. Generate a new output based on the examples

        Returns:
            Tuple of (output_string, is_error)
        """
        # Get tool info if available
        tool_info = self.tool_info.get(tool_id)
        tool_info_str = (
            tool_info.to_prompt_string()
            if tool_info
            else f"Tool ID: {tool_id}\n(No additional tool information available)"
        )

        # Format similar calls as examples with IDs
        examples_with_ids = []
        for i, call in enumerate(similar_calls[:5]):
            example_id = f"EX{i + 1}"
            example_str = f"[{example_id}]\n{call.to_example_string()}"
            examples_with_ids.append((example_id, example_str, call))

        examples_str = "\n\n---\n\n".join(ex_str for _, ex_str, _ in examples_with_ids)

        # Build the prompt
        prompt = f"""You are simulating a tool call. Based on the tool information and similar historical examples, generate a realistic output for the given input.

## Tool Information

{tool_info_str}

## Historical Examples

The following are real examples of this tool being called with similar inputs. Each example has an ID (e.g., [EX1], [EX2]).

{examples_str}

## New Input to Simulate

Input:
{json.dumps(tool_input, indent=2)}

## Instructions

You have TWO options:

### Option 1: Use an existing example's output
If one of the examples has an input that is VERY similar or identical to the new input (just minor formatting differences), you can reuse that example's output directly.
To do this, respond with ONLY: USE_EXAMPLE:<id>
For example: USE_EXAMPLE:EX1

### Option 2: Generate a new output
If none of the examples are similar enough, generate a realistic output that:
- Follows the same format/structure as the examples
- Is appropriate for the given input
- Is consistent with how this tool behaves

<IMPORTANT>
- Prefer Option 1 (USE_EXAMPLE) when the input is very similar to an example
- You can NOT make up any information. You must only use information from the historical examples
- If no example is similar and you can't generate a valid response, return an error JSON
</IMPORTANT>

Your response (either USE_EXAMPLE:<id> or the raw tool output):"""

        system_prompt = """You are a tool output simulator. Your job is to generate realistic tool outputs based on historical examples.

If an example matches closely, respond with: USE_EXAMPLE:<id>
Otherwise, output ONLY the raw tool response - no explanations, no markdown, no additional text."""

        try:
            async with counter_scope("tool_sim_llm"):
                response = await self.llm_client.invoke_model(
                    body=json.dumps(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 40960,
                            "messages": [{"role": "user", "content": prompt}],
                            "system": system_prompt,
                            "thinking": {"type": "enabled", "budget_tokens": 2048},
                        }
                    ),
                    contentType="application/json",
                )

            result = json.loads(response["body"].read())
            generated_text = self._extract_llm_text(result)

            # Check if LLM wants to use an existing example
            generated_text = generated_text.strip()
            if generated_text.startswith("USE_EXAMPLE:"):
                example_id = generated_text.replace("USE_EXAMPLE:", "").strip()
                for ex_id, _, call in examples_with_ids:
                    if ex_id == example_id:
                        return call.tool_output, call.is_error
                # If example ID not found, fall back to first example
                best_match = similar_calls[0]
                return best_match.tool_output, best_match.is_error

            # Check if generated text looks like an error
            is_error = self._looks_like_error(generated_text)
            return generated_text, is_error

        except Exception as e:
            best_match = similar_calls[0]
            return (
                f"[LLM generation failed: {e}]\n\nFallback to most similar result:\n{best_match.tool_output}",
                best_match.is_error,
            )

    async def _generate_pick_or_error(
        self,
        tool_id: str,
        tool_input: dict[str, Any],
        similar_calls: list[ToolCall],
    ) -> tuple[str, bool]:
        """Constrained mode for tools with long outputs (search, retrieval, etc.).

        The LLM MUST either pick one existing example's output verbatim, or
        declare that none match and return an error.  It is NOT allowed to
        generate/synthesize new long-form content.

        Returns:
            Tuple of (output_string, is_error)
        """
        tool_info = self.tool_info.get(tool_id)
        tool_info_str = (
            tool_info.to_prompt_string()
            if tool_info
            else f"Tool ID: {tool_id}\n(No additional tool information available)"
        )

        # Build compact example summaries (input only -- outputs are too long to
        # include verbatim, but the LLM only needs to decide which input is
        # semantically close enough).
        examples_with_ids: list[tuple[str, str, ToolCall]] = []
        for i, call in enumerate(similar_calls[:5]):
            example_id = f"EX{i + 1}"
            input_str = json.dumps(call.tool_input, indent=2)
            status = "ERROR" if call.is_error else "SUCCESS"
            summary = (
                f"[{example_id}]  (status: {status}, output length: {len(call.tool_output)} chars)\nInput:\n{input_str}"
            )
            examples_with_ids.append((example_id, summary, call))

        examples_str = "\n\n---\n\n".join(s for _, s, _ in examples_with_ids)

        prompt = f"""You are simulating a tool call for a search / retrieval / document tool that returns large outputs.

## Tool Information

{tool_info_str}

## New Input

{json.dumps(tool_input, indent=2)}

## Available Historical Results

Below are real calls to this tool with their inputs. You may ONLY choose one of these results.

{examples_str}

## Instructions

Decide whether one of the examples above is a suitable match for the new input.
A match is suitable when the inputs are semantically equivalent — minor keyword
differences are acceptable (e.g. "s3 object storage" vs "simple storage service",
"Python async" vs "Python asyncio", date format variations, etc.).

Respond with EXACTLY one line in one of these two formats:

  PICK:<id>
  NO_MATCH

Examples of valid responses:
  PICK:EX2
  NO_MATCH

Do NOT output anything else."""

        system_prompt = (
            "You decide whether a historical tool result can be reused for a new "
            "query. Respond with PICK:<id> or NO_MATCH. Nothing else."
        )

        try:
            async with counter_scope("tool_sim_llm"):
                response = await self.llm_client.invoke_model(
                    body=json.dumps(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 256,
                            "messages": [{"role": "user", "content": prompt}],
                            "system": system_prompt,
                            "thinking": {"type": "enabled", "budget_tokens": 2048},
                        }
                    ),
                    contentType="application/json",
                )

            result = json.loads(response["body"].read())
            generated_text = self._extract_llm_text(result).strip()

            # Parse response
            if generated_text.startswith("PICK:"):
                example_id = generated_text.replace("PICK:", "").strip()
                for ex_id, _, call in examples_with_ids:
                    if ex_id == example_id:
                        return call.tool_output, call.is_error
                # Invalid example ID — treat as no match
                return self._no_match_error(tool_id, tool_input), True

            # Anything else (including "NO_MATCH") → error
            return self._no_match_error(tool_id, tool_input), True

        except Exception as e:
            return (
                json.dumps(
                    {
                        "error": f"Tool simulation failed: {e}",
                        "tool_id": tool_id,
                    }
                ),
                True,
            )

    @staticmethod
    def _no_match_error(tool_id: str, tool_input: dict[str, Any]) -> str:
        """Return a structured error for when no historical result matches."""
        return json.dumps(
            {
                "error": "No matching historical result found for this query",
                "tool_id": tool_id,
                "input": tool_input,
            }
        )

    @staticmethod
    def _extract_llm_text(result: dict) -> str:
        """Extract the first non-empty text block from an LLM response."""
        content_blocks = result.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    return text
        # Fallback: concatenate all text blocks
        parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        """Heuristic check if generated text looks like an error response."""
        if not text:
            return False
        lower_text = text.lower()
        if any(err in lower_text for err in ["error", "exception", "failed", "invalid"]):
            try:
                parsed = json.loads(text)
                return "error" in parsed or "Error" in parsed
            except json.JSONDecodeError:
                return text.strip().startswith("{") and "error" in lower_text
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about loaded tool calls."""
        if not self.loaded:
            return {"loaded": False}

        tools_by_count = sorted(
            [(tool_id, len(calls)) for tool_id, calls in self.tool_calls.items()],
            key=lambda x: -x[1],
        )

        return {
            "loaded": True,
            "total_tools": len(self.tool_calls),
            "total_calls": sum(len(calls) for calls in self.tool_calls.values()),
            "unique_inputs": sum(len(idx) for idx in self.exact_match_index.values()),
            "tools_with_info": len(self.tool_info),
            "top_tools": tools_by_count[:20],
            "tools_with_1_call": sum(1 for calls in self.tool_calls.values() if len(calls) == 1),
            "tools_with_10plus_calls": sum(1 for calls in self.tool_calls.values() if len(calls) >= 10),
        }

    def list_tools(self) -> list[str]:
        """List all available tool IDs."""
        return sorted(self.tool_calls.keys())

    def get_tool_examples(self, tool_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get example calls for a specific tool."""
        if tool_id not in self.tool_calls:
            return []

        calls = self.tool_calls[tool_id][:limit]
        return [
            {
                "tool_input": c.tool_input,
                "tool_output": c.tool_output[:500] + "..." if len(c.tool_output) > 500 else c.tool_output,
                "is_error": c.is_error,
            }
            for c in calls
        ]


async def main():
    """Demo usage of the ToolCallSimulator."""
    print("=" * 80)
    print("Tool Call Simulator Demo")
    print("=" * 80)

    # Create simulator without LLM client (will use fallback)
    simulator = await ToolCallSimulator.create(
        data_batches_path="data_batches/",
        servers_path="03_exploration/mcp_servers_joined.json",
        llm_client=None,  # No LLM - will use most similar result as fallback
    )

    # Print stats
    stats = simulator.get_stats()
    print("\nStatistics:")
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Unique inputs: {stats['unique_inputs']}")
    print(f"  Tools with info: {stats['tools_with_info']}")
    print(f"  Tools with 10+ calls: {stats['tools_with_10plus_calls']}")

    print("\nTop 10 tools by call count:")
    for tool_id, count in stats["top_tools"][:10]:
        print(f"  {tool_id}: {count}")

    # Test exact match
    if stats["top_tools"]:
        test_tool = stats["top_tools"][0][0]
        examples = simulator.get_tool_examples(test_tool, limit=1)
        if examples:
            print(f"\n--- Testing exact match for {test_tool} ---")
            test_input = examples[0]["tool_input"]
            result = await simulator.call_tool(test_tool, test_input)
            print(f"Match type: {result.match_type}")
            print(f"Confidence: {result.confidence}")
            print(f"Is error: {result.is_error}")
            print(f"Output preview: {result.tool_output[:200]}...")

            # Test similar match with modified input
            print("\n--- Testing similar match with modified input ---")
            modified_input = dict(test_input)
            if modified_input:
                first_key = next(iter(modified_input.keys()))
                if isinstance(modified_input[first_key], str):
                    modified_input[first_key] = modified_input[first_key] + " modified test query"
                elif isinstance(modified_input[first_key], int):
                    modified_input[first_key] = modified_input[first_key] + 999

            result = await simulator.call_tool(test_tool, modified_input)
            print(f"Match type: {result.match_type}")
            print(f"Confidence: {result.confidence}")
            print(f"Is error: {result.is_error}")
            print(f"Similar calls found: {len(result.similar_calls)}")
            print(f"Output preview: {result.tool_output[:300]}...")


if __name__ == "__main__":
    asyncio.run(main())
