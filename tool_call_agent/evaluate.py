#!/usr/bin/env python3
"""
Evaluator - Run LLMs on tasks using simulated tool calls.

Launches a sglang server with the given checkpoint on the local node,
uses the same RM router (via sglang_registry) as training for the LLM judge
and tool call simulation.

Required environment variables:
    JUDGE_MODEL - model name for the judge LLM served behind the RM router

Usage Examples:
    # Evaluate a checkpoint
    python tool_call_agent/evaluate.py \\
        --checkpoint /tmp/instance_storage/batch_0_200_step_filter_hf/iter_0000059/ \\
        --num-tasks 100 --parallel 20

    # With tool-call-parser and specific batch
    python tool_call_agent/evaluate.py \\
        --checkpoint /tmp/instance_storage/batch_0_200_step_filter_hf/iter_0000059/ \\
        --tool-call-parser qwen --batch 0 --num-tasks 50
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Literal

import ray
from batch_inference import BatchInferenceClient
from sglang.srt.server_args import ServerArgs
from slime.utils.misc import get_current_node_ip

from scripts.sglang_job import RewardSGLangActor
from shared.http_utils import post
from shared.sglang_registry import get_or_create_registry
from tool_call_agent.tool_call_simulator import ToolCallSimulator

# Debug flag - set via --verbose
DEBUG = False

# Tool mode type
ToolMode = Literal["simple", "random", "dag"]


def debug_log(msg: str, data: Any = None):
    """Print debug message if DEBUG is enabled."""
    if not DEBUG:
        return
    print(f"\n{'=' * 60}")
    print(f"[DEBUG] {msg}")
    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        else:
            print(str(data))
    print(f"{'=' * 60}\n")


logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating tool-calling AI agents. "
    "Evaluate both the final answer AND the tool usage. Output only valid JSON."
)

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge evaluating whether a model correctly completed a tool-calling task.

## Task Description
{task_description}

## Expected Answer (Ground Truth)
{expected_answer}

## Model's Answer
{model_answer}

## Ground Truth Tool Trajectory (Expected Tool Calls)
{gt_trajectory_str}

## Model's Tool Trajectory (Actual Tool Calls)
{model_trajectory_str}

## Evaluation Criteria

The model is considered CORRECT if it meets BOTH criteria:

### 1. Answer Correctness
- The model's answer must be semantically equivalent to the expected answer
- Different wording, languages (e.g., Korean vs English, Chinese vs English), or formats are acceptable if they convey the same information
- Minor variations in formatting, units, or phrasing are acceptable
- Different names for the same thing are acceptable (e.g., "Amazon S3" vs "S3" vs "Simple Storage Service")

### 2. Tool Call Correctness
- The model must have made the necessary tool calls to obtain the answer
- Tool calls are considered correct if they use the same tool with similar inputs
- The order of tool calls may differ as long as all necessary information was gathered
- Extra tool calls are acceptable (the model may explore more than necessary)
- MISSING essential tool calls that would be required to obtain the answer is NOT acceptable
- Similar tool inputs are acceptable (e.g., different date formats, minor parameter variations, format differences)
- Missing tool calls (such as didn't use calculator tool when needed) is NOT acceptable

## Response Format
Respond with a JSON object:
{{
    "correct": true/false,
    "answer_correct": true/false,
    "tools_correct": true/false,
    "reasoning": "Brief explanation of your judgment"
}}

Only output the JSON, no other text."""


def _get_rm_router_url() -> str:
    """Get RM router URL from sglang_registry.

    Same logic as training: try rm_router first, fall back to rm_worker.
    """
    registry = get_or_create_registry("sglang_registry")

    router_urls = ray.get(registry.get_all.remote("rm_router"))
    if router_urls:
        logger.info(f"Using RM router: {router_urls[0]}")
        return router_urls[0]

    worker_urls = ray.get(registry.get_all.remote("rm_worker"))
    if worker_urls:
        url = random.choice(worker_urls)
        logger.info(f"No RM router found, using RM worker directly: {url}")
        return url

    raise RuntimeError("No RM router or workers found in sglang_registry")


def compute_pass_at_k(num_correct: list[int], n: int, k: int) -> float:
    """Compute pass@k metric.

    For each task, given n total attempts and c correct attempts,
    pass@k = 1 - C(n-c, k) / C(n, k).

    This is the unbiased estimator from the Codex paper (Chen et al., 2021).

    Args:
        num_correct: list of correct counts per task (one entry per task)
        n: total number of passes per task
        k: k for pass@k

    Returns:
        Average pass@k across all tasks.
    """
    from math import comb

    if k > n:
        return float("nan")

    pass_rates = []
    for c in num_correct:
        if c >= k:
            # If we have at least k correct, guaranteed to pass
            # But use the proper formula for unbiased estimate
            pass_rate = 1.0 - comb(n - c, k) / comb(n, k)
        else:
            pass_rate = 1.0 - comb(n - c, k) / comb(n, k)
        pass_rates.append(pass_rate)

    return sum(pass_rates) / len(pass_rates) if pass_rates else 0.0


class Evaluator:
    """Evaluates tasks by running Claude with simulated tool calls."""

    def __init__(
        self,
        simulator: ToolCallSimulator,
        eval_client: BatchInferenceClient,
        rm_router_url: str,
        judge_model: str,
        semaphore: asyncio.Semaphore,
        max_rounds: int = 10,
        use_llm_judge: bool = True,
        tool_mode: ToolMode = "simple",
        random_tool_count: int = 20,
        num_passes: int = 4,
        task_timeout: float = 7200,
        thinking: bool = False,
        thinking_budget: int = 10000,
    ):
        self.simulator = simulator
        self.eval_client = eval_client  # Client for model being evaluated
        self.rm_router_url = rm_router_url  # RM router URL from sglang_registry
        self.judge_model = judge_model  # Judge model name (from JUDGE_MODEL env var)
        self.semaphore = semaphore  # Shared concurrency limiter across all passes
        self.max_rounds = max_rounds
        self.use_llm_judge = use_llm_judge
        self.tool_mode = tool_mode
        self.random_tool_count = random_tool_count
        self.num_passes = num_passes
        self.task_timeout = task_timeout
        self.thinking = thinking
        self.thinking_budget = thinking_budget

        # Cache all available tool_ids for random mode (set for O(1) lookup)
        self._all_tool_ids: set[str] = set()

    def _get_tools_for_task(self, task: dict) -> list[dict]:
        """Extract tool definitions based on tool_mode.

        Modes:
        - simple: Only ground truth trajectory tools
        - random: Ground truth + random tools from simulator
        - dag: All tools from the DAG
        """
        tools = []
        seen_tool_ids = set()

        # Always include ground truth trajectory tools
        for step in task.get("ground_truth_trajectory", []):
            tool_id = step.get("tool_id", "")
            if tool_id and tool_id not in seen_tool_ids:
                seen_tool_ids.add(tool_id)
                tool_def = self._make_tool_def(tool_id)
                if tool_def:
                    tools.append(tool_def)

        if self.tool_mode == "dag":
            # Add all tools from the DAG
            for step in task.get("all_dag_tool_calls", []):
                tool_id = step.get("tool_id", "")
                if tool_id and tool_id not in seen_tool_ids:
                    seen_tool_ids.add(tool_id)
                    tool_def = self._make_tool_def(tool_id)
                    if tool_def:
                        tools.append(tool_def)

        elif self.tool_mode == "random":
            # Add random tools from simulator's historical tool calls
            # Only use tools that have actually been called (not all tools from server json)
            if not self._all_tool_ids:
                self._all_tool_ids = set(self.simulator.tool_calls.keys())

            # Filter out already included tools
            available = list(self._all_tool_ids - seen_tool_ids)

            # Sample random tools
            num_to_add = min(self.random_tool_count, len(available))
            random_tool_ids = random.sample(available, num_to_add)

            for tool_id in random_tool_ids:
                seen_tool_ids.add(tool_id)
                tool_def = self._make_tool_def(tool_id)
                if tool_def:
                    tools.append(tool_def)

        return tools

    def _make_tool_def(self, tool_id: str) -> dict | None:
        """Create a tool definition from tool_id.

        Uses EXACTLY the same format as batch_callbacks.py and dag_tool_callback.py:
        - name: {normalized_server}__{tool_name}
        - normalized_server: server.replace("@", "").replace("/", "_").replace("-", "_")
        - tool_name: kept AS-IS (NOT normalized)

        The tool name is NOT normalized because:
        1. This matches the historical data format
        2. The Bedrock API allows hyphens in tool names (pattern: ^[a-zA-Z0-9_-]{1,128}$)
        3. Only dots and colons are problematic, but tools with those chars
           are filtered out in _get_tools_for_task when using random mode
        """
        tool_info = self.simulator.tool_info.get(tool_id)
        if not tool_info:
            return None

        # Ensure input_schema is valid JSON Schema
        # Some tools have invalid schemas (e.g., Zod internal format)
        input_schema = tool_info.input_schema
        if not input_schema or not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        elif "type" not in input_schema:
            # Invalid schema - missing required 'type' field
            # This can happen with Zod schemas or other non-JSON-Schema formats
            input_schema = {"type": "object", "properties": {}}

        # Get the normalized name (same as dag_tool_callback.py)
        api_name = self._normalize_tool_name(tool_id)

        # Check if tool name is valid for Bedrock API (^[a-zA-Z0-9_-]{1,128}$)
        # Skip tools with invalid characters (dots, colons, etc.)
        if not re.match(r"^[a-zA-Z0-9_-]{1,128}$", api_name):
            return None

        return {
            "name": api_name,
            "description": tool_info.description or f"Tool: {tool_info.tool_name}",
            "input_schema": input_schema,
        }

    def _normalize_tool_name(self, tool_id: str) -> str:
        """Convert tool_id to normalized tool name used in tool definitions.

        EXACTLY matches the format used in batch_callbacks.py and dag_tool_callback.py:
        - Format: {normalized_server}__{tool_name}
        - Server: normalized (@ / - replaced with _)
        - Tool name: kept AS-IS (NOT normalized)

        This is critical for consistency with historical tool call data.
        The Bedrock API pattern is ^[a-zA-Z0-9_-]{1,128}$ which allows hyphens.
        """
        if "::" in tool_id:
            server_part, tool_name = tool_id.split("::", 1)
        else:
            # Fallback if no :: separator
            server_part = ""
            tool_name = tool_id

        # Normalize server name ONLY (convert @, /, - to _)
        # Tool name is kept as-is to match dag_tool_callback.py
        normalized_server = server_part.replace("@", "").replace("/", "_").replace("-", "_")

        return f"{normalized_server}__{tool_name}"

    def _denormalize_tool_name(self, normalized_name: str, task: dict) -> str:
        """Convert normalized tool name back to tool_id.

        This reverses _normalize_tool_name by looking up the original tool_id.
        """
        # Look up in ground truth trajectory
        for step in task.get("ground_truth_trajectory", []):
            tool_id = step.get("tool_id", "")
            if self._normalize_tool_name(tool_id) == normalized_name:
                return tool_id

        # Look up in all DAG tool calls
        for step in task.get("all_dag_tool_calls", []):
            tool_id = step.get("tool_id", "")
            if self._normalize_tool_name(tool_id) == normalized_name:
                return tool_id

        # Look up in simulator's tool info (for random tools)
        for tool_id in self.simulator.tool_info:
            if self._normalize_tool_name(tool_id) == normalized_name:
                return tool_id

        return normalized_name

    async def evaluate_task(self, task: dict) -> dict[str, Any]:
        """
        Evaluate a single task by running Claude with tool calls.

        Returns evaluation result with:
        - task_description
        - expected_answer
        - model_answer
        - correct (bool)
        - conversation
        - tool_calls made
        """
        task_desc = task.get("task_description", "")
        expected = task.get("expected_answer", {})

        # Get tools for this task
        tools = self._get_tools_for_task(task)

        debug_log("Task", {"description": task_desc, "expected_answer": expected})
        debug_log("Available tools", [t.get("name") for t in tools])

        if not tools:
            return {
                "task_description": task_desc,
                "expected_answer": expected,
                "model_answer": None,
                "correct": False,
                "error": "No tools available for task",
                "conversation": [],
                "tool_calls": [],
            }

        # Build system prompt
        system = """You are an AI assistant that can use tools to answer questions.
Use the available tools to gather information and answer the user's question.
When you have enough information to answer, provide your final answer in JSON format.
You MUST use available tools to answer the question. Do not use your own judgement or knowledge to answer the question.

IMPORTANT: Your final answer MUST be a JSON object that matches the expected schema.
Do not include any explanation or text outside the JSON."""

        # Build initial user message with answer schema
        answer_schema = task.get("answer_schema", {})
        user_content = f"""{task_desc}

Please provide your answer as a JSON object with this schema:
{json.dumps(answer_schema, indent=2)}"""

        messages = [{"role": "user", "content": user_content}]
        conversation = [{"role": "user", "content": user_content}]
        tool_calls_made = []

        # Run conversation loop
        for round_num in range(self.max_rounds):
            debug_log(f"Round {round_num + 1} starting", None)

            # Build request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 40960,
                "messages": messages,
                "system": system,
                "tools": tools,
            }
            if self.thinking:
                if self.thinking_budget == 0:
                    # adaptive thinking (for Opus 4.7+, Sonnet 4.6, Opus 4.6)
                    body["thinking"] = {"type": "adaptive"}
                else:
                    body["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}

            debug_log(
                f"LLM Request (Round {round_num + 1})",
                {
                    "messages": messages[-2:] if len(messages) > 2 else messages,  # Last 2 messages
                    "num_tools": len(tools),
                },
            )

            # Call model via eval_client (the model being evaluated)
            try:
                response = await self.eval_client.invoke_model(
                    body=json.dumps(body),
                    contentType="application/json",
                )
                result = json.loads(response["body"].read())
            except Exception as e:
                debug_log("LLM Error", str(e))
                return {
                    "task_description": task_desc,
                    "expected_answer": expected,
                    "model_answer": None,
                    "correct": False,
                    "error": f"Model invocation failed: {e}",
                    "conversation": conversation,
                    "tool_calls": tool_calls_made,
                }

            # Process response
            content = result.get("content", [])
            stop_reason = result.get("stop_reason", "")

            debug_log(
                f"LLM Response (Round {round_num + 1})",
                {
                    "stop_reason": stop_reason,
                    "content": content,
                },
            )

            # Build assistant message
            assistant_message = {"role": "assistant", "content": content}
            messages.append(assistant_message)
            conversation.append(assistant_message)

            # Check if we have tool use
            tool_use_blocks = [c for c in content if c.get("type") == "tool_use"]

            if not tool_use_blocks or stop_reason == "end_turn":
                # Extract final answer from text
                text_blocks = [c for c in content if c.get("type") == "text"]
                final_text = "\n".join(c.get("text", "") for c in text_blocks)
                debug_log(f"Task completed (stop_reason={stop_reason})", None)

                # Try to parse JSON from response
                model_answer = self._extract_json(final_text)

                debug_log(
                    "Final answer extracted",
                    {
                        "raw_text": final_text[:500],
                        "parsed_answer": model_answer,
                    },
                )

                # Compare answers - check exact match first, then use LLM judge if needed
                correct = self._exact_match(expected, model_answer)
                judge_reasoning = None

                if not correct and self.use_llm_judge:
                    # Only call LLM judge if not an exact match
                    # Pass ground truth trajectory and model's tool calls for evaluation
                    gt_trajectory = task.get("ground_truth_trajectory", [])
                    judge_result = await self._llm_judge(
                        task_desc,
                        expected,
                        model_answer,
                        gt_trajectory,
                        tool_calls_made,
                    )
                    correct = judge_result.get("correct", False)
                    judge_reasoning = judge_result.get("reasoning", "")

                return {
                    "task_description": task_desc,
                    "expected_answer": expected,
                    "model_answer": model_answer,
                    "correct": correct,
                    "judge_reasoning": judge_reasoning,
                    "conversation": conversation,
                    "tool_calls": tool_calls_made,
                    "rounds": round_num + 1,
                }

            # Execute tool calls
            debug_log(f"Executing {len(tool_use_blocks)} tool call(s)", None)
            tool_results = []

            for tool_block in tool_use_blocks:
                tool_id_normalized = tool_block.get("name", "")
                tool_input = tool_block.get("input", {})
                tool_use_id = tool_block.get("id", "")

                # Convert back to original tool_id
                tool_id = self._denormalize_tool_name(tool_id_normalized, task)

                debug_log(f"Tool call: {tool_id}", {"input": tool_input})

                # Simulate tool call (with ground truth prioritization)
                batch_id = task.get("_batch_id", "")
                task_id = str(task.get("source_request_id", ""))
                result = await self.simulator.call_tool(tool_id, tool_input, batch_id=batch_id, task_id=task_id)

                debug_log(
                    f"Tool result: {tool_id}",
                    {
                        "match_type": result.match_type,
                        "is_error": result.is_error,
                        "output_preview": result.tool_output[:300],
                    },
                )

                # Format similar calls for storage
                similar_calls_data = []
                if result.similar_calls:
                    for sc in result.similar_calls[:3]:  # Top 3 similar calls
                        similar_calls_data.append(
                            {
                                "input": sc.tool_input,
                                "output_preview": sc.tool_output[:200] + "..."
                                if len(sc.tool_output) > 200
                                else sc.tool_output,
                                "is_error": sc.is_error,
                            }
                        )
                    debug_log(f"Similar calls for {tool_id}", similar_calls_data)

                tool_calls_made.append(
                    {
                        "name": tool_id_normalized,
                        "arguments": tool_input,
                        "tool_id": tool_id,
                        "tool_output": result.tool_output[:500] + "..."
                        if len(result.tool_output) > 500
                        else result.tool_output,
                        "match_type": result.match_type,
                        "is_error": result.is_error,
                        "similar_calls": similar_calls_data if result.match_type == "generated" else [],
                    }
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result.tool_output,
                        "is_error": result.is_error,
                    }
                )

            # Add tool results to messages
            user_tool_result = {"role": "user", "content": tool_results}
            messages.append(user_tool_result)
            conversation.append(user_tool_result)

        # Max rounds reached
        debug_log("Max rounds reached", None)
        return {
            "task_description": task_desc,
            "expected_answer": expected,
            "model_answer": None,
            "correct": False,
            "error": "Max rounds reached",
            "conversation": conversation,
            "tool_calls": tool_calls_made,
            "rounds": self.max_rounds,
        }

    def _extract_json(self, text: str | None) -> dict | None:
        """Try to extract a JSON object from text.

        Steps:
        1. Remove everything before the last </think>
        2. Try to extract JSON from ```json ... ``` fences
        3. Try to find the last JSON object in the text
        """
        if not text:
            return None

        # Remove everything before the last </think>
        last_idx = text.rfind("</think>")
        if last_idx != -1:
            text = text[last_idx + len("</think>") :]

        # Try to extract from ```json ... ``` fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strip fences and try the whole text
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()
        if not text:
            return None

        # Try to decode as-is
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the last JSON object in the text (last { ... })
        # Search from the end for a valid JSON object
        last_brace = text.rfind("{")
        while last_brace >= 0:
            try:
                return json.loads(text[last_brace:])
            except json.JSONDecodeError:
                last_brace = text.rfind("{", 0, last_brace)

        return None

    @staticmethod
    def _exact_match(expected: dict, actual: dict | None) -> bool:
        """Check if all expected values match actual values (strip + lowercase).

        Matches the training reward logic in generate.py.
        """
        if actual is None:
            return False
        for key, expected_val in expected.items():
            if key not in actual:
                return False
            actual_val = actual[key]
            if str(expected_val).strip().lower() != str(actual_val).strip().lower():
                return False
        return True

    async def _llm_judge(
        self,
        task_description: str,
        expected_answer: dict,
        model_answer: dict | None,
        ground_truth_trajectory: list[dict],
        model_tool_calls: list[dict],
    ) -> dict[str, Any]:
        """Call the sglang RM server as an LLM judge.

        Uses the same RM router as training (via sglang_registry).

        Returns dict with 'correct' (bool) and 'reasoning' (str).
        """
        if model_answer is None:
            return {
                "correct": False,
                "reasoning": "Model did not produce an answer (null response)",
            }

        # Format ground truth trajectory
        gt_trajectory_str = self._format_gt_trajectory(ground_truth_trajectory)

        # Format model's tool calls
        model_trajectory_str = self._format_model_trajectory(model_tool_calls)

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            task_description=task_description,
            expected_answer=json.dumps(expected_answer, indent=2, ensure_ascii=False),
            model_answer=json.dumps(model_answer, indent=2, ensure_ascii=False),
            gt_trajectory_str=gt_trajectory_str,
            model_trajectory_str=model_trajectory_str,
        )

        debug_log(
            "LLM Judge Request",
            {
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "gt_trajectory_count": len(ground_truth_trajectory),
                "model_trajectory_count": len(model_tool_calls),
            },
        )

        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            "max_tokens": 8192,
            "temperature": 0.0,
        }

        try:
            data = await post(f"{self.rm_router_url}/v1/chat/completions", payload)
            judge_text = data["choices"][0]["message"]["content"]

            debug_log("LLM Judge Response", {"raw_text": judge_text})

            judge_result = self._extract_json(judge_text)
            if judge_result and "correct" in judge_result:
                return {
                    "correct": bool(judge_result["correct"]),
                    "reasoning": judge_result.get("reasoning", ""),
                }
            return {
                "correct": False,
                "reasoning": f"Failed to parse judge response: {judge_text[:200]}",
            }
        except Exception as e:
            debug_log("LLM Judge Error", str(e))
            return {
                "correct": False,
                "reasoning": f"Judge invocation failed: {e}",
            }

    @staticmethod
    def _format_gt_trajectory(trajectory: list[dict]) -> str:
        """Format ground truth trajectory for the judge prompt.

        Matches the training format in generate.py: uses tool_name key.
        """
        if not trajectory:
            return "  (No tool calls expected)"
        lines = []
        for i, tc in enumerate(trajectory):
            tool_name = tc["tool_name"] if "tool_name" in tc else "unknown"
            tool_input = json.dumps(tc["tool_input"] if "tool_input" in tc else {}, ensure_ascii=False)
            if len(tool_input) > 200:
                tool_input = tool_input[:200] + "..."
            lines.append(f"  {i + 1}. {tool_name}\n     Input: {tool_input}")
        return "\n".join(lines)

    @staticmethod
    def _format_model_trajectory(tool_calls: list[dict]) -> str:
        """Format model tool calls for the judge prompt.

        Matches the training format in generate.py: uses name/arguments keys.
        """
        if not tool_calls:
            return "  (No tool calls made)"
        lines = []
        for i, tc in enumerate(tool_calls):
            name = tc["name"]
            tool_input = json.dumps(tc["arguments"], ensure_ascii=False)
            if len(tool_input) > 200:
                tool_input = tool_input[:200] + "..."
            lines.append(f"  {i + 1}. {name}\n     Input: {tool_input}")
        return "\n".join(lines)

    async def run(self, tasks: list[dict], label: str = "Evaluating") -> dict[str, Any]:
        """Run multi-pass evaluation on tasks in parallel and return aggregated results.

        All passes run concurrently, sharing a single semaphore for concurrency control.

        Returns dict with:
            - num_tasks, num_passes
            - per_task_correct: list[int] (correct count per task across passes)
            - pass_at_k: dict[str, float] (pass@1 .. pass@N)
            - per_pass_accuracy: list[float]
            - all_pass_results: list[list[dict]]
            - tool_call_stats: dict
        """
        from ray.experimental.tqdm_ray import tqdm as ray_tqdm

        num_passes = self.num_passes
        total_evals = len(tasks) * num_passes
        correct_count = 0
        completed_count = 0
        completed_lock = asyncio.Lock()

        pbar = ray_tqdm(
            total=total_evals,
            desc=label,
            unit="eval",
        )

        async def _eval_one(pass_num: int, task_idx: int, task: dict) -> tuple[int, int, dict]:
            nonlocal correct_count, completed_count
            async with self.semaphore:
                await asyncio.sleep(random.random())
                try:
                    result = await asyncio.wait_for(self.evaluate_task(task), timeout=self.task_timeout)
                except TimeoutError:
                    result = {
                        "task_description": task.get("task_description", ""),
                        "expected_answer": task.get("expected_answer"),
                        "model_answer": None,
                        "correct": False,
                        "error": f"Task timed out after {self.task_timeout}s",
                        "conversation": [],
                        "tool_calls": [],
                    }

            async with completed_lock:
                is_correct = result.get("correct", False)
                if is_correct:
                    correct_count += 1
                completed_count += 1
                pbar.update(1)
                pbar.set_description(f"{label} [correct={correct_count}/{completed_count}]")

                if DEBUG:
                    status = "PASS" if is_correct else "FAIL"
                    task_desc = task["task_description"]
                    print(f"  pass={pass_num + 1} {status}. Task: {task_desc}")
                    if result.get("error"):
                        print(f"    Error: {result['error']}")
                    elif result.get("judge_reasoning"):
                        print(f"    Judge: {result['judge_reasoning']}")

            return pass_num, task_idx, result

        # Launch all (pass, task) pairs as asyncio tasks
        async_tasks = [asyncio.create_task(_eval_one(p, i, t)) for p in range(num_passes) for i, t in enumerate(tasks)]
        raw_results = await asyncio.gather(*async_tasks)
        pbar.close()

        # Organize results: all_pass_results[pass][task_idx] = result
        all_pass_results: list[list[dict]] = [[{}] * len(tasks) for _ in range(num_passes)]
        for pass_num, task_idx, result in raw_results:
            all_pass_results[pass_num][task_idx] = result

        # Compute per-task correct counts
        per_task_correct = [0] * len(tasks)
        for i in range(len(tasks)):
            for p in range(num_passes):
                if all_pass_results[p][i].get("correct"):
                    per_task_correct[i] += 1

        # Per-pass accuracy
        per_pass_accuracy = [
            sum(1 for r in pr if r.get("correct")) / len(tasks) if tasks else 0 for pr in all_pass_results
        ]
        for p, acc in enumerate(per_pass_accuracy):
            print(f"  Pass {p + 1} accuracy: {acc * 100:.1f}%")

        # Aggregate tool call stats
        all_tool_calls = []
        for pr in all_pass_results:
            for r in pr:
                all_tool_calls.extend(r.get("tool_calls", []))

        tool_call_stats = {}
        if all_tool_calls:
            total = len(all_tool_calls)
            tool_call_stats = {
                "total": total,
                "exact_matches": sum(1 for tc in all_tool_calls if tc.get("match_type") == "exact"),
                "generated": sum(1 for tc in all_tool_calls if tc.get("match_type") == "generated"),
                "no_data": sum(1 for tc in all_tool_calls if tc.get("match_type") == "no_data"),
            }

        pass_at_k = {str(k): compute_pass_at_k(per_task_correct, num_passes, k) for k in range(1, num_passes + 1)}

        return {
            "num_tasks": len(tasks),
            "num_passes": num_passes,
            "per_task_correct": per_task_correct,
            "pass_at_k": pass_at_k,
            "per_pass_accuracy": per_pass_accuracy,
            "tool_call_stats": tool_call_stats,
            "all_pass_results": all_pass_results,
        }


def load_tasks(
    data_batches_path: str,
    num_tasks: int,
    seed: int,
    batch: str | None = None,
    min_difficulty: str | None = None,
    task_file_name: str = "valid_tasks.json",
) -> list[dict]:
    """Load random tasks from task files.

    Args:
        data_batches_path: Base path to data batches directory
        num_tasks: Number of tasks to sample
        seed: Random seed for sampling
        batch: Optional specific batch to use (e.g., '0' for data_batches/0/)
        min_difficulty: Minimum difficulty level to include ('easy', 'medium', 'hard')
        task_file_name: Name of the task file to load (default: valid_tasks.json)
    """
    base_path = Path(data_batches_path)

    if batch:
        # Use specific batch
        task_file = base_path / batch / task_file_name
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        task_files = [task_file]
    else:
        # Use all batches (sorted for deterministic ordering)
        task_files = sorted(base_path.glob(f"*/{task_file_name}"))

    all_tasks = []
    for task_file in task_files:
        # Derive batch_id from directory name (e.g. data_batches/3/valid_tasks.json -> "3")
        batch_id = task_file.parent.name
        with open(task_file) as f:
            tasks = json.load(f)
            for t in tasks:
                t["_batch_id"] = batch_id
            all_tasks.extend(tasks)

    batch_info = f"batch {batch}" if batch else f"{len(task_files)} batch(es)"
    print(f"Loaded {len(all_tasks)} tasks from {batch_info}")

    # Filter by minimum difficulty if specified
    if min_difficulty:
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        min_level = difficulty_order.get(min_difficulty.lower(), 0)

        filtered_tasks = []
        for task in all_tasks:
            task_difficulty = task.get("difficulty", "easy").lower()
            task_level = difficulty_order.get(task_difficulty, 0)
            if task_level >= min_level:
                filtered_tasks.append(task)

        print(f"Filtered to {len(filtered_tasks)} tasks with difficulty >= {min_difficulty}")
        all_tasks = filtered_tasks

    # Random sample using isolated RNG so task selection is deterministic
    # regardless of --parallel, --passes, or model behavior
    rng = random.Random(seed)
    if num_tasks < len(all_tasks):
        selected = rng.sample(all_tasks, num_tasks)
    else:
        selected = all_tasks

    return selected


def resolve_checkpoints(checkpoint_path: str) -> list[str]:
    """Resolve checkpoint path to a list of HF checkpoint directories.

    - If path does not look like a filesystem path (e.g. a model ID) -> return as-is
    - If path contains config.json -> single HF checkpoint (base model or converted iter)
    - If path contains iter_* subdirs -> multiple checkpoints, sorted numerically
    - Otherwise -> error
    """
    p = Path(checkpoint_path)

    # Non-path model IDs (e.g. "anthropic.claude-sonnet-4-20250514-v1:0")
    if not p.exists():
        return [checkpoint_path]

    if (p / "config.json").exists():
        return [str(p)]

    iter_dirs = sorted(p.glob("iter_*"))
    iter_dirs = [d for d in iter_dirs if d.is_dir() and (d / "config.json").exists()]
    if iter_dirs:
        return [str(d) for d in iter_dirs]

    raise FileNotFoundError(
        f"No config.json found in {checkpoint_path} and no iter_*/config.json subdirs found. "
        f"Expected an HF checkpoint or a directory containing iter_* checkpoints."
    )


def _make_checkpoint_name(checkpoint: str) -> str:
    """e.g. /tmp/.../batch_hf/iter_0000059/ -> 'batch_hf__iter_0000059'
    For model IDs (non-paths): 'anthropic.claude-sonnet-4-20250514-v1:0' -> as-is with / replaced
    """
    p = Path(checkpoint)
    if p.exists():
        parts = p.resolve().parts
        return "__".join(parts[-2:]).rstrip("_")
    # Model ID: sanitize for filenames
    return checkpoint.replace("/", "__").replace(":", "_")


@ray.remote
class EvalCheckpointActor:
    """Ray actor that runs the full evaluation pipeline for a single checkpoint.

    Each actor runs in its own process with its own event loop, so sglang launch
    and HTTP calls don't block other checkpoints.
    """

    def __init__(
        self,
        checkpoint: str,
        args_dict: dict,
        tasks: list[dict],
        rm_router_url: str,
        judge_model: str,
        simulator_batches_path: str,
        servers_path: str,
        output_dir: str,
        use_llm_judge: bool,
        max_retries: int = 3,
    ):
        self.checkpoint = checkpoint
        self.args_dict = args_dict
        self.tasks = tasks
        self.rm_router_url = rm_router_url
        self.judge_model = judge_model
        self.simulator_batches_path = simulator_batches_path
        self.servers_path = servers_path
        self.checkpoint_name = _make_checkpoint_name(checkpoint)
        if args_dict.get("thinking"):
            self.checkpoint_name += "_thinking"
        self.output_dir = output_dir
        self.use_llm_judge = use_llm_judge
        self.max_retries = max_retries

    async def run(self) -> dict:
        """Run full evaluation: launch sglang, init clients, evaluate, shutdown.

        If checkpoint is not a local path (external model ID), skips sglang and
        uses debug_mode (direct Bedrock) or litellm_mode.

        Retries from scratch if sglang launch fails, up to max_retries times.
        """
        import httpx
        import slime.utils.http_utils as _http_mod

        # Suppress noisy logs in this worker process
        logging.getLogger("sglang").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)

        # Init HTTP client for slime.utils.http_utils.post()
        if _http_mod._http_client is None:
            _http_mod._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=100),
                timeout=httpx.Timeout(timeout=300),
            )

        args = argparse.Namespace(**self.args_dict)

        # Determine if this is an external model (not a local checkpoint path)
        is_external_model = not Path(self.checkpoint).exists()

        if is_external_model:
            return await self._run_external_model(args)
        else:
            return await self._run_sglang_model(args)

    async def _run_external_model(self, args) -> dict:
        """Run evaluation using an external model (Bedrock debug_mode or litellm).

        AWS credentials/profile come from the standard credential chain
        (AWS_PROFILE env var, ~/.aws/config, instance metadata, etc.).
        """
        print(f"[{self.checkpoint_name}] Using external model: {self.checkpoint}")

        # Init eval client - use debug_mode for direct Bedrock invoke_model
        eval_client = BatchInferenceClient()
        await eval_client.setup(
            model_id=self.checkpoint,
            region=getattr(args, "region", "us-west-2"),
            debug_mode=True,
        )

        # Init judge client (still uses RM router)
        judge_client = BatchInferenceClient()
        await judge_client.setup(
            model_id="not-used",
            litellm_mode=True,
            litellm_model=f"hosted_vllm/{self.judge_model}",
            litellm_api_base=f"{self.rm_router_url}/v1",
        )

        # Init simulator
        simulator = await ToolCallSimulator.create(
            data_batches_path=self.simulator_batches_path,
            servers_path=self.servers_path,
            llm_client=judge_client,
        )
        if args.long_output_threshold is not None:
            simulator.LONG_OUTPUT_THRESHOLD = args.long_output_threshold

        use_llm_judge = self.use_llm_judge
        semaphore = asyncio.Semaphore(args.parallel)
        evaluator = Evaluator(
            simulator=simulator,
            eval_client=eval_client,
            rm_router_url=self.rm_router_url,
            judge_model=self.judge_model,
            semaphore=semaphore,
            max_rounds=args.max_rounds,
            use_llm_judge=use_llm_judge,
            tool_mode=args.tool_mode,
            random_tool_count=args.random_tool_count,
            num_passes=args.passes,
            task_timeout=args.task_timeout,
            thinking=args.thinking,
            thinking_budget=args.thinking_budget,
        )

        eval_result = await evaluator.run(self.tasks, label=self.checkpoint_name)

        await eval_client.close()
        await judge_client.close()

        result = {
            "checkpoint": self.checkpoint,
            "checkpoint_name": self.checkpoint_name,
            **eval_result,
        }

        _save_result(result, Path(self.output_dir), args, self.rm_router_url, self.judge_model, use_llm_judge)
        _print_checkpoint_summary(result)

        return result

    async def _run_sglang_model(self, args) -> dict:
        """Run evaluation by launching a local sglang server for the checkpoint."""
        node_ip = get_current_node_ip()

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            eval_sglang = None
            try:
                if attempt > 1:
                    print(f"[{self.checkpoint_name}] Retry {attempt}/{self.max_retries}...")
                print(f"[{self.checkpoint_name}] Launching sglang (may wait for GPUs)...")

                sglang_cli_args = [
                    "--model-path",
                    self.checkpoint,
                    "--tool-call-parser",
                    args.tool_call_parser,
                    "--tp",
                    str(int(args.num_gpus)),
                    "--log-level",
                    "warning",
                ]
                sglang_parser = argparse.ArgumentParser()
                ServerArgs.add_cli_args(sglang_parser)
                sglang_ns = sglang_parser.parse_args(sglang_cli_args)

                actor_options = dict(
                    num_gpus=args.num_gpus,
                    num_cpus=1,
                )
                if not args.no_pin_node:
                    actor_options["resources"] = {f"node:{node_ip}": 0.001}

                eval_sglang = RewardSGLangActor.options(
                    **actor_options,
                ).remote(args=sglang_ns, registry_name="sglang_registry")

                eval_server_url = await eval_sglang.start.remote()
                print(f"[{self.checkpoint_name}] Eval server ready: {eval_server_url}")

                # Init eval client
                eval_client = BatchInferenceClient()
                await eval_client.setup(
                    model_id="not-used",
                    litellm_mode=True,
                    litellm_model=f"hosted_vllm/{self.checkpoint}",
                    litellm_api_base=f"{eval_server_url}/v1",
                )

                # Init judge client
                judge_client = BatchInferenceClient()
                await judge_client.setup(
                    model_id="not-used",
                    litellm_mode=True,
                    litellm_model=f"hosted_vllm/{self.judge_model}",
                    litellm_api_base=f"{self.rm_router_url}/v1",
                )

                # Init simulator
                simulator = await ToolCallSimulator.create(
                    data_batches_path=self.simulator_batches_path,
                    servers_path=self.servers_path,
                    llm_client=judge_client,
                )
                if args.long_output_threshold is not None:
                    simulator.LONG_OUTPUT_THRESHOLD = args.long_output_threshold

                use_llm_judge = self.use_llm_judge
                semaphore = asyncio.Semaphore(args.parallel)
                evaluator = Evaluator(
                    simulator=simulator,
                    eval_client=eval_client,
                    rm_router_url=self.rm_router_url,
                    judge_model=self.judge_model,
                    semaphore=semaphore,
                    max_rounds=args.max_rounds,
                    use_llm_judge=use_llm_judge,
                    tool_mode=args.tool_mode,
                    random_tool_count=args.random_tool_count,
                    num_passes=args.passes,
                    task_timeout=args.task_timeout,
                    thinking=args.thinking,
                    thinking_budget=args.thinking_budget,
                )

                eval_result = await evaluator.run(self.tasks, label=self.checkpoint_name)

                await eval_client.close()
                await judge_client.close()

                result = {
                    "checkpoint": self.checkpoint,
                    "checkpoint_name": self.checkpoint_name,
                    **eval_result,
                }

                # Save result immediately
                _save_result(result, Path(self.output_dir), args, self.rm_router_url, self.judge_model, use_llm_judge)
                _print_checkpoint_summary(result)

                return result

            except Exception as e:
                last_error = e
                print(f"[{self.checkpoint_name}] Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    print(f"[{self.checkpoint_name}] Will retry...")
                    await asyncio.sleep(5)
            finally:
                if eval_sglang is not None:
                    try:
                        print(f"[{self.checkpoint_name}] Shutting down sglang server...")
                        ray.kill(eval_sglang)
                        print(f"[{self.checkpoint_name}] Server stopped.")
                    except Exception:
                        pass

        raise RuntimeError(f"[{self.checkpoint_name}] All {self.max_retries} attempts failed. Last error: {last_error}")


def _print_checkpoint_summary(result: dict):
    """Print summary for a single checkpoint evaluation."""
    name = result["checkpoint_name"]
    print(f"\n  {name}:")
    for k, score in result["pass_at_k"].items():
        print(f"    pass@{k}: {score * 100:.1f}%")
    accs = result["per_pass_accuracy"]
    avg_acc = sum(accs) / len(accs) if accs else 0
    print(f"    avg accuracy: {avg_acc * 100:.1f}%")


def _save_result(
    result: dict,
    output_dir: Path,
    args,
    rm_router_url: str,
    judge_model: str,
    use_llm_judge: bool,
):
    """Save a single checkpoint's result to a JSON file."""
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = result["checkpoint_name"]
    output_file = output_dir / f"eval_{checkpoint_name}_{timestamp}.json"

    output_data = {
        "metadata": {
            "checkpoint": result["checkpoint"],
            "checkpoint_name": checkpoint_name,
            "timestamp": timestamp,
            "seed": args.seed,
            "parallel": args.parallel,
            "max_rounds": args.max_rounds,
            "tool_mode": args.tool_mode,
            "random_tool_count": args.random_tool_count,
            "tool_call_parser": args.tool_call_parser,
            "num_gpus": args.num_gpus,
            "data_batches": args.data_batches,
            "batch": args.batch,
            "min_difficulty": args.min_difficulty,
            "long_output_threshold": args.long_output_threshold,
            "use_llm_judge": use_llm_judge,
            "judge_model": judge_model,
            "rm_router_url": rm_router_url,
        },
        **{k: v for k, v in result.items() if k not in ("checkpoint", "checkpoint_name")},
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"  Results saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate tasks with simulated tools")

    # Checkpoint (the model to evaluate)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to HF checkpoint directory, or an external model ID. "
            "For Bedrock: use the model ID directly (e.g. anthropic.claude-sonnet-4-20250514-v1:0). "
            "For local: path to HF checkpoint or directory containing iter_* subdirs."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=1,
        help="Number of GPUs for the sglang eval server (ignored for external models)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-west-2",
        help="AWS region for Bedrock (default: us-west-2, only used for external models)",
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=os.environ.get("TOOL_CALL_PARSER", "qwen"),
        help="Tool call parser for sglang (default: $TOOL_CALL_PARSER or 'qwen')",
    )

    # Task selection
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-batches",
        type=str,
        default="/tmp/instance_storage/data_batches",
        help="Path to data batches directory",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Specific batch directory to use (e.g., '0' for data_batches/0/). If not set, uses all batches.",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default="valid_tasks.json",
        help="Name of the task file to load from each batch directory (default: valid_tasks.json)",
    )
    parser.add_argument(
        "--servers",
        type=str,
        default="/tmp/instance_storage/mcp_servers_joined.json",
        help="Path to servers JSON",
    )

    parser.add_argument("--max-rounds", type=int, default=10, help="Max conversation rounds")
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=7200,
        help="Per-task timeout in seconds (default: 7200 = 2 hours)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable extended thinking for the eval model",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=10000,
        help="Thinking budget tokens (default: 10000). Set to 0 for adaptive thinking (Opus 4.7+, Sonnet 4.6).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/instance_storage/eval_results",
        help="Output directory for results (a timestamped JSON file is created inside)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=400,
        help="Number of tasks to evaluate in parallel (default: 400)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-as-a-judge (use simple heuristic comparison)",
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
        "--min-difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Minimum task difficulty to include (easy, medium, hard)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--long-output-threshold",
        type=int,
        default=None,
        help="Override ToolCallSimulator.LONG_OUTPUT_THRESHOLD (default: 2000). Set very large to always use free generation.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help="Number of passes per task for pass@k evaluation (default: 4)",
    )
    parser.add_argument(
        "--pin-node",
        dest="no_pin_node",
        action="store_false",
        help="Pin each sglang eval server to the current node's IP (default: off; let Ray schedule on any node)",
    )
    parser.set_defaults(no_pin_node=True)
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only evaluate checkpoints that don't already have results in the output directory",
    )
    args = parser.parse_args()

    # Set global DEBUG flag
    global DEBUG
    DEBUG = args.verbose

    # Suppress noisy sglang and litellm logs
    logging.getLogger("sglang").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    print("=" * 70)
    print("Task Evaluator with Simulated Tool Calls")
    print("=" * 70)

    # Initialize Ray and resolve RM router from sglang_registry
    ray.init(address="auto", namespace="sglang", ignore_reinit_error=True)
    rm_router_url = _get_rm_router_url()
    judge_model = os.environ.get("JUDGE_MODEL", "/data/base_models/Qwen/Qwen3-235B-A22B-Thinking-2507")
    print(f"RM router: {rm_router_url}")
    print(f"Judge model: {judge_model}")

    # Resolve checkpoints
    checkpoints = resolve_checkpoints(args.checkpoint)

    # Filter out checkpoints that already have results
    if args.only_missing:
        output_dir = Path(args.output)
        existing = set()
        if output_dir.exists():
            for f in output_dir.glob("eval_*.json"):
                existing.add(f.name)
        original_count = len(checkpoints)
        checkpoints = [
            ckpt for ckpt in checkpoints if not any(f"eval_{_make_checkpoint_name(ckpt)}_" in name for name in existing)
        ]
        skipped = original_count - len(checkpoints)
        if skipped:
            print(f"Skipping {skipped} checkpoint(s) with existing results in {args.output}")

    print(f"Checkpoints to evaluate: {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f"  {ckpt}")

    # Load tasks (same tasks for all checkpoints)
    tasks = load_tasks(
        args.data_batches,
        args.num_tasks,
        args.seed,
        args.batch,
        args.min_difficulty,
        task_file_name=args.task_file,
    )
    print(f"Selected {len(tasks)} tasks for evaluation\n")

    # Determine which batch directories to load tool calls from
    if args.batch:
        simulator_batches_path = str(Path(args.data_batches) / args.batch)
    else:
        simulator_batches_path = args.data_batches

    use_llm_judge = not args.no_llm_judge
    print(f"LLM-as-a-judge: {'enabled' if use_llm_judge else 'disabled'}")
    print(f"Tool mode: {args.tool_mode}", end="")
    if args.tool_mode == "random":
        print(f" (+{args.random_tool_count} random tools)")
    else:
        print()
    print(f"Parallel tasks per checkpoint: {args.parallel}")
    print(f"Passes per task: {args.passes}")

    # Serialize args for Ray actors (Namespace isn't picklable by default)
    args_dict = vars(args)

    # Launch one EvalCheckpointActor per checkpoint.
    # Each actor runs in its own process: launches sglang, inits clients, runs eval.
    # sglang actors may queue waiting for GPUs; that's expected.
    actors = []
    for ckpt in checkpoints:
        actor = EvalCheckpointActor.options(num_cpus=1).remote(
            checkpoint=ckpt,
            args_dict=args_dict,
            tasks=tasks,
            rm_router_url=rm_router_url,
            judge_model=judge_model,
            simulator_batches_path=simulator_batches_path,
            servers_path=args.servers,
            output_dir=str(Path(args.output)),
            use_llm_judge=use_llm_judge,
        )
        actors.append(actor)

    # Kick off all actors in parallel and collect results, tolerating individual failures
    result_refs = [actor.run.remote() for actor in actors]
    all_results = []
    failed = []
    for i, ref in enumerate(result_refs):
        try:
            result = ray.get(ref)
            all_results.append(result)
        except Exception as e:
            ckpt_name = _make_checkpoint_name(checkpoints[i])
            print(f"\n[ERROR] Checkpoint {ckpt_name} failed: {e}")
            failed.append(ckpt_name)

    # Print summary
    output_dir = Path(args.output)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in all_results:
        _print_checkpoint_summary(result)
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for name in failed:
            print(f"    {name}")
    print(f"\n  {len(all_results)} succeeded, {len(failed)} failed out of {len(checkpoints)} checkpoints")


if __name__ == "__main__":
    main()
