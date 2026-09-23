"""
Custom generate function for tool-calling agent with multi-turn support.
Based on slime's default generate implementation from sglang_rollout.py.

Data Format:
    See DATA_FORMAT.md for the complete data format specification.

    Each sample has:
    - index: Unique sample identifier
    - metadata: Contains task_description, tools, answer_schema, etc.

    The prompt is dynamically generated from metadata.

RULES:
    - NEVER use .get() or getattr() - FAIL FAST on missing keys/attributes
    - Use direct access [] for dicts and . for attributes
    - Use explicit "in" checks only when a field is truly optional
"""

import asyncio
import json
import logging
import os
import re
import traceback
from argparse import Namespace
from typing import Any

import numpy as np
import pybase64
import weave
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from shared.global_counter import counter_scope
from shared.http_utils import post
from shared.rollout_timer import get_sample_timers, rollout_timer_total, set_sample_id
from shared.sample_helpers import (
    USE_FULL_LOGPROBS,
    add_assistant_message,
    add_generation_prompt,
    add_tool_response,
    get_pending_token_count,
    update_rollout_routed_experts,
)
from shared.sglang_registry import get_or_create_registry
from shared.tool_call_parser import create_openai_adapter
from tool_call_agent.tool_workers import perform_tool_call

logger = logging.getLogger(__name__)


# Configuration
MAX_TURNS = 10
MAX_TOKENS = 1024 * 64

# Weave tracing state
_weave_initialized = False


def _init_weave(args: Namespace) -> bool:
    """Initialize weave if wandb is enabled. Returns True if weave is active."""
    global _weave_initialized

    if not args.use_wandb:
        print("[weave] not initialized: use_wandb is False")
        return False

    if _weave_initialized:
        return True

    weave.init(args.wandb_project)
    _weave_initialized = True
    # Mute noisy weave call link logs (🍩 https://wandb.ai/...)
    logging.getLogger("weave.trace.weave_client").setLevel(logging.WARNING)
    print(f"[weave] initialized with project: {args.wandb_project}")
    return True


# Default system prompt for tool-calling agent
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant that can use tools to answer questions.
Use the available tools to gather information and answer the user's question.
When you have enough information, provide your final answer in JSON format.

IMPORTANT: Your final answer MUST be a JSON object matching the expected schema.
Do not include any explanation or text outside the JSON

You should only call ONE TOOL AT A TIME.
"""


def build_prompt(metadata: dict, tokenizer) -> tuple[str, list[dict]]:
    """
    Dynamically build prompt from sample metadata.

    Args:
        metadata: Sample metadata containing task_description, answer_schema, etc.
        tokenizer: Tokenizer with apply_chat_template method.

    Returns:
        Tuple of (formatted prompt string, list of messages).

    Required fields in metadata:
        - task_description: str

    Optional fields:
        - answer_schema: dict (placeholder template format, e.g. {"key": "<description>"})
        - system_prompt: str (if missing, uses DEFAULT_SYSTEM_PROMPT)
    """
    # Required field - will raise KeyError if missing
    task_description = metadata["task_description"]

    # Optional fields - explicit checks
    answer_schema = metadata["answer_schema"] if "answer_schema" in metadata else None
    system_prompt = metadata["system_prompt"] if "system_prompt" in metadata else DEFAULT_SYSTEM_PROMPT

    # Build user message content
    if answer_schema:
        user_content = f"""{task_description}

Please provide your answer as a JSON object with this format:
{json.dumps(answer_schema, indent=2)}"""
    else:
        user_content = task_description

    # Build messages for chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # we add the generation prompt in the generate function
        tools=metadata["tools"],
    )

    return prompt, messages


@weave.op()
async def agent_turn(
    turn: int,
    url: str,
    sample: Sample,
    sampling_params: dict[str, Any],
    args: Namespace,
    state: GenerateState,
    tool_adapter,
) -> dict:
    """
    Execute a single agent turn: call model, then parse and execute tool calls.

    Flow:
    1. Add generation prompt
    2. Call model
    3. Add assistant message
    4. Handle MoE routing replay
    5. Parse tool calls from output, execute them, return results

    Tool responses are NOT added to the sample here — the caller (sample_rollout)
    is responsible for calling add_tool_response before the next turn.

    Args:
        turn: Current turn number
        url: SGLang server URL
        sample: Sample being processed
        sampling_params: Sampling parameters for generation
        args: Runtime arguments
        state: GenerateState with tokenizer
        tool_adapter: Tool call parser adapter

    Returns:
        Dict with keys:
            finish_reason: str
            output_text: str
            tool_results: list[str] — tool output strings (empty if no tool calls)
    """
    # Check token limit including pending tokens
    pending_count = get_pending_token_count(sample)
    total_tokens = len(sample.tokens) + pending_count + sampling_params["max_new_tokens"]
    if total_tokens > MAX_TOKENS:
        sample.status = Sample.Status.TRUNCATED
        sample.reward = 0
        # if hasn't started
        if sample.response_length == 0:
            sample.status = Sample.Status.ABORTED
            sample.metadata["abort_reason"] = "prompt_too_long"
            logger.warning(f"Sample {sample.index} has not started generating yet, WTF?")
            logger.warning(f"sample.tokens: {len(sample.tokens)}")
            # pending count
            logger.warning(f"pending count: {pending_count}")
            # max new tokens
            logger.warning(f"max new tokens: {sampling_params['max_new_tokens']}")
            # total tokens
            logger.warning(f"total tokens: {total_tokens}")
            # max tokens
            logger.warning(f"max tokens: {MAX_TOKENS}")
            # sample.response_length
            logger.warning(f"sample.response_length: {sample.response_length}")
            logger.warning(f"sample.prompt: {sample.prompt}")
            logger.warning("=" * 100)
        # # Set empty routed_experts to avoid NoneType error in training
        # if args.use_rollout_routing_replay:
        #     sample.rollout_routed_experts = np.zeros(
        #         (len(sample.tokens) - 1, args.num_layers, args.moe_router_topk),
        #         dtype=np.int32,
        #     )
        return {
            "finish_reason": "length",
            "output_text": "",
            "tool_results": [],
        }

    # Add generation prompt to pending (will be committed with assistant message)
    add_generation_prompt(sample, state)

    # Build input_ids: committed tokens + pending tokens
    pending_tokens = sample.metadata["pending_tokens"]
    input_ids = sample.tokens + pending_tokens

    # Prepare payload.
    # Full-update mode asks for the whole prefix's logprobs (logprob_start_len=0) directly on the
    # generation request — sglang returns them in meta_info["input_token_logprobs"] from the same
    # prefill, no extra forward pass. Incremental mode leaves logprob_start_len unset (only this
    # turn's generated-token logprobs are needed).
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if USE_FULL_LOGPROBS:
        payload["logprob_start_len"] = 0
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    # Call model
    while True:
        async with counter_scope("assistant_generate"):
            with rollout_timer_total("assistant_turn"):
                output = await post(url, payload)
        output_text = output["text"]
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason != "abort":
            break
        # updating weight
        await asyncio.sleep(1)

    # Extract tokens and log probs
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    # Full-update mode: the prefix's logprobs come back on the generation response itself (we sent
    # logprob_start_len=0). This covers input_ids only — the prefix BEFORE this turn's generation;
    # this turn's generated-token logprobs are new_response_log_probs (output_token_logprobs).
    # Incremental mode (default) leaves this None and keeps per-turn generation logprobs.
    input_log_probs = output["meta_info"].get("input_token_logprobs") if USE_FULL_LOGPROBS else None

    # Add assistant message to sample (trained on). add_assistant_message decides incremental vs
    # full-update based on SLIME_ROLLOUT_FULL_LOGPROBS; input_log_probs is used only in full mode.
    add_assistant_message(
        sample,
        new_response_tokens,
        state,
        log_probs=new_response_log_probs,
        input_log_probs=input_log_probs,
    )

    # Handle MoE routing replay. sglang returns routed experts for the full request every turn;
    # update_rollout_routed_experts appends the new tail (incremental) or overwrites with this
    # turn's full-sequence routing (full-update), per SLIME_ROLLOUT_FULL_ROUTING (independent of
    # the rollout_log_probs mode).
    if "routed_experts" in output["meta_info"]:
        new_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
        update_rollout_routed_experts(sample, new_experts)

    # Parse and execute tool calls from this turn's output
    tool_results: list[str] = []
    if tool_adapter is not None:
        parse_result = tool_adapter.parse_response_to_openai_format(output_text)

        if parse_result["success"]:
            openai_message = parse_result["openai_message"]

            if openai_message.tool_calls and len(openai_message.tool_calls) > 0:
                for tool_call in openai_message.tool_calls:
                    tool_name = tool_call.function["name"]
                    tool_args_str = tool_call.function["arguments"]

                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}

                    with rollout_timer_total("tool_turn"):
                        async with counter_scope("tool_call"):
                            task_id = str(sample.metadata["task_id"]) if "task_id" in sample.metadata else ""
                            result, match_type = await perform_tool_call(tool_name, tool_args, task_id=task_id)

                    # Track tool call in metadata
                    sample.metadata["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "round_index": turn,
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": result,
                            "match_type": match_type,
                        }
                    )

                    tool_results.append(result)

    return {
        "finish_reason": finish_reason,
        "output_text": output_text,
        "tool_results": tool_results,
    }


@weave.op()
async def sample_rollout(
    sample_index: int,
    prompt: str,
    url: str,
    sample: Sample,
    sampling_params: dict[str, Any],
    args: Namespace,
    state: GenerateState,
    tool_adapter,
) -> Sample:
    """
    Execute multi-turn agent rollout for a single sample.

    This function is the parent trace in weave - each agent_turn call becomes a child trace.
    After each turn, it adds tool responses to the sample before the next turn.

    Args:
        sample_index: Unique identifier for this sample
        prompt: Initial prompt text
        url: SGLang server URL
        sample: Sample being processed
        sampling_params: Sampling parameters
        args: Runtime arguments
        state: GenerateState with tokenizer
        tool_adapter: Tool call parser adapter

    Returns:
        Completed sample with status set
    """
    tool_results = None
    for turn in range(MAX_TURNS):
        if tool_results is not None:
            for result in tool_results:
                add_tool_response(sample, result, state)
            tool_results = None
        turn_result = await agent_turn(
            turn=turn,
            url=url,
            sample=sample,
            sampling_params=sampling_params,
            args=args,
            state=state,
            tool_adapter=tool_adapter,
        )

        finish_reason = turn_result["finish_reason"]
        tool_results = turn_result["tool_results"]

        # Check terminal conditions
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            sample.metadata["abort_reason"] = "model_abort"
            break

        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        # No tool calls — generation is complete
        if not tool_results:
            sample.status = Sample.Status.COMPLETED
            break
    else:
        # Exceeded max turns
        sample.status = Sample.Status.TRUNCATED

    # Store the number of turns taken for metrics logging
    sample.metadata["num_turns"] = turn + 1
    return sample


@weave.op()
async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Generate with multi-turn tool calling support.

    Entry point called by external framework. Initializes state and delegates to sample_rollout.
    On any exception, logs the traceback and returns the sample with ABORTED status.
    """
    try:
        assert sample.index is not None, "sample.index is None"
        set_sample_id(sample.index)
        async with counter_scope("rollout"):
            with rollout_timer_total("generate"):
                state = GenerateState(args)
                url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

                assert sample.status == Sample.Status.PENDING, f"Sample status is {sample.status}"

                # Get metadata - required to be a dict
                metadata = sample.metadata
                assert isinstance(metadata, dict), f"sample.metadata must be a dict, got {type(metadata)}"

                # Get tools from metadata (required field) - pass directly in OpenAI format
                tools = metadata["tools"]
                tool_adapter = create_openai_adapter(tools) if tools else None

                # Build prompt dynamically from metadata
                prompt, initial_messages = build_prompt(metadata, state.tokenizer)
                sample.prompt = prompt
                sample.response = ""

                # Initialize messages list for tracking conversation
                metadata["messages"] = list(initial_messages)

                # Initialize tool calls list for tracking tool interactions
                metadata["tool_calls"] = []

                # Initialize prompt tokens
                prompt_ids = state.tokenizer.encode(prompt, add_special_tokens=False)
                sample.tokens = prompt_ids
                sample.response_length = 0
                sample.rollout_log_probs = []
                sample.loss_mask = []

                # Initialize token length tracking
                metadata["prompt_token_length"] = len(prompt_ids)
                metadata["user_token_length"] = 0
                metadata["assistant_token_length"] = 0
                metadata["tool_response_token_length"] = 0

                async with asyncio.timeout(3600):
                    sample = await sample_rollout(
                        sample_index=sample.index,
                        prompt=prompt,
                        url=url,
                        sample=sample,
                        sampling_params=sampling_params,
                        args=args,
                        state=state,
                        tool_adapter=tool_adapter,
                    )
                async with counter_scope("reward_judge"):
                    sample.reward = await reward_func(args, sample)
                if args.use_rollout_routing_replay:
                    assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1, (
                        f"rollout_routed_experts shape mismatch: {sample.rollout_routed_experts.shape[0]} != {len(sample.tokens) - 1}"
                    )
                sample.metadata["timing"] = get_sample_timers()
                return sample
    except TimeoutError:
        logger.warning("generate timeout! sample index: %d", sample.index)
        sample.status = Sample.Status.ABORTED
        sample.metadata["abort_reason"] = "timeout"
        sample.reward = 0
        if args.use_rollout_routing_replay:
            assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1, (
                f"rollout_routed_experts shape mismatch: {sample.rollout_routed_experts.shape[0]} != {len(sample.tokens) - 1}"
            )
        sample.metadata["timing"] = get_sample_timers()
        return sample
    except Exception as exc:
        traceback.print_exc()
        sample.status = Sample.Status.ABORTED
        sample.metadata["abort_reason"] = f"exception:{type(exc).__name__}"
        sample.reward = 0
        if args.use_rollout_routing_replay and sample.rollout_routed_experts.shape[0] != len(sample.tokens) - 1:
            extra_tokens = sample.tokens[sample.rollout_routed_experts.shape[0] :]
            decoded_extra_tokens = state.tokenizer.decode(extra_tokens)
            logger.warning(f"extra tokens: {decoded_extra_tokens}")
            sample.rollout_routed_experts = sample.rollout_routed_experts[: len(sample.tokens) - 1]
        sample.metadata["timing"] = get_sample_timers()
        return sample


# ── LLM-as-a-Judge reward ────────────────────────────────────────────────────

# Cached RM URLs and judge model (resolved once from sglang registry)
_rm_urls: list[str] | None = None
_judge_model: str | None = None

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


@weave.op()
def _extract_json(text: str | None) -> dict | None:
    """Try to extract a JSON object from text.

    Steps:
    1. Remove everything before the last </think>
    2. Strip ```json ... ``` fences
    3. Try to json.loads the result
    """
    if not text:
        return None

    # Remove everything before the last </think>
    last_idx = text.rfind("</think>")
    if last_idx != -1:
        text = text[last_idx + len("</think>") :]

    # Remove ```json ... ``` fences (keep inner content)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    text = text.strip()
    if not text:
        return None

    # Try to decode
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def _format_gt_trajectory(trajectory: list[dict]) -> str:
    """Format ground truth trajectory for the judge prompt."""
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


def _format_model_trajectory(tool_calls: list[dict]) -> str:
    """Format model tool calls for the judge prompt."""
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


async def _get_judge_endpoint(sample_index: int) -> tuple[str, str]:
    """Get the RM router URL and judge model name.

    Now uses rm_router for load balancing instead of direct RM access.
    Falls back to rm_worker list if router is not available.
    """
    global _rm_urls, _judge_model

    if _rm_urls is None:
        registry = get_or_create_registry("sglang_registry")

        # Try to get rm_router first
        router_urls = await registry.get_all.remote("rm_router")
        if router_urls:
            _rm_urls = router_urls
            logger.info(f"Using RM router: {_rm_urls}")
        else:
            # Fallback to direct rm_worker access
            _rm_urls = await registry.get_all.remote("rm_worker")
            if not _rm_urls:
                raise RuntimeError("No RM router or workers found in registry")
            logger.info(f"Using RM workers directly (no router): {_rm_urls}")

        _judge_model = os.environ["JUDGE_MODEL"]
        logger.info(f"Judge endpoints resolved: {_rm_urls}  model={_judge_model}")

    # If using router, always use the first (single router)
    # If using workers directly, distribute by sample_index
    if len(_rm_urls) == 1:
        url = _rm_urls[0]
    else:
        url = _rm_urls[sample_index % len(_rm_urls)]
    return url, _judge_model


@weave.op()
async def _llm_judge(
    task_description: str,
    expected_answer: dict,
    model_answer: dict | None,
    ground_truth_trajectory: list[dict],
    model_tool_calls: list[dict],
    sample_index: int,
) -> dict[str, Any]:
    """Call the sglang RM server as an LLM judge.

    Returns dict with 'correct' (bool), 'reasoning' (str).
    """
    if model_answer is None:
        return {"correct": False, "reasoning": "Model did not produce an answer"}

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_description=task_description,
        expected_answer=json.dumps(expected_answer, indent=2, ensure_ascii=False),
        model_answer=json.dumps(model_answer, indent=2, ensure_ascii=False),
        gt_trajectory_str=_format_gt_trajectory(ground_truth_trajectory),
        model_trajectory_str=_format_model_trajectory(model_tool_calls),
    )

    base_url, model_name = await _get_judge_endpoint(sample_index)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
    }

    data = await post(f"{base_url}/v1/chat/completions", payload)

    judge_text = data["choices"][0]["message"]["content"]
    judge_result = _extract_json(judge_text)

    if judge_result is not None and "correct" in judge_result:
        return {
            "correct": bool(judge_result["correct"]),
            "reasoning": judge_result["reasoning"] if "reasoning" in judge_result else "",
        }

    return {
        "correct": False,
        "reasoning": f"Failed to parse judge response: {judge_text}",
    }


def _exact_match(expected: dict, actual: dict | None) -> bool:
    """Check if all expected values match actual values (strip + lowercase)."""
    if actual is None:
        return False
    for key, expected_val in expected.items():
        if key not in actual:
            return False
        actual_val = actual[key]
        if str(expected_val).strip().lower() != str(actual_val).strip().lower():
            return False
    return True


@weave.op()
async def reward_func(args, sample, **kwargs):
    """Reward function: exact match first, then LLM-as-a-judge fallback.

    Returns:
        float: 1.0 if correct, 0.0 otherwise.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    metadata = sample.metadata

    # Early exit for aborted samples (reward already set to 0)
    if sample.status == Sample.Status.ABORTED:
        return 0.0

    # Extract model's final answer from last assistant message
    messages = metadata["messages"]
    model_answer_text = None
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            model_answer_text = msg["content"]
            break

    model_answer = _extract_json(model_answer_text)

    # Ground truth from metadata (evaluation fields)
    expected_answer = metadata["expected_answer"]

    # Try exact match first — skip LLM judge if it passes
    if _exact_match(expected_answer, model_answer):
        return 1.0

    task_description = metadata["task_description"]
    ground_truth_trajectory = metadata["ground_truth_trajectory"]
    model_tool_calls = metadata["tool_calls"]

    try:
        result = await _llm_judge(
            task_description=task_description,
            expected_answer=expected_answer,
            model_answer=model_answer,
            ground_truth_trajectory=ground_truth_trajectory,
            model_tool_calls=model_tool_calls,
            sample_index=sample.index,
        )
    except Exception:
        traceback.print_exc()
        return 0.0

    return 1.0 if result["correct"] else 0.0
