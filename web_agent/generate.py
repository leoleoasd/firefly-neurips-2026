"""
Custom generate function for web agent with multi-turn browser interaction.

Data Format:
    Each sample has:
    - index: Unique sample identifier
    - metadata: Contains WebArena task config (intent, sites, eval, etc.)

Message Flow:
    1. System prompt contains the objective (intent)
    2. First user message is the initial browser observation
    3. Subsequent observations are returned as tool responses

RULES:
    - NEVER use .get() or getattr() - FAIL FAST on missing keys/attributes
    - Use direct access [] for dicts and . for attributes
    - Use explicit "in" checks only when a field is truly optional
"""

import asyncio
import json
import logging
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import weave
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from shared.global_counter import counter_scope
from shared.http_utils import post
from shared.ray_semaphore import acquire_semaphore
from shared.rollout_timer import get_sample_timers, rollout_timer_total, set_sample_id
from shared.sample_helpers import (
    add_assistant_message,
    add_generation_prompt,
    add_system_message,
    add_tool_response,
    get_pending_token_count,
    update_rollout_routed_experts,
)
from shared.tool_call_parser import create_openai_adapter
from web_agent.browser_env import get_browser_pool, get_browser_tool_schema

logger = logging.getLogger(__name__)


# Configuration
MAX_TURNS = 200
MAX_TOKENS = 1024 * 128
FORMAT_PENALTY = -0.05

# Weave tracing state
_weave_initialized = False

# Cached system prompt template
_system_prompt_template: str | None = None


def _get_system_prompt_template() -> str:
    """Load and cache system prompt template."""
    global _system_prompt_template
    if _system_prompt_template is None:
        prompt_path = Path(__file__).parent / "prompts" / "tool_cot.txt"
        _system_prompt_template = prompt_path.read_text()
    return _system_prompt_template


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


def build_initial_prompt(metadata: dict, initial_observation: str, tokenizer) -> tuple[str, list[dict]]:
    """
    Build the initial prompt with system message and first user message.

    Args:
        metadata: Sample metadata containing WebArena task config
        initial_observation: Initial browser observation text
        tokenizer: Tokenizer with apply_chat_template method

    Returns:
        Tuple of (formatted prompt string, list of messages)
    """
    # Get objective from metadata (WebArena 'intent' field)
    objective = metadata["intent"]

    # Build system prompt with objective (using cached template)
    system_prompt = _get_system_prompt_template().format(objective=objective)

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_observation},
    ]

    # Get browser tool schema
    tools = [get_browser_tool_schema()]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
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
    Execute a single agent turn: call model, parse tool call.

    Returns:
        Dict with keys:
            finish_reason: str
            output_text: str
            action: dict | None - parsed browser action (or None if no valid tool call)
            format_error: str | None - error message if format is wrong (triggers penalty)
    """
    # Check token limit
    pending_count = get_pending_token_count(sample)
    total_tokens = len(sample.tokens) + pending_count + sampling_params["max_new_tokens"]

    if total_tokens > MAX_TOKENS:
        sample.status = Sample.Status.TRUNCATED
        sample.reward = 0
        if sample.response_length == 0:
            sample.status = Sample.Status.ABORTED
            logger.warning(f"Sample {sample.index} exceeded token limit before generating")
        return {
            "finish_reason": "length",
            "output_text": "",
            "action": None,
            "format_error": None,
        }

    # Add generation prompt
    add_generation_prompt(sample, state)

    # Build input_ids
    pending_tokens = sample.metadata["pending_tokens"]
    input_ids = sample.tokens + pending_tokens

    # Prepare payload
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
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

    # Add assistant message (trained on)
    add_assistant_message(sample, new_response_tokens, state, log_probs=new_response_log_probs)

    # Handle MoE routing replay. update_rollout_routed_experts appends the new tail (incremental)
    # or overwrites with this turn's full-sequence routing (full-update), per SLIME_ROLLOUT_FULL_ROUTING.
    if "routed_experts" in output["meta_info"]:
        new_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
        update_rollout_routed_experts(sample, new_experts)

    # Parse tool call and check for format errors
    if tool_adapter is None:
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": None,
        }

    parse_result = tool_adapter.parse_response_to_openai_format(output_text)

    # Error 1: Unable to parse tool call from response
    if not parse_result["success"]:
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": "Your response could not be parsed as a tool call. You must call the step_browser tool with a valid action. Use the correct tool call format.",
        }

    openai_message = parse_result["openai_message"]

    # Error 1b: Parsed successfully but no tool calls present
    if not openai_message.tool_calls or len(openai_message.tool_calls) == 0:
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": "Your response did not contain a tool call. You must call the step_browser tool on every turn to interact with the browser.",
        }

    # Error 3: Multiple parallel tool calls
    if len(openai_message.tool_calls) > 1:
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": f"You made {len(openai_message.tool_calls)} parallel tool calls. You must make exactly one step_browser tool call per turn.",
        }

    tool_call = openai_message.tool_calls[0]
    tool_name = tool_call.function["name"]
    tool_args_str = tool_call.function["arguments"]

    # Error 2: Wrong tool name
    if tool_name != "step_browser":
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": f"You called tool '{tool_name}' but the only available tool is 'step_browser'. Use step_browser to interact with the browser.",
        }

    # Parse arguments JSON
    try:
        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
    except json.JSONDecodeError:
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": f"Failed to parse tool call arguments as JSON: {tool_args_str!r}. Provide valid JSON arguments.",
        }

    # Check for required 'action' key
    if "action" not in tool_args:
        return {
            "finish_reason": finish_reason,
            "output_text": output_text,
            "action": None,
            "format_error": f"Tool call arguments missing required 'action' key. Got keys: {list(tool_args.keys())}. You must include an 'action' field.",
        }

    # Valid tool call - track it
    sample.metadata["tool_calls"].append(
        {
            "id": tool_call.id,
            "turn": turn,
            "name": tool_name,
            "arguments": tool_args,
        }
    )

    return {
        "finish_reason": finish_reason,
        "output_text": output_text,
        "action": tool_args,
        "format_error": None,
    }


@weave.op()
async def browser_init(worker, instance_id: str, task_config: dict) -> str:
    """
    Initialize browser environment and get initial observation.

    Wrapped in weave.op() for tracing.
    Uses global semaphore to limit concurrent container launches.

    Args:
        worker: BrowserWorker ray actor handle
        instance_id: Browser instance identifier
        task_config: WebArena task configuration from metadata

    Returns:
        Initial observation as formatted string for LLM
    """
    # Use semaphore to limit concurrent container launches
    with rollout_timer_total("browser_init"):
        async with counter_scope("browser_init_queue"):
            async with acquire_semaphore():
                async with counter_scope("browser_init"):
                    initial_observation = await worker.create_instance.remote(instance_id, task_config)
    return initial_observation


@weave.op()
async def browser_step(worker, instance_id: str, action: dict) -> tuple[str, bool, float]:
    """
    Execute a browser action and return the observation.

    Wrapped in weave.op() for tracing.

    Args:
        worker: BrowserWorker ray actor handle
        instance_id: Browser instance identifier
        action: Action dict with 'action' key and parameters

    Returns:
        Tuple of (formatted_observation, terminated, score)
    """
    with rollout_timer_total("browser_step"):
        async with counter_scope("browser_step"):
            observation, terminated, score = await worker.step.remote(instance_id, action)
    return observation, terminated, score


@weave.op()
async def sample_rollout(
    sample_index: int,
    url: str,
    sample: Sample,
    sampling_params: dict[str, Any],
    args: Namespace,
    state: GenerateState,
    tool_adapter,
    worker,
    instance_id: str,
) -> Sample:
    """
    Execute multi-turn browser agent rollout for a single sample.

    Flow:
        1. Get initial observation (already done, passed via prompt)
        2. Loop:
           a. Model generates tool call
           b. Execute action in browser
           c. Get observation as tool response
           d. Check if terminated

    Format errors (bad tool call, wrong tool, multiple tools, browser exceptions)
    apply a flat FORMAT_PENALTY if any error occurred at any point during the
    rollout. The model receives a system message explaining the error and the
    rollout continues.
    """
    had_format_error = False

    for turn in range(MAX_TURNS):
        # Generate model response
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
        action = turn_result["action"]
        format_error = turn_result["format_error"]

        # Check terminal conditions
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            break

        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        # Format error: mark, tell model what went wrong, continue
        if format_error is not None:
            had_format_error = True
            logger.info(f"Sample {sample_index} turn {turn}: format error: {format_error}")
            add_system_message(sample, format_error, state)
            continue

        # action should not be None here (format_error covers that case)
        assert action is not None

        # Execute action in browser, catching env exceptions
        try:
            observation, terminated, _score = await browser_step(worker, instance_id, action)
        except Exception as e:
            # Browser env exception: mark, tell model, continue
            had_format_error = True
            error_msg = (
                f"Browser action failed with error: {type(e).__name__}: {e}. Check your action format and try again."
            )
            logger.info(f"Sample {sample_index} turn {turn}: browser exception: {e}")
            add_system_message(sample, error_msg, state)
            continue

        # Store the observation as tool response
        add_tool_response(sample, observation, state)

        # Check if task terminated
        if terminated:
            sample.status = Sample.Status.COMPLETED
            # Store model answer if available
            model_answer = await worker.get_model_answer.remote(instance_id)
            sample.metadata["model_answer"] = model_answer
            break
    else:
        # Exceeded max turns
        sample.status = Sample.Status.TRUNCATED

    # Store the number of turns taken and format penalty for metrics
    sample.metadata["num_turns"] = turn + 1
    sample.metadata["format_penalty"] = FORMAT_PENALTY if had_format_error else 0.0
    sample.metadata["had_format_error"] = had_format_error
    return sample


@weave.op()
async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Generate with multi-turn browser interaction.

    Entry point called by slime framework.

    Uses two semaphores:
    - container_running: Limits total concurrent containers (held for entire session)
    - container_launch: Limits concurrent container launches (held only during init)
    """
    # Acquire container_running semaphore for the entire browser session
    async with acquire_semaphore("container_running"):
        worker = None
        instance_id = None
        browser_created = False

        try:
            assert sample.index is not None, "sample.index is None"
            set_sample_id(sample.index)
            async with counter_scope("rollout"):
                with rollout_timer_total("generate"):
                    state = GenerateState(args)
                    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

                    assert sample.status == Sample.Status.PENDING

                    # Initialize weave
                    _init_weave(args)

                    # Get metadata - contains WebArena task config
                    metadata = sample.metadata
                    assert isinstance(metadata, dict)

                    # Get browser worker (prefer local one on same Ray node
                    # as this AsyncRolloutWorkerActor) and create instance
                    pool = get_browser_pool()
                    worker = pool.get_local_worker(sample.index)
                    instance_id = f"sample_{sample.index}"

                    # Create browser environment and get initial observation
                    # (container_launch semaphore is acquired inside browser_init)
                    # If create_instance fails, it cleans up after itself internally.
                    # We only set browser_created=True after success so the finally
                    # block knows whether there is an instance to release.
                    initial_observation = await browser_init(worker, instance_id, metadata)
                    browser_created = True

                    # Build initial prompt with system + first user message
                    prompt, initial_messages = build_initial_prompt(metadata, initial_observation, state.tokenizer)
                    sample.prompt = prompt
                    sample.response = ""

                    # Initialize tracking
                    metadata["messages"] = list(initial_messages)
                    metadata["tool_calls"] = []
                    metadata["model_answer"] = None

                    # Initialize tokens
                    prompt_ids = state.tokenizer.encode(prompt, add_special_tokens=False)
                    sample.tokens = prompt_ids
                    sample.response_length = 0
                    sample.rollout_log_probs = []
                    sample.loss_mask = []

                    # Token length tracking
                    metadata["prompt_token_length"] = len(prompt_ids)
                    metadata["user_token_length"] = 0
                    metadata["assistant_token_length"] = 0
                    metadata["tool_response_token_length"] = 0

                    # Create tool adapter
                    tools = [get_browser_tool_schema()]
                    # strip_thinking=True: web_agent strips <think>…</think> before parsing.
                    tool_adapter = create_openai_adapter(tools, strip_thinking=True)

                    # Run rollout
                    async with asyncio.timeout(1500):
                        sample = await sample_rollout(
                            sample_index=sample.index,
                            url=url,
                            sample=sample,
                            sampling_params=sampling_params,
                            args=args,
                            state=state,
                            tool_adapter=tool_adapter,
                            worker=worker,
                            instance_id=instance_id,
                        )

                    # Compute reward
                    async with counter_scope("reward"):
                        sample.reward = await reward_func(args, sample, worker, instance_id)

                    # Validate routing replay if enabled
                    if args.use_rollout_routing_replay:
                        assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1

                    sample.metadata["timing"] = get_sample_timers()
                    return sample

        except TimeoutError:
            logger.warning("generate timeout! sample index: %d", sample.index)
            sample.status = Sample.Status.ABORTED
            sample.metadata["abort_reason"] = "timeout"
            # Preserve any format penalty accumulated before the timeout
            sample.reward = (
                float(sample.metadata.get("format_penalty", 0.0)) if isinstance(sample.metadata, dict) else 0.0
            )
            sample.metadata["timing"] = get_sample_timers()
            return sample

        except Exception as exc:
            traceback.print_exc()
            sample.status = Sample.Status.ABORTED
            sample.metadata["abort_reason"] = f"exception:{type(exc).__name__}"
            # Preserve any format penalty accumulated before the exception
            sample.reward = (
                float(sample.metadata.get("format_penalty", 0.0)) if isinstance(sample.metadata, dict) else 0.0
            )
            sample.metadata["timing"] = get_sample_timers()
            return sample

        finally:
            # Always release browser instance if it was successfully created.
            # create_instance cleans up after itself on failure, so we only
            # need to release when browser_created is True.
            if browser_created and worker is not None and instance_id is not None:
                try:
                    await worker.release_instance.remote(instance_id)
                except Exception:
                    logger.error(
                        f"Failed to release browser instance {instance_id}, "
                        f"resources may have leaked on the worker node"
                    )


@weave.op()
async def reward_func(args, sample: Sample, worker, instance_id: str) -> float:
    """
    Compute reward for a web agent sample.

    Reward = task_score + format_penalty_total

    Where:
        - task_score: WebArena eval score (typically 0.0 or 1.0)
        - format_penalty_total: negative penalty accumulated during rollout from
          malformed tool calls and browser exceptions (FORMAT_PENALTY per error)

    So a perfectly-formatted successful run scores 1.0, a successful run with
    one format error scores 0.95, a failed run with no format errors scores 0.0,
    a failed run with one format error scores -0.05, etc.
    """
    format_penalty = sample.metadata.get("format_penalty", 0.0) if isinstance(sample.metadata, dict) else 0.0

    if sample.status == Sample.Status.ABORTED:
        # Aborted runs (timeout/exception in generate) don't get browser score;
        # still apply format penalty accumulated up to the abort point.
        return float(format_penalty)

    # Get task score from browser environment
    try:
        task_score = await worker.get_score.remote(instance_id)
        task_score = float(task_score)
    except Exception:
        traceback.print_exc()
        task_score = 0.0

    return task_score + float(format_penalty)
