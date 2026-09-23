"""
Trainable agent for tau2-bench gym env - MoE Version

This module handles multi-turn interactions with token-in/token-out mode
and routed_experts collection for MoE routing replay.

Key differences from agent.py:
- Uses input_ids instead of text for requests (token-in/token-out)
- Collects routed_experts from each turn
- Aggregates routed_experts for the full trajectory
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybase64
import ray
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from tau2.data_model.message import (
    AssistantMessage as Tau2AssistantMessage,
)
from tau2.data_model.message import (
    ToolMessage as Tau2ToolMessage,
)
from tau2.data_model.message import (
    UserMessage as Tau2UserMessage,
)
from tau2.utils.tools import to_functional_format

from shared.global_counter import get_global_counter
from shared.rollout_timer import rollout_timer_total
from shared.sample_helpers import (
    add_assistant_message,
    add_generation_prompt,
    add_tool_response,
    add_user_message,
    get_pending_token_count,
)
from shared.tool_call_parser import parse_tools

try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None

logger = logging.getLogger(__name__)

# Whole-session context limit (prompt + all turns + tool responses + next
# generation). Mirrors tool_call_agent's MAX_TOKENS guard. Override via env.
MAX_SESSION_TOKENS = int(os.environ.get("TAU2_MAX_SESSION_TOKENS", str(32 * 1024)))

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def system_prompt(domain_policy: str) -> str:
    return SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)


@weave.op()
def tau2_observation_to_messages(observation: list, incremental: bool = True) -> list[dict[str, Any]]:
    """
    Convert tau2 gym observation (list of Message objects) directly to chat messages.

    Reads structured Message objects from env._agent.observation instead of
    parsing the lossy string representation returned by env.step()/env.reset().

    Args:
        observation: list of tau2 Message objects from env._agent.observation
        incremental: if True, only return messages after the last AssistantMessage.
                     This mirrors the behaviour of AgentGymEnv._format_observation
                     with all_messages_as_observation=False (default), which resets
                     the output after every assistant turn so the caller only sees
                     new tool / user messages.

    Returns:
        list of chat message dicts compatible with tokenizer.apply_chat_template
    """

    msgs = list(observation)

    if incremental and msgs:
        # Find the last AssistantMessage and return only what follows
        last_assistant_idx = -1
        for i, m in enumerate(msgs):
            if isinstance(m, Tau2AssistantMessage):
                last_assistant_idx = i
        if last_assistant_idx >= 0:
            msgs = msgs[last_assistant_idx + 1 :]

    result: list[dict[str, Any]] = []
    for m in msgs:
        if isinstance(m, Tau2UserMessage):
            if m.tool_calls:
                # User-side tool call - convert to functional text representation

                tc_str = ", ".join(to_functional_format(tc) for tc in m.tool_calls)
                result.append({"role": "user", "content": tc_str})
            elif m.content is not None:
                result.append({"role": "user", "content": m.content})
        elif isinstance(m, Tau2ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": m.content or "",
                    "tool_call_id": m.id or "0",
                }
            )
        elif isinstance(m, Tau2AssistantMessage):
            # Should rarely appear when incremental=True

            if m.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id or "0",
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in m.tool_calls
                ]
                result.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            else:
                result.append({"role": "assistant", "content": m.content or ""})
        else:
            # SystemMessage or other
            result.append(
                {
                    "role": getattr(m, "role", "system"),
                    "content": getattr(m, "content", "") or "",
                }
            )

    return result


def parsed_response_to_action_str(parsed: dict[str, Any]) -> str:
    """
    Convert sglang parsed result to tau2 step() action string.
    """
    from tau2.data_model.message import ToolCall

    normal_text = parsed.get("normal_text") or ""
    calls = parsed.get("calls") or []
    if not calls:
        return normal_text.strip() if normal_text else ""
    if len(calls) > 1:
        logger.debug("Multiple tool calls, using first only.")
    call = calls[0]
    name = call.get("name", "")
    params = call.get("parameters")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    tc = ToolCall(name=name, arguments=params, requestor="assistant")
    return to_functional_format(tc)


def decode_routed_experts(routed_experts_data, num_layers: int, moe_router_topk: int) -> np.ndarray:
    """
    Decode routed_experts from sglang response.

    Supports both formats:
    - List of lists: [[expert_ids...], ...] - newer sglang format
    - Base64 string: encoded int32 array - older format

    The routed_experts is full sequence (not incremental), so we infer
    the number of tokens from the data size.

    Args:
        routed_experts_data: Either a list of lists or base64-encoded string
        num_layers: Number of model layers
        moe_router_topk: Top-k for MoE router

    Returns:
        np.ndarray of shape (num_tokens, num_layers, moe_router_topk)
    """
    if isinstance(routed_experts_data, str):
        # Base64-encoded format
        routed_bytes = pybase64.b64decode(routed_experts_data.encode("ascii"))
        routed_flat = np.frombuffer(routed_bytes, dtype=np.int32)
        routed_experts = routed_flat.reshape(-1, num_layers, moe_router_topk)
    else:
        # List format from newer sglang
        # Shape should already be (num_tokens, num_layers, moe_router_topk)
        routed_experts = np.array(routed_experts_data, dtype=np.int32)
        if routed_experts.ndim == 2:
            # If 2D, reshape assuming it's (num_tokens * num_layers, moe_router_topk)
            routed_experts = routed_experts.reshape(-1, num_layers, moe_router_topk)
    return routed_experts


@dataclass
class TurnResult:
    """Result from a single turn of the agent-environment loop."""

    response_text: str
    new_token_ids: list[int]
    new_log_probs: list[float]
    routed_experts: np.ndarray | None
    routed_experts_token_count: int
    terminated: bool
    truncated: bool
    aborted: bool
    parse_failed: bool
    called_tool_signature: str
    called_tool_name: str
    reward: float
    obs: str
    step_info: dict[str, Any]


def final_reward_with_penalties(
    base_reward: float,
    had_tool_parse_failure: bool,
    assistant_token_length: int,
    consecutive_same_tool_count: int,
    had_successful_find_name_by_tool_call: bool,
) -> tuple[float, dict[str, Any]]:
    """
    Apply optional reward penalties (env vars).

    TAU2_TOOL_FORMAT_PENALTY: float added to reward when any turn failed tool-call
        parsing (typically negative, e.g. -0.1). Default 0.
    TAU2_MAX_ASSISTANT_TOKENS: if >0, penalize when total assistant output tokens
        exceed this (uses metadata assistant_token_length).
    TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN: float multiplier on
        max(0, assistant_tokens - max); default -0.0001 when max is set.
    TAU2_CONSECUTIVE_SAME_TOOL_PENALTY: float penalty for each consecutive
        repeated tool call pair (same name + same args).
        Example sequence A(x),A(x),B(y),B(y),B(y) => 3 repeats.
    TAU2_CONSECUTIVE_SAME_TOOL_NAME_PENALTY: float penalty for each consecutive
        repeated tool name call (same name, regardless of args).
        Example sequence A(x),A(y),B(z),B(w) => 2 repeats.
    Also applies a fixed -0.5 penalty when no successful `find_name_by_...`
    tool call is made in the trajectory.
    """
    tool_penalty = float(os.environ.get("TAU2_TOOL_FORMAT_PENALTY", "0"))
    max_assistant = int(os.environ.get("TAU2_MAX_ASSISTANT_TOKENS", "0"))
    per_excess = float(os.environ.get("TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN", "-0.0001"))
    consecutive_same_tool_penalty = float(os.environ.get("TAU2_CONSECUTIVE_SAME_TOOL_PENALTY", "0"))

    info: dict[str, Any] = {"base_reward": base_reward}
    reward = base_reward

    if had_tool_parse_failure and tool_penalty != 0.0:
        reward += tool_penalty
        info["tool_format_penalty"] = tool_penalty

    if max_assistant > 0 and assistant_token_length > max_assistant:
        excess = assistant_token_length - max_assistant
        length_penalty = per_excess * excess
        reward += length_penalty
        info["length_penalty"] = length_penalty
        info["assistant_token_length"] = assistant_token_length
        info["max_assistant_tokens"] = max_assistant
        info["length_excess_tokens"] = excess

    if consecutive_same_tool_count > 0 and consecutive_same_tool_penalty != 0.0:
        repeat_penalty_total = consecutive_same_tool_penalty * consecutive_same_tool_count
        reward += repeat_penalty_total
        info["consecutive_same_tool_penalty_total"] = repeat_penalty_total
        info["consecutive_same_tool_count"] = consecutive_same_tool_count

    if not had_successful_find_name_by_tool_call:
        missing_find_name_by_penalty = -0.5
        reward += missing_find_name_by_penalty
        info["missing_find_name_by_penalty"] = missing_find_name_by_penalty

    info["final_reward"] = reward
    return reward, info


@weave.op
async def run_single_turn_async(
    env,
    url: str,
    sample: Sample,
    state: GenerateState,
    sampling_params: dict[str, Any],
    tools_info: list[dict],
    policy: str,
    turn_idx: int,
    num_layers: int,
    moe_router_topk: int,
    return_routed_experts: bool = False,
) -> TurnResult:
    """
    Execute a single turn of the agent-environment loop.

    This function:
    1. Sends the current token sequence to sglang for generation
    2. Parses the response and extracts the action
    3. Executes the action in the environment
    4. Returns the result without modifying the sample (caller handles that)

    Args:
        env: The tau2 gym environment
        url: The sglang server URL
        sample: The Sample object (used for reading current tokens)
        state: GenerateState containing tokenizer
        sampling_params: Sampling parameters for generation
        tools_info: Tool schemas for parsing
        policy: The policy string
        turn_idx: Current turn index (for logging)
        num_layers: Number of MoE layers
        moe_router_topk: Top-k for MoE routing
        return_routed_experts: Whether to collect routed_experts

    Returns:
        TurnResult with response data and environment step result
    """
    loop = asyncio.get_event_loop()

    # Build input_ids: committed tokens + pending tokens
    pending_tokens = sample.metadata["pending_tokens"]
    input_ids = sample.tokens + pending_tokens

    # Build payload with input_ids (token-in/token-out mode)
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if return_routed_experts:
        payload["return_routed_experts"] = True

    logger.info(
        f"Turn {turn_idx}: Sending request to sglang, input_ids_length={len(input_ids)}, return_routed_experts={return_routed_experts}"
    )
    with rollout_timer_total("assistant_turn"):
        output = await post(url, payload)

    # Check for abort
    if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
        logger.warning(f"Turn {turn_idx}: Request aborted by sglang")
        return TurnResult(
            response_text="",
            new_token_ids=[],
            new_log_probs=[],
            routed_experts=None,
            routed_experts_token_count=0,
            terminated=False,
            truncated=False,
            aborted=True,
            parse_failed=False,
            called_tool_signature="",
            called_tool_name="",
            reward=0.0,
            obs="",
            step_info={},
        )

    # Get new tokens and log probs from output
    meta_info = output.get("meta_info", {})
    if return_routed_experts:
        logger.info(
            f"Turn {turn_idx}: Response meta_info keys: {list(meta_info.keys())}, has routed_experts: {'routed_experts' in meta_info}"
        )

    if "output_token_logprobs" in meta_info:
        new_token_ids = [item[1] for item in meta_info["output_token_logprobs"]]
        new_log_probs = [item[0] for item in meta_info["output_token_logprobs"]]
    else:
        new_token_ids = []
        new_log_probs = []

    # Decode response text
    response_text = output.get("text", "")
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-10]

    # Extract routed_experts if available
    routed_experts = None
    routed_experts_token_count = 0
    if return_routed_experts:
        if "routed_experts" in meta_info:
            routed_experts_data = meta_info["routed_experts"]
            routed_experts = decode_routed_experts(routed_experts_data, num_layers, moe_router_topk)
            routed_experts_token_count = len(input_ids) + len(new_token_ids)
            logger.debug(
                f"Turn {turn_idx}: Updated routed_experts shape={routed_experts.shape}, expected_tokens={routed_experts_token_count}"
            )
        else:
            logger.warning(
                f"Turn {turn_idx}: return_routed_experts=True but 'routed_experts' not in meta_info. "
                f"Available keys: {list(meta_info.keys())}. "
                f"Make sure sglang server has enable_return_routed_experts=True."
            )

    # Parse response for action
    text_for_action = response_text
    if "</think>" in text_for_action:
        # remove everything before the last </think>
        last_idx = text_for_action.rfind("</think>")
        if last_idx != -1:
            text_for_action = text_for_action[last_idx + len("</think>") :]

    if "<message>" in text_for_action and "</message>" in text_for_action:
        _, _, rest = text_for_action.partition("<message>")
        inner, _, _ = rest.partition("</message>")
        text_for_action = inner.strip()

    try:
        parsed = parse_tools(text_for_action, tools_info)
    except Exception as e:
        logger.warning(f"Turn {turn_idx}: Parse failed: {e}")
        return TurnResult(
            response_text=response_text,
            new_token_ids=new_token_ids,
            new_log_probs=new_log_probs,
            routed_experts=routed_experts,
            routed_experts_token_count=routed_experts_token_count,
            terminated=False,
            truncated=False,
            aborted=True,  # Treat parse failure as abort
            parse_failed=True,
            called_tool_signature="",
            called_tool_name="",
            reward=0.0,
            obs="",
            step_info={},
        )

    calls = parsed["calls"]
    called_tool_signature = ""
    called_tool_name = ""
    if calls:
        first_call = calls[0]
        call_name = first_call["name"]
        called_tool_name = call_name
        call_params = first_call["parameters"]
        if isinstance(call_params, str):
            try:
                call_params = json.loads(call_params)
            except json.JSONDecodeError:
                call_params = {}
        if not isinstance(call_params, dict):
            call_params = {}
        # Canonical signature so argument key order does not affect equality.
        called_tool_signature = f"{call_name}:{json.dumps(call_params, sort_keys=True, separators=(',', ':'))}"

    # Execute action in environment
    action_str = parsed_response_to_action_str(parsed)
    counter = get_global_counter()
    counts = ray.get(counter.inc.remote("active_env_steps"))
    print(f"[GlobalCounter] env.step START turn={turn_idx} {counts}", flush=True)
    try:
        with rollout_timer_total("tool_turn"):
            obs, reward, terminated, truncated, step_info = await loop.run_in_executor(
                None, lambda: env.step(action_str)
            )
    finally:
        counts = ray.get(counter.dec.remote("active_env_steps"))
        print(f"[GlobalCounter] env.step END   turn={turn_idx} {counts}", flush=True)

    return TurnResult(
        response_text=response_text,
        new_token_ids=new_token_ids,
        new_log_probs=new_log_probs,
        routed_experts=routed_experts,
        routed_experts_token_count=routed_experts_token_count,
        terminated=terminated,
        truncated=truncated,
        aborted=False,
        parse_failed=False,
        called_tool_signature=called_tool_signature,
        called_tool_name=called_tool_name,
        reward=reward,
        obs=obs,
        step_info=step_info,
    )


@weave.op
async def run_tau2_env_loop_async_moe(
    env,
    url: str,
    sampling_params: dict[str, Any],
    sample: Sample,
    state: GenerateState,
    args: Any,
    max_steps: int = 100,
    return_routed_experts: bool = False,
) -> Sample:
    """
    Async loop for MoE models with token-in/token-out and routed_experts collection.

    Uses sample_helper functions to maintain sample state consistently.

    Args:
        env: The tau2 gym environment
        url: The sglang server URL
        sampling_params: Sampling parameters for generation
        sample: The Sample object to populate (will be modified in place)
        state: GenerateState containing tokenizer and args
        args: Training arguments
        max_steps: Maximum number of interaction turns
        return_routed_experts: Whether to collect routed_experts for MoE routing replay

    Returns:
        The modified Sample with tokens, response, loss_mask, reward, etc.
    """
    loop = asyncio.get_event_loop()
    tokenizer = state.tokenizer
    counter = get_global_counter()

    counts = ray.get(counter.inc.remote("active_env_loops"))
    logger.warning(f"run_tau2_env_loop_async_moe called, global_counts={counts}")

    try:
        return await _run_tau2_env_loop_inner(
            env,
            url,
            sampling_params,
            sample,
            state,
            args,
            max_steps,
            return_routed_experts,
            loop,
            tokenizer,
            counter,
        )
    finally:
        counts = ray.get(counter.dec.remote("active_env_loops"))
        logger.warning(f"run_tau2_env_loop_async_moe done, global_counts={counts}")


async def _run_tau2_env_loop_inner(
    env,
    url,
    sampling_params,
    sample,
    state,
    args,
    max_steps,
    return_routed_experts,
    loop,
    tokenizer,
    counter,
) -> Sample:
    # Reset environment and get initial observation
    counts = ray.get(counter.inc.remote("active_env_steps"))
    print(f"[GlobalCounter] env.reset START {counts}", flush=True)
    try:
        _obs, info = await loop.run_in_executor(None, env.reset)
    finally:
        counts = ray.get(counter.dec.remote("active_env_steps"))
        print(f"[GlobalCounter] env.reset END   {counts}", flush=True)

    policy = info["policy"]
    tools = info["tools"]
    tools_info = [t.openai_schema for t in tools] if tools else []

    task_id = info["task"].id

    logger.info(f"Starting tau2 MoE env loop: task_id={task_id}, max_steps={max_steps}")

    # Build initial messages from observation
    # Read structured Message objects directly from the gym agent instead of
    # parsing the lossy string observation.  incremental=False because the
    # initial observation has no prior assistant turns to skip.
    messages = [
        {"role": "system", "content": system_prompt(policy)},
        # {"role": "assistant", "content": "Hi! How can I help you today?"}
        *tau2_observation_to_messages(env._agent.observation, incremental=False),
    ]

    # Initialize sample with prompt
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools_info)
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Set up sample initial state
    sample.prompt = prompt_text
    sample.tokens = list(prompt_token_ids)  # Will grow with responses
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.reward = 0.0
    sample.status = Sample.Status.COMPLETED
    sample.metadata = dict(info)
    sample.metadata["messages"] = list(messages)  # Track messages for sample_helper

    # Initialize pending fields for the new commit-on-assistant-message architecture
    sample.metadata["pending_tokens"] = []
    sample.metadata["pending_response"] = ""
    sample.metadata["pending_loss_mask"] = []
    sample.metadata["pending_log_probs"] = []

    # Initialize token length tracking
    sample.metadata["prompt_token_length"] = len(prompt_token_ids)
    sample.metadata["user_token_length"] = 0
    sample.metadata["assistant_token_length"] = 0
    sample.metadata["tool_response_token_length"] = 0

    # Initialize rollout_log_probs tracking
    sample.rollout_log_probs = []

    # MoE parameters from args
    num_layers = args.num_layers
    moe_router_topk = args.moe_router_topk
    num_prompt_tokens = len(prompt_token_ids)

    # Last routed_experts from sglang (full sequence, not incremental)
    last_routed_experts: np.ndarray | None = None
    last_routed_experts_token_count: int = 0
    turn_result: TurnResult | None = None

    total_reward = 0.0
    had_tool_parse_failure = False
    consecutive_same_tool_count = 0
    prev_called_tool_signature = ""
    had_successful_find_name_by_tool_call = False

    for turn_idx in range(max_steps):
        if turn_idx != 0:
            # Add environment response messages
            # Read structured messages directly from gym agent; incremental=True
            # returns only messages after the last AssistantMessage (i.e. new
            # tool responses / user messages from the previous step).
            new_parsed = tau2_observation_to_messages(env._agent.observation, incremental=True)
            for m in new_parsed:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "tool":
                    add_tool_response(sample, content, state, loss_mask_value=0)
                else:
                    add_user_message(sample, content, state, loss_mask_value=0)

            # Add generation prompt for next turn
            add_generation_prompt(sample, state, loss_mask_value=0)

        # Whole-session context limit: stop before generating a turn that would
        # exceed MAX_SESSION_TOKENS (committed + pending + this turn's max_new).
        # TRUNCATED (reward 0, kept in group, NOT aborted) so the model is pushed
        # to finish within budget; ABORTED only if nothing was ever generated
        # (prompt itself too long). Mirrors tool_call_agent's MAX_TOKENS guard.
        pending_count = get_pending_token_count(sample)
        projected_tokens = len(sample.tokens) + pending_count + sampling_params["max_new_tokens"]
        if projected_tokens > MAX_SESSION_TOKENS:
            if sample.response_length == 0:
                sample.status = Sample.Status.ABORTED
                sample.metadata["abort_reason"] = "prompt_too_long"
            else:
                sample.status = Sample.Status.TRUNCATED
            logger.warning(
                f"Session context limit at turn {turn_idx}: tokens={len(sample.tokens)} "
                f"pending={pending_count} max_new={sampling_params['max_new_tokens']} "
                f"projected={projected_tokens} > MAX_SESSION_TOKENS={MAX_SESSION_TOKENS}; "
                f"status={sample.status.name}"
            )
            break
        # Run single turn
        turn_result = await run_single_turn_async(
            env=env,
            url=url,
            sample=sample,
            state=state,
            sampling_params=sampling_params,
            tools_info=tools_info,
            policy=policy,
            turn_idx=turn_idx,
            num_layers=num_layers,
            moe_router_topk=moe_router_topk,
            return_routed_experts=return_routed_experts,
        )

        # Handle abort
        if turn_result.aborted:
            if turn_result.parse_failed:
                had_tool_parse_failure = True
            sample.status = Sample.Status.ABORTED
            # Still add assistant message if we got tokens (for parse failures)
            if turn_result.new_token_ids:
                add_assistant_message(
                    sample=sample,
                    token_ids=turn_result.new_token_ids,
                    state=state,
                    loss_mask_value=1,
                    log_probs=turn_result.new_log_probs,
                )
            break

        if turn_result.called_tool_signature:
            if prev_called_tool_signature == turn_result.called_tool_signature:
                consecutive_same_tool_count += 1
            prev_called_tool_signature = turn_result.called_tool_signature
        if turn_result.called_tool_name.startswith("find_user_id_"):
            had_successful_find_name_by_tool_call = True

        # Update routed_experts tracking
        if turn_result.routed_experts is not None:
            last_routed_experts = turn_result.routed_experts
            last_routed_experts_token_count = turn_result.routed_experts_token_count

        # Add assistant message using sample_helper
        add_assistant_message(
            sample=sample,
            token_ids=turn_result.new_token_ids,
            state=state,
            loss_mask_value=1,
            log_probs=turn_result.new_log_probs,
        )

        # Update reward and metadata
        total_reward = turn_result.reward
        sample.metadata.update(turn_result.step_info)

        if turn_result.terminated or turn_result.truncated:
            sample.status = Sample.Status.TRUNCATED if turn_result.truncated else Sample.Status.COMPLETED
            break

    # Set final reward (tau2 env reward plus optional penalties)
    assistant_token_length = sample.metadata["assistant_token_length"]
    sample.reward, penalty_info = final_reward_with_penalties(
        total_reward,
        had_tool_parse_failure,
        assistant_token_length,
        consecutive_same_tool_count,
        had_successful_find_name_by_tool_call,
    )
    sample.metadata["reward_penalty_info"] = penalty_info

    # Handle routed_experts for MoE routing replay
    if return_routed_experts and last_routed_experts is not None:
        # routed_experts from sglang contains routing for ALL tokens processed (prompt + response)
        # The training code expects: routed_experts.shape[0] == len(tokens) - 1

        assert last_routed_experts.shape[0] == last_routed_experts_token_count - 1, (
            f"last_routed_experts.shape[0] = {last_routed_experts.shape[0]}, last_routed_experts_token_count = {last_routed_experts_token_count}"
        )
        sample.rollout_routed_experts = last_routed_experts

        # Ensure tokens align with routed_experts
        # Slice tokens to match what routed_experts covers
        if len(sample.tokens) > last_routed_experts_token_count:
            logger.warning(
                "routed_experts shorter than tokens: len(tokens)=%d > routed_token_count=%d; slicing tokens to match",
                len(sample.tokens),
                last_routed_experts_token_count,
            )
            sample.tokens = sample.tokens[:last_routed_experts_token_count]
            sample.response_length = len(sample.tokens) - num_prompt_tokens
            sample.loss_mask = sample.loss_mask[: sample.response_length]
            if sample.rollout_log_probs:
                sample.rollout_log_probs = sample.rollout_log_probs[: sample.response_length]

        logger.info(
            f"Final routed_experts: shape={sample.rollout_routed_experts.shape}, "
            f"len(tokens)={len(sample.tokens)}, response_length={sample.response_length}, "
            f"expected_relation=(routed={sample.rollout_routed_experts.shape[0]}, tokens-1={len(sample.tokens) - 1})"
        )

    logger.info(
        f"Tau2 MoE env loop completed: task_id={task_id}, status={sample.status.name}, "
        f"total_tokens={len(sample.tokens)}, reward={sample.reward}, response_length={sample.response_length}"
    )

    return sample
