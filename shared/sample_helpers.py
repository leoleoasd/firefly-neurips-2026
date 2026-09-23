"""
Helper functions for working with Sample objects in multi-turn / tool-calling scenarios.

These functions help manage tokens, response, loss_mask, and rollout_log_probs
in a way that's consistent with slime's training requirements.

Design:
    - All functions except add_assistant_message accumulate to pending_* fields in metadata
    - add_assistant_message "commits" pending changes + assistant tokens to sample.tokens
    - This ensures sample.tokens always ends after an assistant message

Rollout-value update modes. rollout_log_probs and rollout_routed_experts each have an
independent incremental/full toggle:
    - rollout_log_probs: SLIME_ROLLOUT_FULL_LOGPROBS env var (see USE_FULL_LOGPROBS).
    - rollout_routed_experts: SLIME_ROLLOUT_FULL_ROUTING env var (see USE_FULL_ROUTING).
The two were historically toggled together; they are now decoupled so log-probs and routing
replay can use different modes.

    - Incremental (default): add_assistant_message appends each turn's generation-time
      log_probs, and update_rollout_routed_experts appends only the new routing tail. Both keep
      the values the model *actually* used when it generated each turn.
    - Full update: each turn, OVERWRITE rollout_log_probs from sglang's input_token_logprobs
      (the prefill-recomputed log-prob of every position, returned for free by the normal
      generate request when sent with logprob_start_len=0 — no extra forward pass) AND/OR overwrite
      rollout_routed_experts with this turn's full-sequence routing. These come from the same
      recomputed forward pass, consistent with the full context rather than per-turn values.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample


# Full-update log-prob mode (env-var toggle). When enabled, add_assistant_message overwrites
# rollout_log_probs each turn from sglang's prefill-recomputed input_token_logprobs (passed in
# by the caller) instead of appending the per-turn generation log-probs. See module docstring.
USE_FULL_LOGPROBS = bool(int(os.environ.get("SLIME_ROLLOUT_FULL_LOGPROBS", "0")))

# Full-update routing mode (independent env-var toggle). When enabled, update_rollout_routed_experts
# overwrites rollout_routed_experts each turn with this turn's full-sequence routing instead of
# appending only the new tail. Decoupled from USE_FULL_LOGPROBS. See module docstring.
USE_FULL_ROUTING = bool(int(os.environ.get("SLIME_ROLLOUT_FULL_ROUTING", "0")))


# Cache for prefix text length per tokenizer (keyed by tokenizer id)
_prefix_length_cache: dict[int, int] = {}
# Cache for generation prompt text per tokenizer
_generation_prompt_cache: dict[int, str] = {}
# Cache for newline token id per tokenizer
_newline_token_cache: dict[int, int] = {}
# Cache for whether the chat template has a trailing \n after the eot token
_has_post_eot_newline_cache: dict[int, bool] = {}


def _get_prefix_length(tokenizer) -> int:
    """Get cached prefix length for a tokenizer."""
    tokenizer_id = id(tokenizer)
    if tokenizer_id not in _prefix_length_cache:
        dummy_message = {"role": "user", "content": ""}
        prefix_text = tokenizer.apply_chat_template([dummy_message], tokenize=False, add_generation_prompt=False)
        _prefix_length_cache[tokenizer_id] = len(prefix_text)
    return _prefix_length_cache[tokenizer_id]


def _get_newline_token(tokenizer) -> int:
    """Get cached newline token id for a tokenizer."""
    tokenizer_id = id(tokenizer)
    if tokenizer_id not in _newline_token_cache:
        tokens = tokenizer.encode("\n", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected '\\n' to be a single token, got {tokens}"
        _newline_token_cache[tokenizer_id] = tokens[0]
    return _newline_token_cache[tokenizer_id]


def _get_generation_prompt(tokenizer) -> str:
    """Get cached generation prompt text for a tokenizer."""
    tokenizer_id = id(tokenizer)
    if tokenizer_id not in _generation_prompt_cache:
        dummy_message = {"role": "user", "content": ""}
        without_gen = tokenizer.apply_chat_template([dummy_message], tokenize=False, add_generation_prompt=False)
        with_gen = tokenizer.apply_chat_template([dummy_message], tokenize=False, add_generation_prompt=True)
        _generation_prompt_cache[tokenizer_id] = with_gen[len(without_gen) :]
    return _generation_prompt_cache[tokenizer_id]


def _has_post_eot_newline(tokenizer) -> bool:
    """Check if the chat template has a trailing '\\n' after the eot token between messages."""
    tokenizer_id = id(tokenizer)
    if tokenizer_id not in _has_post_eot_newline_cache:
        messages = [
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        _has_post_eot_newline_cache[tokenizer_id] = (tokenizer.eos_token + "\n") in rendered
    return _has_post_eot_newline_cache[tokenizer_id]


def _ensure_pending_fields(sample: Sample) -> None:
    """Ensure pending fields exist in metadata."""
    if "pending_tokens" not in sample.metadata:
        sample.metadata["pending_tokens"] = []
    if "pending_response" not in sample.metadata:
        sample.metadata["pending_response"] = ""
    if "pending_loss_mask" not in sample.metadata:
        sample.metadata["pending_loss_mask"] = []
    if "pending_log_probs" not in sample.metadata:
        sample.metadata["pending_log_probs"] = []


def _ensure_token_length_fields(sample: Sample) -> None:
    """Ensure token length tracking fields exist in metadata."""
    if "prompt_token_length" not in sample.metadata:
        sample.metadata["prompt_token_length"] = 0
    if "user_token_length" not in sample.metadata:
        sample.metadata["user_token_length"] = 0
    if "assistant_token_length" not in sample.metadata:
        sample.metadata["assistant_token_length"] = 0
    if "tool_response_token_length" not in sample.metadata:
        sample.metadata["tool_response_token_length"] = 0


def _add_to_pending(
    sample: Sample,
    tokens: list[int],
    text: str,
    loss_mask_value: int,
    log_probs: list[float] | None = None,
) -> None:
    """Add tokens/text to pending fields (not committed to sample.tokens yet)."""
    _ensure_pending_fields(sample)
    sample.metadata["pending_tokens"] += tokens
    sample.metadata["pending_response"] += text
    sample.metadata["pending_loss_mask"] += [loss_mask_value] * len(tokens)
    if log_probs is not None:
        sample.metadata["pending_log_probs"] += log_probs
    else:
        sample.metadata["pending_log_probs"] += [0.0] * len(tokens)


def add_text(sample: Sample, text: str, state: GenerateState, loss_mask_value: int = 1) -> None:
    """
    Tokenize and append text to pending (not committed until add_assistant_message).

    Args:
        sample: The sample to modify.
        text: The text to append to the response.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (1 = train on, 0 = mask out).
    """
    new_tokens = state.tokenizer.encode(text, add_special_tokens=False)
    _add_to_pending(sample, new_tokens, text, loss_mask_value)


def add_message(
    sample: Sample,
    message: dict[str, str],
    state: GenerateState,
    loss_mask_value: int = 1,
) -> None:
    """
    Apply chat template to a message and append only the formatted message part to pending.

    Args:
        sample: The sample to modify.
        message: A message dict with 'role' and 'content' keys.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (1 = train on, 0 = mask out).
    """
    prefix_length = _get_prefix_length(state.tokenizer)
    # Apply chat template with dummy + our message
    dummy_message = {"role": "user", "content": ""}
    full_text = state.tokenizer.apply_chat_template(
        [dummy_message, message], tokenize=False, add_generation_prompt=False
    )
    # Extract only the new message part
    message_text = full_text[prefix_length:]
    add_text(sample, message_text, state, loss_mask_value)


def add_user_message(sample: Sample, content: str, state: GenerateState, loss_mask_value: int = 0) -> None:
    """
    Add a user message to pending.

    Args:
        sample: The sample to modify.
        content: The user message content.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (default 0 = mask out).
    """
    _ensure_token_length_fields(sample)
    pending_before = len(sample.metadata["pending_tokens"]) if "pending_tokens" in sample.metadata else 0
    message = {"role": "user", "content": content}
    add_message(sample, message, state, loss_mask_value)
    pending_after = len(sample.metadata["pending_tokens"])
    sample.metadata["user_token_length"] += pending_after - pending_before
    sample.metadata["messages"].append(message)


def add_tool_response(sample: Sample, content: str, state: GenerateState, loss_mask_value: int = 0) -> None:
    """
    Add a tool response message to pending.

    Args:
        sample: The sample to modify.
        content: The tool response content.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (default 0 = mask out).
    """
    _ensure_token_length_fields(sample)
    pending_before = len(sample.metadata["pending_tokens"]) if "pending_tokens" in sample.metadata else 0
    message = {"role": "tool", "content": content}
    add_message(sample, message, state, loss_mask_value)
    pending_after = len(sample.metadata["pending_tokens"])
    sample.metadata["tool_response_token_length"] += pending_after - pending_before
    sample.metadata["messages"].append(message)


def add_system_message(sample: Sample, content: str, state: GenerateState, loss_mask_value: int = 0) -> None:
    """
    Add a system message to pending (e.g. for format error feedback).

    Args:
        sample: The sample to modify.
        content: The system message content.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (default 0 = mask out).
    """
    _ensure_token_length_fields(sample)
    message = {"role": "system", "content": content}
    add_message(sample, message, state, loss_mask_value)
    sample.metadata["messages"].append(message)


def add_generation_prompt(sample: Sample, state: GenerateState, loss_mask_value: int = 0) -> None:
    """
    Add the generation prompt (e.g. "<|im_start|>assistant\n") to pending.

    Tokens are tracked under assistant_token_length since they're part of assistant turns.

    Args:
        sample: The sample to modify.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (default 0 = mask out).
    """
    _ensure_token_length_fields(sample)
    pending_before = len(sample.metadata["pending_tokens"]) if "pending_tokens" in sample.metadata else 0
    gen_prompt = _get_generation_prompt(state.tokenizer)
    add_text(sample, gen_prompt, state, loss_mask_value)
    pending_after = len(sample.metadata["pending_tokens"])
    sample.metadata["assistant_token_length"] += pending_after - pending_before


def _overwrite_full_rollout_log_probs(
    sample: Sample,
    input_log_probs: list | None,
    assistant_log_probs: list[float],
) -> None:
    """Rebuild the whole response span's rollout_log_probs (full-update mode).

    Call AFTER sample.tokens / response_length / loss_mask have been updated for this turn.

    ``input_log_probs`` is ``meta_info["input_token_logprobs"]`` from THIS turn's generate request,
    which was sent with logprob_start_len=0. It therefore covers the PREFIX only — one entry per
    input position 0..len(input_ids)-1 (each ``[logprob, token_id, ...]``; the first entry is null),
    where ``input_ids`` is every committed + pending token that existed BEFORE this turn generated.
    This turn's generated-token logprobs are NOT in here — they are ``assistant_log_probs`` (from
    ``output_token_logprobs``). The full current sequence's logprobs are therefore the prefix values
    followed by ``assistant_log_probs``; the response span is the LAST ``response_length`` entries of
    that (slicing from the end skips the null position-0, which sits in the prompt). Masked
    (non-response) positions are zeroed to match the training-side convention.

    If ``input_log_probs`` is None or too short (logprobs unavailable, or a degenerate response —
    e.g. a full prefix-cache hit returns no input logprobs), fall back to appending this turn's
    generation-time logprobs (incremental behavior) so we never corrupt or crash the rollout.
    """
    response_length = sample.response_length

    if input_log_probs is None:
        # No prefix logprobs this turn — fall back to appending (incremental).
        sample.rollout_log_probs += sample.metadata["pending_log_probs"] + assistant_log_probs
        return

    # Full current sequence = prefix logprobs (this turn's input) ++ this turn's generated logprobs.
    prefix_values = [item[0] if item[0] is not None else 0.0 for item in input_log_probs]
    full_values = prefix_values + assistant_log_probs
    if response_length > len(full_values):
        # Prefix logprobs came back short (degenerate/empty response — full prefix-cache hit, or an
        # aborted request). Don't crash; fall back to appending this turn's generation-time logprobs.
        sample.rollout_log_probs += sample.metadata["pending_log_probs"] + assistant_log_probs
        return
    response_logprobs = full_values[-response_length:] if response_length > 0 else []

    if sample.loss_mask is not None:
        assert len(sample.loss_mask) == response_length, (
            f"loss_mask length ({len(sample.loss_mask)}) must match response_length ({response_length})"
        )
        sample.rollout_log_probs = [
            lp if mask else 0.0 for lp, mask in zip(response_logprobs, sample.loss_mask, strict=True)
        ]
    else:
        sample.rollout_log_probs = response_logprobs


def add_assistant_message(
    sample: Sample,
    token_ids: list[int],
    state: GenerateState,
    loss_mask_value: int = 1,
    log_probs: list[float] | None = None,
    input_log_probs: list | None = None,
) -> None:
    """
    Add an assistant message and commit all pending changes to sample.tokens.

    This is the only function that modifies sample.tokens, ensuring it always
    ends after an assistant message.

    Callers should ALWAYS pass the complete log-prob info from this turn's generate
    response — both ``log_probs`` (this turn's generated tokens) and ``input_log_probs``
    (the whole prefix's prefill-recomputed logprobs). This function then chooses, based on
    :data:`USE_FULL_LOGPROBS`, whether to append incrementally or overwrite the full span,
    so the mode logic lives here once rather than in every caller.

    Args:
        sample: The sample to modify.
        token_ids: The token ids to add.
        state: The GenerateState containing the tokenizer.
        loss_mask_value: The loss mask value for the new tokens (default 1 = train on).
        log_probs: This turn's generated-token log-probs (from ``output_token_logprobs``).
        input_log_probs: This turn's ``meta_info["input_token_logprobs"]`` (requires the
            generate request to have used ``logprob_start_len=0``). Used only in full-update
            mode; one entry per input position ``[logprob, token_id, ...]``, first entry null.
    """
    _ensure_pending_fields(sample)
    _ensure_token_length_fields(sample)

    # Get pending data
    pending_tokens = sample.metadata["pending_tokens"]
    pending_response = sample.metadata["pending_response"]
    pending_loss_mask = sample.metadata["pending_loss_mask"]
    pending_log_probs = sample.metadata["pending_log_probs"]

    # Decode assistant message
    assistant_text = state.tokenizer.decode(token_ids, skip_special_tokens=False)

    # Build assistant log probs
    if log_probs is not None:
        assert len(log_probs) == len(token_ids), (
            f"log_probs length ({len(log_probs)}) must match token_ids length ({len(token_ids)})"
        )
        assistant_log_probs = log_probs
    else:
        assistant_log_probs = [0.0] * len(token_ids)

    # Commit: pending + assistant tokens
    all_new_tokens = pending_tokens + token_ids
    all_new_response = pending_response + assistant_text
    all_new_loss_mask = pending_loss_mask + [loss_mask_value] * len(token_ids)
    all_new_log_probs = pending_log_probs + assistant_log_probs

    # Update sample.tokens and response
    sample.tokens = sample.tokens + all_new_tokens
    sample.response += all_new_response
    sample.response_length += len(all_new_tokens)

    # Track assistant token length (only the assistant tokens, not pending)
    sample.metadata["assistant_token_length"] += len(token_ids)

    # Update loss_mask
    if sample.loss_mask is None:
        sample.loss_mask = [1] * (sample.response_length - len(all_new_tokens))
    sample.loss_mask += all_new_loss_mask

    # Update rollout_log_probs.
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    if USE_FULL_LOGPROBS:
        # Full-update: rebuild the entire response span from this turn's prefill-recomputed
        # input_token_logprobs (the prefix: pending + all prior turns) plus this turn's
        # generated-token logprobs (the tail not covered by input_log_probs, since those
        # tokens didn't exist when the request was sent). Overwrites every prior turn's
        # values with fresher full-context logprobs.
        _overwrite_full_rollout_log_probs(sample, input_log_probs, assistant_log_probs)
    else:
        # Incremental: keep each turn's generation-time logprobs as-is.
        sample.rollout_log_probs += all_new_log_probs

    # Track assistant message in metadata — strip the eot token from content
    eot_token = state.tokenizer.eos_token
    if eot_token and assistant_text.endswith(eot_token):
        content = assistant_text[: -len(eot_token)]
    else:
        content = assistant_text
    sample.metadata["messages"].append({"role": "assistant", "content": content})

    # Clear pending fields
    sample.metadata["pending_tokens"] = []
    sample.metadata["pending_response"] = ""
    sample.metadata["pending_loss_mask"] = []
    sample.metadata["pending_log_probs"] = []

    # sglang stops generation at the eot token without any trailing characters.
    # Some chat templates (e.g. Qwen) expect a '\n' after the eot token between
    # messages, while others (e.g. Llama) do not.  Only add it when needed so
    # the next message (user / tool_response) is formatted correctly.
    #
    # This '\n' is chat-template scaffolding, not a token the model generated, so it
    # is masked out (loss_mask_value=0): we must not train on it, and its placeholder
    # rollout log-prob (0.0 == prob 1.0) would otherwise corrupt the TIS importance
    # ratio for a trained position.
    if _has_post_eot_newline(state.tokenizer):
        newline_token = _get_newline_token(state.tokenizer)
        _add_to_pending(sample, [newline_token], "\n", loss_mask_value=0)


def get_pending_token_count(sample: Sample) -> int:
    """Get the number of pending tokens not yet committed."""
    if "pending_tokens" not in sample.metadata:
        return 0
    return len(sample.metadata["pending_tokens"])


def update_rollout_routed_experts(sample: Sample, new_experts) -> None:
    """Update rollout_routed_experts for this turn, honoring the incremental/full mode.

    sglang returns routed experts for the ENTIRE request (positions ``0..seqlen-1``) every
    turn — i.e. ``new_experts`` always covers the full prefix + this turn's tokens. The mode
    (toggled by :data:`USE_FULL_ROUTING`, independent of the rollout_log_probs switch) decides
    what we keep:

    - Incremental (default): keep the values from earlier turns (what the model *actually*
      routed through when it generated those tokens) and only append this turn's new tail.
      Re-using a later turn's recomputed routing for earlier tokens would drift due to fp16
      non-determinism, EP load balancing, KV-cache eviction, etc., making
      ``--use-rollout-routing-replay`` train against routing that didn't really happen.
    - Full update: OVERWRITE with this turn's full-sequence routing (this turn's recomputed
      forward pass), consistent with the full context rather than per-turn values.

    Args:
        sample: The sample whose ``rollout_routed_experts`` to update.
        new_experts: numpy array of shape ``(seqlen-1, num_layers, topk)`` for this turn
            (covers the full prefix + this turn's generated tokens).
    """
    if USE_FULL_ROUTING:
        sample.rollout_routed_experts = new_experts
        return

    if sample.rollout_routed_experts is None:
        sample.rollout_routed_experts = new_experts
        return

    existing_len = sample.rollout_routed_experts.shape[0]
    new_len = new_experts.shape[0]
    if new_len <= existing_len:
        # Nothing new — should not happen in practice but be defensive.
        return

    import numpy as np  # local import to keep top-level light

    tail = new_experts[existing_len:]
    sample.rollout_routed_experts = np.concatenate([sample.rollout_routed_experts, tail], axis=0)
