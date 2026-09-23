"""
Convert WebArena tool-agent trajectories (Bedrock Converse API) to SFT JSONL.

Source: results/claude_45_*/ sessions with reasoningContent + toolUse blocks
Target: HF messages JSONL with reasoning_content + tool_calls for Megatron-Bridge SFT

Output per sample:
  {"messages": [
    {"role": "system", "content": "<tool_system prompt>"},
    {"role": "user", "content": "Here is the initial state..."},
    {"role": "assistant", "reasoning_content": "...", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "<observation>"},
    ...
  ], "tools": [<step_browser schema>]}

Filters: task succeeded AND has reasoning content in at least one assistant turn.
"""

import argparse
import json
import logging
from pathlib import Path

from rl_web_agent.prompts import load_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STEP_BROWSER_TOOL = {
    "type": "function",
    "function": {
        "name": "step_browser",
        "description": "Execute an action in the web browser environment and get the resulting observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Type of action to perform",
                    "enum": [
                        "click",
                        "type",
                        "hover",
                        "select",
                        "clear",
                        "key_press",
                        "goto_url",
                        "back",
                        "forward",
                        "refresh",
                        "new_tab",
                        "switch_tab",
                        "close_tab",
                        "terminate",
                    ],
                },
                "target": {
                    "type": "string",
                    "description": "Semantic ID of the element to interact with",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for type action)",
                },
                "enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing",
                },
                "value": {
                    "type": "string",
                    "description": "Value to select (for select action)",
                },
                "key": {
                    "type": "string",
                    "description": "Key to press (for key_press action)",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "tab_id": {
                    "type": "integer",
                    "description": "Tab ID for tab operations",
                },
                "answer": {
                    "type": "string",
                    "description": "Final answer for terminate action",
                },
            },
            "required": ["action"],
        },
    },
}


def extract_reasoning(content_blocks: list[dict]) -> str:
    """Extract reasoning text from Bedrock reasoningContent or thinking blocks."""
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if "reasoningContent" in block:
            rc = block["reasoningContent"]
            if isinstance(rc, dict) and "reasoningText" in rc:
                rt = rc["reasoningText"]
                return rt["text"] if isinstance(rt, dict) else str(rt)
        if "thinking" in block:
            td = block["thinking"]
            if isinstance(td, dict):
                return td.get("thinking", "") or td.get("text", "")
            return str(td)
    return ""


def convert_session(session: dict, task_config: dict) -> dict | None:
    """
    Convert a Bedrock tool-agent session to HF messages format.
    Returns {"messages": [...], "tools": [...]} or None if no reasoning found.
    """
    objective = task_config["intent"]
    system_prompt = load_prompt("tool_system").format(objective=objective)

    messages = [{"role": "system", "content": system_prompt}]
    conversation = session["conversation_history"]
    has_reasoning = False
    is_first_user = True

    for msg in conversation:
        role = msg["role"]

        if role == "user":
            text_parts = []
            for block in msg["content"]:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolResult" in block:
                    for tc in block["toolResult"].get("content", []):
                        if "text" in tc:
                            text_parts.append(tc["text"])
                        elif "json" in tc:
                            text_parts.append(json.dumps(tc["json"]))

            obs_text = "\n".join(text_parts)
            if not obs_text:
                continue

            if is_first_user:
                messages.append({"role": "user", "content": obs_text})
                is_first_user = False
            else:
                messages.append({"role": "tool", "content": obs_text})

        elif role == "assistant":
            content_blocks = msg["content"]
            reasoning = extract_reasoning(content_blocks)
            text_content = ""
            tool_calls = []

            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    text_content += block["text"]
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    tool_calls.append(
                        {
                            "function": {
                                "name": tu["name"],
                                "arguments": json.dumps(tu["input"]),
                            }
                        }
                    )

            assistant_msg: dict = {"role": "assistant", "content": text_content}
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
                has_reasoning = True
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

    if not has_reasoning:
        return None

    return {"messages": messages, "tools": [STEP_BROWSER_TOOL]}


def load_task(task_dir: Path) -> tuple[dict | None, dict | None]:
    result_path = task_dir / "result.json"
    session_path = task_dir / "session.json"
    if not result_path.exists() or not session_path.exists():
        return None, None
    return json.loads(result_path.read_text()), json.loads(session_path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Convert WebArena tool-agent trajectories to SFT JSONL")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--glob-pattern", type=str, default="claude_45_*")
    parser.add_argument("--output", type=Path, default=Path("sft_data/training.jsonl"))
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    batch_dirs = sorted(args.results_dir.glob(args.glob_pattern))
    if not batch_dirs:
        logger.error(f"No batch directories matching '{args.glob_pattern}' in {args.results_dir}")
        return

    logger.info(f"Found {len(batch_dirs)} batch directories")

    samples: list[dict] = []
    stats = {
        "total_tasks": 0,
        "skipped_no_files": 0,
        "skipped_failed": 0,
        "skipped_no_reasoning": 0,
        "skipped_low_score": 0,
        "converted": 0,
    }

    for batch_dir in batch_dirs:
        for task_dir in sorted(batch_dir.glob("task_*")):
            stats["total_tasks"] += 1

            result, session = load_task(task_dir)
            if result is None or session is None:
                stats["skipped_no_files"] += 1
                continue

            r = result["result"]
            if r is None or not r["success"]:
                stats["skipped_failed"] += 1
                continue

            if r["score"] <= args.min_score:
                stats["skipped_low_score"] += 1
                continue

            hf_data = convert_session(session, result["task_config"])
            if hf_data is None:
                stats["skipped_no_reasoning"] += 1
                continue

            samples.append(hf_data)
            stats["converted"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for hf_data in samples:
            f.write(json.dumps(hf_data, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(samples)} samples to {args.output}")
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")

    if samples:
        sample = samples[0]
        msgs = sample["messages"]
        n_tool_calls = sum(1 for m in msgs if m.get("tool_calls"))
        n_reasoning = sum(1 for m in msgs if m.get("reasoning_content"))
        logger.info(f"First sample: {len(msgs)} messages, {n_tool_calls} tool_calls, {n_reasoning} with reasoning")


if __name__ == "__main__":
    main()
