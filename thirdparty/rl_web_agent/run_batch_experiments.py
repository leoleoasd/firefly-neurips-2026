#!/usr/bin/env python3
"""
Run batch agent experiments across multiple models and parsers.

Models run in parallel, parsers run sequentially within each model.

Usage:
    python run_batch_experiments.py [options]

Examples:
    # Dry run to see what commands would be executed
    python run_batch_experiments.py --dry-run

    # Run specific models and parsers
    python run_batch_experiments.py --models claude_40_sonnet,gpt_4o --parsers numeric

    # Run all combinations
    python run_batch_experiments.py

    # Run specific task IDs
    python run_batch_experiments.py --task-ids 104,117,118
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Model definitions: name -> (provider, model_id_or_deployment, extra_args)
# provider: "bedrock", "azure_openai", or "openai"
# For bedrock: model_id_or_deployment is the inference profile suffix
# For azure_openai: model_id_or_deployment is the deployment name
# For openai: model_id_or_deployment is the model name
MODELS = {
    # Bedrock Claude models
    "claude_35_sonnet": (
        "bedrock",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        [],
    ),
    "claude_37_sonnet": (
        "bedrock",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        [],
    ),
    "claude_40_sonnet": (
        "bedrock",
        "global.anthropic.claude-sonnet-4-20250514-v1:0",
        ["+llm.bedrock.additionalModelRequestFields.anthropic_beta=[context-1m-2025-08-07]"],
    ),
    "claude_45_sonnet": (
        "bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ["+llm.bedrock.additionalModelRequestFields.anthropic_beta=[context-1m-2025-08-07]"],
    ),
    # Bedrock Claude models - VLM mode (sends screenshots to LLM)
    "claude_40_sonnet_vlm": (
        "bedrock",
        "global.anthropic.claude-sonnet-4-20250514-v1:0",
        [
            "+llm.bedrock.additionalModelRequestFields.anthropic_beta=[context-1m-2025-08-07]",
            "--vlm",
        ],
    ),
    "claude_45_sonnet_vlm": (
        "bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        [
            "+llm.bedrock.additionalModelRequestFields.anthropic_beta=[context-1m-2025-08-07]",
            "--vlm",
        ],
    ),
    # Bedrock DeepSeek models
    "deepseek_r1": (
        "bedrock",
        "us.deepseek.r1-v1:0",
        [],
    ),
    # Azure OpenAI models
    "gpt_4o": (
        "azure_openai",
        "gpt-4o",
        [],
    ),
    "gpt_4o_mini": (
        "azure_openai",
        "gpt-4o-mini",
        [],
    ),
    "gpt_5": (
        "azure_openai",
        "gpt-5",
        [],
    ),
    "o3": (
        "azure_openai",
        "o3",
        [],
    ),
    "o3_fix_prompt": (
        "azure_openai",
        "o3",
        [],
    ),
    # OpenAI-compatible models (local/vLLM)
    # "qwen25_32b": (
    #     "openai",
    #     "Qwen/Qwen2.5-32B",
    #     ["llm.openai.base_url=http://0.0.0.0:8003/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    # ),
    # # OpenAI-compatible models (local/vLLM)
    # "qwen3_4b": (
    #     "openai",
    #     "Qwen/Qwen3-4B",
    #     [
    #         "llm.openai.base_url=http://0.0.0.0:8002/v1",
    #         "llm.openai.api_key=none",
    #         "llm.openai.timeout=360"
    #     ],
    # ),
    # # OpenAI-compatible models (local/vLLM)
    # "qwen3_8b": (
    #     "openai",
    #     "Qwen/Qwen3-8B",
    #     [
    #         "llm.openai.base_url=http://0.0.0.0:8000/v1",
    #         "llm.openai.api_key=none",
    #         "llm.openai.timeout=360"
    #     ],
    # ),
    # # OpenAI-compatible models (local/vLLM)
    # "qwen25_7b": (
    #     "openai",
    #     "Qwen/Qwen2.5-7B-Instruct",
    #     [
    #         "llm.openai.base_url=http://0.0.0.0:8000/v1",
    #         "llm.openai.api_key=none",
    #         "llm.openai.timeout=360"
    #     ],
    # ),
    # "llama31_8b": (
    #     "openai",
    #     "meta-llama/Llama-3.1-8B-Instruct",
    #     ["llm.openai.base_url=http://0.0.0.0:8000/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    # ),
    # "qwen25_3b": (
    #     "openai",
    #     "Qwen/Qwen2.5-3B-Instruct",
    #     ["llm.openai.base_url=http://0.0.0.0:8000/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    # ),
    # "qwen25_32b": (
    #     "openai",
    #     "Qwen/Qwen2.5-32B-Instruct",
    #     ["llm.openai.base_url=http://0.0.0.0:8001/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    # ),
    "qwen30ba3b": (
        "openai",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        ["--agent_type", "tool", "llm.openai.base_url=http://0.0.0.0:41035/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    ),
    "qwen30ba3b_finetuned": (
        "openai",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        ["--agent_type", "tool", "llm.openai.base_url=http://0.0.0.0:45723/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    ),
    "qwen4b_finetuned": (
        "openai",
        "qwen/Qwen3-4B-Thinking-2507",
        ["--agent_type", "tool", "llm.openai.base_url=http://127.0.0.1:30000/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    ),
    "qwen4b": (
        "openai",
        "qwen/Qwen3-4B-Thinking-2507",
        ["--agent_type", "tool", "llm.openai.base_url=http://127.0.0.1:30001/v1", "llm.openai.api_key=none", "llm.openai.timeout=360"],
    ),
}

# Parser definitions (name -> script path)
PARSERS = {
    "default": "rl_web_agent/javascript/parser.js",
    # "numeric": "rl_web_agent/javascript/parser_numeric_id.js",
    # "no_visual_cue": "rl_web_agent/javascript/parser_no_visual_cue.js",
    # "no_visual_cue_numeric": "rl_web_agent/javascript/parser_no_visual_cue_numeric.js",
}


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""

    model_name: str
    provider: str
    model_id: str  # Full ARN for bedrock, deployment name for azure
    parser_name: str
    parser_path: str
    output_dir: str
    region: str
    max_concurrent: int
    sites: str
    tasks_dir: str
    max_tokens: int
    task_ids: str | None = None  # Comma-separated list of task IDs to run
    only_failed: bool = False
    extra_args: list[str] = field(default_factory=list)


def build_command(config: RunConfig) -> list[str]:
    """Build the batch agent command."""
    cmd = [
        sys.executable,
        "-m",
        "rl_web_agent.entrypoints.batch_agent",
        "--max_concurrent",
        str(config.max_concurrent),
        "--max_concurrent_launch",
        "5",
        "--output_dir",
        config.output_dir,
        "--tasks_dir",
        config.tasks_dir,
    ]

    if config.only_failed:
        cmd.append("--only-failed")

    cmd.extend(
        [
            f"llm.provider={config.provider}",
            "llm.generation.temperature=null",
            "llm.generation.top_p=null",
            "llm.max_concurrent=100",
            f"llm.generation.max_tokens={config.max_tokens}",
            f"environment.parser_script_path={config.parser_path}",
        ]
    )

    # Add task IDs if specified, otherwise use sites
    if config.task_ids:
        cmd.extend(["--task_ids", config.task_ids])
    else:
        cmd.extend(["--sites", config.sites])

    # Add provider-specific arguments
    if config.provider == "bedrock":
        cmd.append(f"llm.bedrock.region={config.region}")
        cmd.append(f"llm.bedrock.model_id={config.model_id}")
    elif config.provider == "azure_openai":
        cmd.append(f"llm.azure_openai.deployment={config.model_id}")
    elif config.provider == "openai":
        cmd.append(f"llm.openai.model={config.model_id}")

    # Add model-specific extra arguments
    cmd.extend(config.extra_args)
    return cmd


def run_single_experiment(config: RunConfig) -> tuple[str, bool, str]:
    """Run a single experiment and return (name, success, message)."""
    name = f"{config.model_name}_{config.parser_name}"
    cmd = build_command(config)

    print(f"[{name}] Starting...")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        return (name, False, f"exit code {result.returncode}")
    return (name, True, config.output_dir)


def run_model_experiments(
    model_name: str,
    provider: str,
    model_id: str,
    extra_args: list[str],
    parser_list: list[str],
    args: argparse.Namespace,
) -> list[tuple[str, bool, str]]:
    """Run all parser experiments for a single model sequentially."""
    results = []

    for parser_name in parser_list:
        parser_path = PARSERS[parser_name]
        output_dir = f"{args.output_dir}/{model_name}_{parser_name}"

        config = RunConfig(
            model_name=model_name,
            provider=provider,
            model_id=model_id,
            parser_name=parser_name,
            parser_path=parser_path,
            output_dir=output_dir,
            region=args.region,
            max_concurrent=args.max_concurrent,
            sites=args.sites,
            tasks_dir=args.tasks_dir,
            max_tokens=args.max_tokens,
            task_ids=args.task_ids,
            only_failed=args.only_failed,
            extra_args=extra_args,
        )

        result = run_single_experiment(config)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run batch agent experiments across multiple models and parsers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  Bedrock: {', '.join(k for k, v in MODELS.items() if v[0] == 'bedrock')}
  Azure:   {', '.join(k for k, v in MODELS.items() if v[0] == 'azure_openai')}

Available parsers:
  {', '.join(PARSERS.keys())}

Examples:
  # Dry run
  python run_batch_experiments.py --dry-run

  # Run specific models
  python run_batch_experiments.py --models claude_40_sonnet,gpt_4o

  # Run specific parser only
  python run_batch_experiments.py --parsers numeric

  # Run specific task IDs only
  python run_batch_experiments.py --task-ids 104,117,118

  # Run specific task IDs with specific model
  python run_batch_experiments.py --models claude_40_sonnet --task-ids 104,117,118
""",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated list of models (default: all). Options: {','.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--parsers",
        type=str,
        default=None,
        help=f"Comma-separated list of parsers (default: all). Options: {','.join(PARSERS.keys())}",
    )
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region for Bedrock (default: us-west-2)")
    parser.add_argument("--account", type=str, default=None, help="AWS account ID for Bedrock (default: AWS_ACCOUNT_ID env var)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent tasks per model (default: 10)")
    parser.add_argument("--sites", type=str, default="shopping,shopping_admin,gitlab", help="Sites to test (ignored if --task-ids is specified)")
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated list of specific task IDs to run (overrides --sites)")
    parser.add_argument("--tasks-dir", type=str, default="dataset/test_webarena_lite", help="Tasks directory")
    parser.add_argument("--output-dir", type=str, default="results/webarena_lite", help="Base output directory")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for generation")
    parser.add_argument("--only-failed", action="store_true", help="Only re-run failed tasks from previous results")

    args = parser.parse_args()

    # Parse model and parser lists
    model_list = args.models.split(",") if args.models else list(MODELS.keys())
    parser_list = args.parsers.split(",") if args.parsers else list(PARSERS.keys())

    # Validate models
    for model in model_list:
        if model not in MODELS:
            print(f"ERROR: Unknown model '{model}'. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    # Validate parsers
    for p in parser_list:
        if p not in PARSERS:
            print(f"ERROR: Unknown parser '{p}'. Available: {', '.join(PARSERS.keys())}")
            sys.exit(1)

    total_runs = len(model_list) * len(parser_list)

    print("=" * 50)
    print("Batch Experiment Runner (Parallel)")
    print("=" * 50)
    print(f"Models: {', '.join(model_list)} (running in parallel)")
    print(f"Parsers: {', '.join(parser_list)} (sequential per model)")
    if args.task_ids:
        print(f"Task IDs: {args.task_ids}")
    else:
        print(f"Sites: {args.sites}")
    print(f"Total runs: {total_runs}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 50)
    print()

    def get_model_id(model_name: str, provider: str, model_suffix: str) -> str:
        """Build the full model ID based on provider."""
        if provider == "bedrock":
            # Fail fast if neither --account nor the AWS_ACCOUNT_ID env var is set
            account = args.account or os.environ["AWS_ACCOUNT_ID"]
            return f"arn:aws:bedrock:{args.region}:{account}:inference-profile/{model_suffix}"
        else:  # azure_openai
            return model_suffix  # deployment name

    if args.dry_run:
        # Dry run: just print commands
        for model_name in model_list:
            provider, model_suffix, extra_args = MODELS[model_name]
            model_id = get_model_id(model_name, provider, model_suffix)

            for parser_name in parser_list:
                parser_path = PARSERS[parser_name]
                output_dir = f"{args.output_dir}/{model_name}_{parser_name}"

                config = RunConfig(
                    model_name=model_name,
                    provider=provider,
                    model_id=model_id,
                    parser_name=parser_name,
                    parser_path=parser_path,
                    output_dir=output_dir,
                    region=args.region,
                    max_concurrent=args.max_concurrent,
                    sites=args.sites,
                    tasks_dir=args.tasks_dir,
                    max_tokens=args.max_tokens,
                    task_ids=args.task_ids,
                    only_failed=args.only_failed,
                    extra_args=extra_args,
                )

                print(f"[{model_name}_{parser_name}] Would run:")
                print(" ".join(build_command(config)))
                print()
        sys.exit(0)

    # Run models in parallel, parsers sequential within each model
    all_results = []

    with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
        futures = {}

        for model_name in model_list:
            provider, model_suffix, extra_args = MODELS[model_name]
            model_id = get_model_id(model_name, provider, model_suffix)

            future = executor.submit(
                run_model_experiments,
                model_name,
                provider,
                model_id,
                extra_args,
                parser_list,
                args,
            )
            futures[future] = model_name

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"[{model_name}] All parsers completed")
            except Exception as e:
                print(f"[{model_name}] FAILED with exception: {e}")
                all_results.append((model_name, False, str(e)))

    # Summary
    failed_runs = [(name, msg) for name, success, msg in all_results if not success]
    successful_runs = [(name, msg) for name, success, msg in all_results if success]

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    if successful_runs:
        print(f"Successful ({len(successful_runs)}):")
        for name, output_dir in successful_runs:
            print(f"  - {name} -> {output_dir}")

    if failed_runs:
        print(f"Failed ({len(failed_runs)}):")
        for name, msg in failed_runs:
            print(f"  - {name}: {msg}")

    print("=" * 50)

    sys.exit(1 if failed_runs else 0)


if __name__ == "__main__":
    main()
