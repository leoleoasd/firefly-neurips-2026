#!/usr/bin/env python3
"""
Script to analyze task failures from batch_summary_with_config.json using AWS Bedrock.

Provides string reasons for why each task failed, or 'n/a' if the reason cannot be determined.

PREREQUISITES:
- AWS credentials configured (AWS CLI: `aws configure`, environment variables, or IAM role)
- boto3 package installed: `pip install boto3`
- Access to AWS Bedrock Claude models (enable in AWS Bedrock console)
- Default model: anthropic.claude-3-sonnet-20240229-v1:0
- Default region: us-east-1

USAGE:
    python analyze_failures.py

    Run from project root directory. The script will:
    1. Load data from results/batch_summary_with_config.json
    2. Identify all failed tasks (success = false)
    3. Analyze each failure using AWS Bedrock Claude
    4. Generate concise reasons for why each task failed
    5. Save results to results/failure_analysis.json

OUTPUT FORMAT:
    {
      "analysis_summary": {
        "total_tasks": 187,
        "failed_tasks": 120,
        "analyzed_tasks": 120
      },
      "failure_analysis": {
        "task_id": {
          "intent": "Task objective",
          "success": false,
          "score": 0.0,
          "failure_reason": "Reason why it failed or 'n/a'"
        }
      }
    }

CONFIGURATION:
- Model: Change model in create_bedrock_config()
- Region: Change region_name in gen_config
- Temperature: Adjust for creativity (default: 0.1)
- Max Tokens: Adjust response length (default: 500)

TROUBLESHOOTING:
- AWS credentials: Run `aws sts get-caller-identity` to test
- Import errors: Script falls back to direct boto3 implementation
- Model access: Enable Claude models in AWS Bedrock console
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import webarena dependencies, fallback to direct boto3 if needed
try:
    # Add the thirdparty/webarena directory to Python path for imports
    webarena_path = Path(__file__).parent / "thirdparty" / "webarena"
    if webarena_path.exists():
        sys.path.insert(0, str(webarena_path))

    from llms.providers.claude_bedrock_utils import generate_from_bedrock_claude_chat

    WEBARENA_AVAILABLE = True
    logger.info("Using WebArena LLM utilities")
except ImportError as e:
    logger.warning(f"WebArena LLM utilities not available: {e}")
    logger.info("Falling back to direct boto3 implementation")
    WEBARENA_AVAILABLE = False

    try:
        from dataclasses import dataclass

        import boto3

        @dataclass
        class LMConfig:
            provider: str
            model: str
            mode: str
            gen_config: dict[str, Any]

    except ImportError:
        logger.error("boto3 not available. Please install with: pip install boto3")
        sys.exit(1)


def generate_bedrock_response_direct(messages, model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.1, max_tokens=500, top_p=0.9, region_name="us-east-1"):
    """Direct implementation using boto3 if webarena utilities are not available."""
    try:
        bedrock = boto3.client("bedrock-runtime", region_name=region_name)

        # Convert messages to Bedrock format
        bedrock_messages = []
        system_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_messages.append(msg["content"])
            else:
                bedrock_messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

        inference_config = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens,
        }

        request_body = {
            "modelId": model_id,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config,
        }

        if system_messages:
            request_body["system"] = [{"text": system_messages[0]}]

        response = bedrock.converse(**request_body)
        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        logger.error(f"Error calling Bedrock directly: {e}")
        return "n/a"


def create_bedrock_config() -> LMConfig:
    """Create configuration for AWS Bedrock Claude."""
    return LMConfig(
        provider="bedrock",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        mode="chat",
        gen_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "context_length": 4000,
            "max_tokens": 500,
            "region_name": "us-east-1",
            "stop_token": None,
        },
    )


def create_analysis_prompt(task_data: dict[str, Any]) -> str:
    """Create a prompt for analyzing why a task failed."""
    intent = task_data.get("intent", "Unknown")
    answer = task_data.get("answer", "No answer provided")
    eval_info = task_data.get("eval", {})

    # Extract reference information
    reference_answers = eval_info.get("reference_answers", {})
    must_include = reference_answers.get("must_include", [])
    fuzzy_match = reference_answers.get("fuzzy_match", [])
    string_note = eval_info.get("string_note", "")

    prompt = f"""You are analyzing why a web automation task failed. The task was to complete an objective on a website, and the AI agent provided an answer that was marked as incorrect.

TASK OBJECTIVE: {intent}

AGENT'S ANSWER: {answer}

EVALUATION CRITERIA:
- Must include elements: {must_include if must_include else "None specified"}
- Fuzzy match criteria: {fuzzy_match if fuzzy_match else "None specified"}
- Additional notes: {string_note if string_note else "None"}

Based on this information, provide a concise reason (1-2 sentences) explaining why the task likely failed. Focus on:
1. Missing required information
2. Incorrect information provided
3. Incomplete task execution
4. Misunderstanding of the objective

If you cannot determine a clear reason from the available information, respond with "n/a".

Reason:"""

    return prompt


def analyze_task_failure(task_data: dict[str, Any], config: LMConfig) -> str:
    """Analyze a single task failure and return the reason."""
    try:
        prompt = create_analysis_prompt(task_data)

        messages = [{"role": "user", "content": prompt}]

        if WEBARENA_AVAILABLE:
            response = generate_from_bedrock_claude_chat(
                messages=messages,
                model_id=config.model,
                temperature=config.gen_config["temperature"],
                max_tokens=config.gen_config["max_tokens"],
                top_p=config.gen_config["top_p"],
                context_length=config.gen_config["context_length"],
                region_name=config.gen_config["region_name"],
                stop_token=config.gen_config["stop_token"],
            )
        else:
            response = generate_bedrock_response_direct(
                messages=messages,
                model_id=config.model,
                temperature=config.gen_config["temperature"],
                max_tokens=config.gen_config["max_tokens"],
                top_p=config.gen_config["top_p"],
                region_name=config.gen_config["region_name"],
            )

        # Clean up the response
        reason = response.strip()
        if reason.lower().startswith("reason:"):
            reason = reason[7:].strip()

        return reason

    except Exception as e:
        logger.error(f"Error analyzing task: {e}")
        return "n/a"


def check_aws_credentials() -> bool:
    """Check if AWS credentials are available."""
    try:
        import boto3

        # Try to create a client to test credentials
        boto3.client("sts").get_caller_identity()
        return True
    except Exception as e:
        logger.error(f"AWS credentials not available: {e}")
        return False


def main():
    """Main function to analyze all failed tasks."""
    # Check AWS credentials
    if not check_aws_credentials():
        logger.error("Please configure AWS credentials using AWS CLI, environment variables, or IAM role")
        sys.exit(1)

    # Load the batch summary data
    data_file = Path("results/batch_summary_with_config.json")
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Make sure you're running this script from the project root directory")
        sys.exit(1)

    logger.info(f"Loading data from {data_file}")
    with open(data_file) as f:
        data = json.load(f)

    # Create Bedrock configuration
    config = create_bedrock_config()
    logger.info("Initialized AWS Bedrock configuration")

    # Analyze each task
    results = {}
    failed_tasks = []

    for task_id, task_data in data["results"].items():
        if not task_data.get("success", True):  # Only analyze failed tasks
            failed_tasks.append(task_id)

    logger.info(f"Found {len(failed_tasks)} failed tasks to analyze")

    if not failed_tasks:
        logger.info("No failed tasks found to analyze")
        return

    for i, task_id in enumerate(failed_tasks, 1):
        task_data = data["results"][task_id]
        logger.info(f"Analyzing task {task_id} ({i}/{len(failed_tasks)})")

        reason = analyze_task_failure(task_data, config)
        results[task_id] = {"intent": task_data.get("intent", "Unknown"), "success": task_data.get("success", False), "score": task_data.get("score", 0.0), "failure_reason": reason}

        logger.info(f"Task {task_id}: {reason}")

    # Save results
    output_file = Path("results/failure_analysis.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({"analysis_summary": {"total_tasks": data["total_tasks"], "failed_tasks": len(failed_tasks), "analyzed_tasks": len(results)}, "failure_analysis": results}, f, indent=2)

    logger.info(f"Analysis complete. Results saved to {output_file}")

    # Print summary
    print("\n=== FAILURE ANALYSIS SUMMARY ===")
    print(f"Total tasks: {data['total_tasks']}")
    print(f"Failed tasks: {len(failed_tasks)}")
    print(f"Successfully analyzed: {len(results)}")

    print("\n=== SAMPLE RESULTS ===")
    for task_id, result in list(results.items())[:3]:
        print(f"\nTask {task_id}:")
        print(f"  Intent: {result['intent']}")
        print(f"  Reason: {result['failure_reason']}")


if __name__ == "__main__":
    main()
