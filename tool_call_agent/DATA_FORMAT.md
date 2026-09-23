# Tool-Calling Agent Data Format

This document describes the data format for training samples used by the tool-calling agent.

## Code Rules

**FAIL FAST**: Never use `.get()` or `getattr()` for required fields. Use direct access `[]` for dicts and `.` for attributes. Missing required fields should raise immediate errors.

## Overview

Each training sample consists of:
- `index`: Unique identifier for the sample
- `metadata`: Contains all task-related information

The prompt is **dynamically generated** by the `generate.py` script based on the metadata, rather than being pre-computed.

## Data Schema

```python
{
    "index": int,                      # Unique sample identifier
    "metadata": {
        # ============ Required Fields (MUST exist, will crash if missing) ============
        "task_description": str,       # The task/question for the model to solve
        "tools": list[ToolDef],        # Available tools in OpenAI format

        # ============ Optional Fields (checked with "in" before access) ============
        "answer_schema": dict | None,  # JSON Schema for expected answer format
        "system_prompt": str | None,   # Custom system prompt (uses default if not provided)

        # ============ Evaluation Fields (Optional) ============
        "expected_answer": dict | None,           # Ground truth answer for evaluation
        "ground_truth_trajectory": list | None,   # Expected tool call sequence
        "difficulty": str | None,                 # "easy" | "medium" | "hard"
        "task_id": str | None,                    # Original task identifier
    }
}
```

## Tool Definition Format (OpenAI Format)

Tools are defined in OpenAI's function calling format:

```python
ToolDef = {
    "type": "function",
    "function": {
        "name": str,              # Tool name (alphanumeric, underscores, hyphens)
        "description": str,       # What the tool does
        "parameters": {           # JSON Schema for parameters
            "type": "object",
            "properties": {
                "param_name": {
                    "type": str,
                    "description": str,
                    ...
                },
                ...
            },
            "required": list[str]
        }
    }
}
```

## Ground Truth Trajectory Format

For evaluation purposes, you can optionally include the expected tool call sequence:

```python
GroundTruthStep = {
    "tool_name": str,      # Must match tools[].function.name
    "tool_input": dict,    # Expected input arguments
    "tool_output": str,    # Expected output (for simulation/verification)
}
```

## Example

```json
{
    "index": 42,
    "metadata": {
        "task_description": "What is the current weather in Tokyo and convert the temperature from Celsius to Fahrenheit?",
        "answer_schema": {
            "type": "object",
            "properties": {
                "celsius": {"type": "number"},
                "fahrenheit": {"type": "number"},
                "condition": {"type": "string"}
            },
            "required": ["celsius", "fahrenheit", "condition"]
        },
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a specified city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_temperature",
                    "description": "Convert temperature between Celsius and Fahrenheit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "from_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            "to_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["value", "from_unit", "to_unit"]
                    }
                }
            }
        ],
        "expected_answer": {
            "celsius": 22,
            "fahrenheit": 71.6,
            "condition": "sunny"
        },
        "ground_truth_trajectory": [
            {
                "tool_name": "get_weather",
                "tool_input": {"city": "Tokyo"},
                "tool_output": "{\"temperature\": 22, \"condition\": \"sunny\"}"
            },
            {
                "tool_name": "convert_temperature",
                "tool_input": {"value": 22, "from_unit": "celsius", "to_unit": "fahrenheit"},
                "tool_output": "{\"result\": 71.6}"
            }
        ],
        "difficulty": "medium",
        "task_id": "weather_001"
    }
}
```

## Prompt Generation

The `generate.py` script dynamically builds the prompt using the following logic:

1. **System Prompt**: Uses `metadata.system_prompt` if provided, otherwise uses a default system prompt
2. **User Message**: Combines `task_description` with `answer_schema` (if provided)
3. **Chat Template**: Applies the model's chat template to format the conversation

### Default System Prompt

```
You are an AI assistant that can use tools to answer questions.
Use the available tools to gather information and answer the user's question.
When you have enough information, provide your final answer in JSON format.

IMPORTANT: Your final answer MUST be a JSON object matching the expected schema.
Do not include any explanation or text outside the JSON.
```

## File Format

Training data should be stored as JSON Lines (`.jsonl`) with one sample per line:

```
{"index": 0, "metadata": {...}}
{"index": 1, "metadata": {...}}
{"index": 2, "metadata": {...}}
```

Or as a JSON array in a `.json` file:

```json
[
    {"index": 0, "metadata": {...}},
    {"index": 1, "metadata": {...}},
    {"index": 2, "metadata": {...}}
]
```
