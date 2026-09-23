# BFCL Codebase Analysis (Internal Reference)

> This document is an internal reference for understanding the Berkeley Function Calling Leaderboard codebase.
> Last updated: March 2026

---

## 1. Project Overview

**Berkeley Function Calling Leaderboard (BFCL)** is a comprehensive evaluation benchmark for testing LLMs' function calling (tool use) capabilities. Developed by UC Berkeley Sky Computing Lab.

**Key URLs:**
- Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- Package: `bfcl-eval` on PyPI

**Version History:**
| Version | Features |
|---------|----------|
| V1 | Simple, parallel, multiple function call evaluation |
| V2 | Enterprise + OSS live data |
| V3 | Multi-turn, multi-step scenarios |
| V4 | Agentic capabilities (web search, memory), format sensitivity |

---

## 2. Directory Structure

```
berkeley-function-call-leaderboard/
├── bfcl_eval/                          # Main package
│   ├── __main__.py                     # CLI entry (Typer-based)
│   ├── _llm_response_generation.py     # Response generation pipeline
│   ├── utils.py                        # Shared utilities
│   │
│   ├── constants/
│   │   ├── category_mapping.py         # Test category definitions
│   │   ├── model_config.py             # Model configs (150+ models)
│   │   ├── enums.py                    # ModelStyle, Language, ReturnFormat
│   │   ├── eval_config.py              # Paths: RESULT_PATH, SCORE_PATH
│   │   └── type_mappings.py            # Type conversion maps
│   │
│   ├── data/                           # Test datasets
│   │   ├── BFCL_v4_*.json              # Test entries
│   │   ├── possible_answer/            # Ground truth
│   │   └── multi_turn_func_doc/        # Multi-turn function docs
│   │
│   ├── eval_checker/
│   │   ├── eval_runner.py              # Main evaluation orchestrator
│   │   ├── eval_runner_helper.py       # Helper functions
│   │   ├── ast_eval/
│   │   │   ├── ast_checker.py          # AST validation
│   │   │   └── type_convertor/         # Java/JS type conversion
│   │   ├── multi_turn_eval/
│   │   │   ├── multi_turn_checker.py   # Multi-turn validation
│   │   │   ├── multi_turn_utils.py     # Execution utilities
│   │   │   └── func_source_code/       # Executable API implementations
│   │   └── agentic_eval/
│   │       └── agentic_checker.py      # Agentic task validation
│   │
│   └── model_handler/
│       ├── base_handler.py             # Abstract base class (993 lines)
│       ├── utils.py                    # convert_to_tool, ast_parse, etc.
│       ├── parser/                     # JSON, XML, Java, JS parsers
│       ├── api_inference/              # 26 API handlers
│       │   ├── openai_completion.py
│       │   ├── openai_response.py
│       │   ├── claude.py
│       │   ├── gemini.py
│       │   └── ...
│       └── local_inference/            # 29 local handlers
│           ├── base_oss_handler.py     # OSS base class
│           ├── llama.py
│           ├── qwen.py
│           └── ...
│
├── result/                             # Generated responses (runtime)
├── score/                              # Evaluation scores (runtime)
├── pyproject.toml                      # Package config
└── uv.lock                             # UV lockfile
```

---

## 3. Class Hierarchy

```
BaseHandler (ABC) - bfcl_eval/model_handler/base_handler.py
    │
    ├── API Handlers (api_inference/)
    │   ├── OpenAICompletionsHandler    # OpenAI Chat Completions API
    │   ├── OpenAIResponsesHandler      # OpenAI Responses API
    │   ├── ClaudeHandler               # Anthropic
    │   ├── GeminiHandler               # Google
    │   ├── MistralHandler
    │   ├── CohereHandler
    │   ├── DeepSeekAPIHandler
    │   ├── GrokHandler
    │   ├── NovaHandler                 # Amazon
    │   └── ... (17 more)
    │
    └── OSSHandler (local_inference/base_oss_handler.py)
        ├── LlamaHandler
        ├── Llama31Handler
        ├── QwenHandler
        ├── QwenFCHandler
        ├── GemmaHandler
        ├── MistralFCHandler
        ├── GraniteHandler
        ├── DeepSeekReasoningHandler
        └── ... (21 more)
```

---

## 4. Core Flow

### 4.1 Generation Flow (`_llm_response_generation.py`)

```
main(args)
    │
    ├─► get_involved_test_entries()     # Parse categories, load entries
    │
    ├─► collect_test_cases()            # Filter existing, handle overwrites
    │
    └─► generate_results()
        │
        ├─► build_handler(model_name)   # Instantiate handler from config
        │
        ├─► [If OSS] spin_up_local_server()
        │
        └─► ThreadPoolExecutor + dependency scheduling
            │
            ├─► multi_threaded_inference()
            │   └─► handler.inference(test_entry)
            │       ├─► inference_single_turn_FC()
            │       ├─► inference_single_turn_prompting()
            │       ├─► inference_multi_turn_FC()
            │       └─► inference_multi_turn_prompting()
            │
            └─► handler.write(result)
```

### 4.2 Evaluation Flow (`eval_runner.py`)

```
main(model, test_categories, ...)
    │
    └─► runner(model_names, test_categories, result_dir, score_dir)
        │
        └─► For each model result file:
            │
            ├─► get_handler(model_name)
            │
            └─► evaluate_task(test_category, ...)
                │
                ├─► load_dataset_entry()
                ├─► load_ground_truth_entry()
                │
                └─► Select runner based on category:
                    ├─► relevance_file_runner()      # irrelevance/relevance
                    ├─► format_sensitivity_runner()  # format_sensitivity
                    ├─► multi_turn_runner()          # multi_turn_*
                    ├─► agentic_runner()             # memory_*, web_search_*
                    └─► ast_file_runner()            # all others
```

---

## 5. BaseHandler Key Methods

```python
class BaseHandler:
    # Entry point - routes to appropriate method
    def inference(test_entry, include_input_log, exclude_state_log):
        if is_fc_model:
            if multi_turn: return inference_multi_turn_FC()
            else: return inference_single_turn_FC()
        else:
            if multi_turn: return inference_multi_turn_prompting()
            else: return inference_single_turn_prompting()

    # Final methods (cannot be overridden)
    @final def inference_single_turn_FC()
    @final def inference_single_turn_prompting()
    @final def inference_multi_turn_FC()      # Complex: handles turns, steps, state
    @final def inference_multi_turn_prompting()
    @final def write()

    # Abstract - MUST implement in subclass
    def _query_FC(inference_data)
    def _pre_query_processing_FC(inference_data, test_entry)
    def _compile_tools(inference_data, test_entry)
    def _parse_query_response_FC(api_response) -> dict
    def add_first_turn_message_FC(inference_data, first_turn_message)
    def _add_next_turn_user_message_FC(inference_data, user_message)
    def _add_assistant_message_FC(inference_data, model_response_data)
    def _add_execution_results_FC(inference_data, results, model_response_data)

    # Same for prompting mode
    def _query_prompting(inference_data)
    def _pre_query_processing_prompting(test_entry)
    def _parse_query_response_prompting(api_response) -> dict
    def add_first_turn_message_prompting(inference_data, first_turn_message)
    # ... etc

    # Decoding
    def decode_ast(result, language, has_tool_call_tag)
    def decode_execute(result, has_tool_call_tag)
```

---

## 6. Test Categories

```python
# From constants/category_mapping.py

NON_LIVE_CATEGORY = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple", "irrelevance"
]

LIVE_CATEGORY = [
    "live_simple", "live_multiple", "live_parallel",
    "live_parallel_multiple", "live_irrelevance", "live_relevance"
]

MULTI_TURN_CATEGORY = [
    "multi_turn_base", "multi_turn_miss_func",
    "multi_turn_miss_param", "multi_turn_long_context"
]

WEB_SEARCH_CATEGORY = ["web_search_base", "web_search_no_snippet"]
MEMORY_CATEGORY = ["memory_kv", "memory_vector", "memory_rec_sum"]
AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY

NON_SCORING_CATEGORY = ["format_sensitivity"]

# Collections
TEST_COLLECTION_MAPPING = {
    "all": ALL_CATEGORIES,
    "all_scoring": ALL_SCORING_CATEGORIES,
    "multi_turn": MULTI_TURN_CATEGORY,
    "single_turn": SINGLE_TURN_CATEGORY,
    "live": LIVE_CATEGORY,
    "non_live": NON_LIVE_CATEGORY,
    "python": [...],
    "non_python": ["simple_java", "simple_javascript"],
    "memory": MEMORY_CATEGORY,
    "web_search": WEB_SEARCH_CATEGORY,
    "agentic": AGENTIC_CATEGORY,
}
```

---

## 7. Data Formats

### 7.1 Test Entry (Input)

```json
{
    "id": "simple_python_0",
    "question": [[
        {"role": "user", "content": "Find the area of a triangle..."}
    ]],
    "function": [{
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle given its base and height.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "The base of the triangle."},
                "height": {"type": "integer", "description": "The height of the triangle."},
                "unit": {"type": "string", "description": "The unit of measure (optional)"}
            },
            "required": ["base", "height"]
        }
    }]
}
```

### 7.2 Multi-Turn Entry (Additional Fields)

```json
{
    "id": "multi_turn_base_42",
    "question": [
        [{"role": "user", "content": "Turn 1 message"}],
        [{"role": "user", "content": "Turn 2 message"}]
    ],
    "function": [...],
    "involved_classes": ["GorillFileSystem", "TradingBot"],
    "initial_config": {"GorillFileSystem": {"root_dir": "/home/user"}},
    "depends_on": ["multi_turn_base_41"]  // Optional dependency
}
```

### 7.3 Ground Truth (possible_answer/)

```json
{
    "id": "simple_python_0",
    "ground_truth": [{
        "calculate_triangle_area": {
            "base": [10],
            "height": [5],
            "unit": ["units", ""]  // "" means optional
        }
    }]
}
```

### 7.4 Result Entry (Output)

```json
{
    "id": "simple_python_0",
    "result": [{"calculate_triangle_area": "{\"base\": 10, \"height\": 5}"}],
    "input_token_count": 150,
    "output_token_count": 30,
    "latency": 0.5,
    "inference_log": [...]  // Optional, if include_input_log=True
}
```

### 7.5 Multi-Turn Result

```json
{
    "id": "multi_turn_base_42",
    "result": [
        [["func1()", "func2()"]],           // Turn 1: list of steps, each step is list of calls
        [["func3()"], ["func4()", "func5()"]]  // Turn 2: 2 steps
    ],
    "input_token_count": [[150, 200], [180]],  // Per turn, per step
    "output_token_count": [[30, 40], [35]],
    "latency": [[0.5, 0.6], [0.7]],
    "inference_log": [...]
}
```

---

## 8. Evaluation Logic

### 8.1 AST Checker (`ast_eval/ast_checker.py`)

```python
def ast_checker(func_description, model_output, possible_answer, language, test_category, model_name):
    if "parallel" in test_category:
        return parallel_function_checker_no_order(...)  # Any order matching
    elif "multiple" in test_category:
        return multiple_function_checker(...)           # One function from options
    else:
        return simple_function_checker(...)             # Single function check

def simple_function_checker():
    # 1. Check function name matches
    # 2. Check required params present
    # 3. Type check each param (Python/Java/JavaScript)
    # 4. Value check (with string standardization)
    # 5. Check optional params
```

**Type Checking:**
- Python: Direct type comparison
- Java/JavaScript: String → type conversion via tree-sitter parsers

**Value Checking:**
- Strings: Case-insensitive, punctuation-normalized
- Lists: Element-wise comparison
- Dicts: Key-value validation

### 8.2 Multi-Turn Checker (`multi_turn_eval/multi_turn_checker.py`)

```python
def multi_turn_checker():
    for turn_index, ground_truth in enumerate(multi_turn_ground_truth_list):
        # 1. Execute model's function calls
        model_results, model_instances = execute_multi_turn_func_call(
            model_responses, initial_config, involved_classes, ...
        )

        # 2. Execute ground truth calls
        gt_results, gt_instances = execute_multi_turn_func_call(
            ground_truth, initial_config, involved_classes, ...
        )

        # 3. State check: compare instance attributes
        state_checker(model_instances, gt_instances)

        # 4. Response check: verify execution results match
        response_checker(all_model_results, gt_results, turn_index)
```

**State Checker:** Compares all non-private attributes (`not startswith('_')`) between model and ground truth instances.

**Response Checker:** Verifies ground truth responses are a subsequence of model responses (unordered).

### 8.3 Relevance/Irrelevance Checker

```python
def relevance_file_runner():
    for entry in model_results:
        try:
            decoded = handler.decode_ast(result, ...)
            contain_func_call = not is_empty_output(decoded)
        except:
            contain_func_call = False

        if "irrelevance" in test_category:
            success = not contain_func_call  # Should NOT output function call
        else:
            success = contain_func_call      # Should output function call
```

---

## 9. Handler Implementation Guide

### 9.1 API Handler Example (OpenAI)

```python
class OpenAICompletionsHandler(BaseHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.client = OpenAI(...)

    def _compile_tools(self, inference_data, test_entry):
        tools = convert_to_tool(test_entry["function"], GORILLA_TO_OPENAPI, self.model_style)
        inference_data["tools"] = tools
        return inference_data

    def _query_FC(self, inference_data):
        return self.client.chat.completions.create(
            messages=inference_data["message"],
            tools=inference_data["tools"],
            model=self.model_name,
            temperature=self.temperature
        )

    def _parse_query_response_FC(self, api_response):
        return {
            "model_responses": [{fc.function.name: fc.function.arguments}
                               for fc in api_response.choices[0].message.tool_calls],
            "tool_call_ids": [fc.id for fc in api_response.choices[0].message.tool_calls],
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model:
            return [{name: json.loads(params)} for item in result for name, params in item.items()]
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)
```

### 9.2 Key Utility Functions (`model_handler/utils.py`)

```python
# Convert function docs to API-specific tool format
convert_to_tool(functions, type_mapping, model_style) -> list[dict]

# Convert model output to executable function call strings
convert_to_function_call(model_output) -> list[str]

# Parse model output to structured format
ast_parse(model_output, language) -> list[dict]

# Add system prompt with function docs for prompting models
system_prompt_pre_processing_chat_model(messages, functions, test_id) -> list[dict]

# Decorator for API retry with exponential backoff
@retry_with_backoff(error_type=RateLimitError, max_retries=5)
```

---

## 10. CLI Commands

```bash
# List models
bfcl models

# List test categories
bfcl test-categories

# Generate responses
bfcl generate \
    --model gpt-4o \
    --test-category simple_python \
    --temperature 0.001 \
    --num-threads 10

# For local models
bfcl generate \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --test-category all \
    --num-gpus 2 \
    --backend vllm \
    --gpu-memory-utilization 0.9

# Evaluate
bfcl evaluate \
    --model gpt-4o \
    --test-category all \
    --partial-eval  # Allow partial results

# View scores
bfcl scores
```

---

## 11. Configuration

### 11.1 Model Config (`constants/model_config.py`)

```python
@dataclass
class ModelConfig:
    model_name: str           # API/HuggingFace model name
    display_name: str         # Leaderboard display name
    url: str                  # Reference URL
    org: str                  # Organization
    license: str              # License type
    model_handler: type       # Handler class
    input_price: float        # USD per million input tokens
    output_price: float       # USD per million output tokens
    is_fc_model: bool         # True = FC mode, False = prompting mode
    underscore_to_dot: bool   # Convert func_name to func.name

MODEL_CONFIG_MAPPING = {
    "gpt-4o-2024-11-20-FC": ModelConfig(
        model_name="gpt-4o-2024-11-20",
        display_name="GPT-4o (2024-11-20) (FC)",
        url="https://openai.com/...",
        org="OpenAI",
        license="Proprietary",
        model_handler=OpenAICompletionsHandler,
        input_price=2.5,
        output_price=10,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
    # ... 150+ models
}
```

### 11.2 Environment Variables (`.env`)

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...

# Local server config
LOCAL_SERVER_ENDPOINT=http://localhost
LOCAL_SERVER_PORT=1053
```

---

## 12. Key Design Patterns

| Pattern | Usage |
|---------|-------|
| **Template Method** | `BaseHandler` defines algorithm skeleton, subclasses implement steps |
| **Strategy** | Different handlers implement different inference strategies |
| **Factory** | `MODEL_CONFIG_MAPPING` creates handlers based on config |
| **Decorator** | `@retry_with_backoff` for API retry logic |
| **Dependency Injection** | Handlers receive config via constructor |
| **Observer** | tqdm progress tracking during generation |

---

## 13. Multi-Turn Execution Backend

### 13.1 Executable APIs (`func_source_code/`)

```
gorilla_file_system.py    # Simulated file system
trading_bot.py            # Stock trading simulation
ticket_api.py             # Ticket booking system
posting_api.py            # Social media posting
math_api.py               # Mathematical operations
vehicle_control.py        # Vehicle control simulation
message_api.py            # Messaging system
travel_api.py             # Travel booking
memory_kv.py              # Key-value memory backend
memory_vector.py          # Vector store memory
memory_rec_sum.py         # Recursive summarization memory
```

### 13.2 Execution Flow

```python
# multi_turn_utils.py
def execute_multi_turn_func_call(func_call_list, initial_config, involved_classes, ...):
    # 1. Initialize instances from initial_config
    instances = {cls: cls(**config) for cls, config in initial_config.items()}

    # 2. Execute each function call
    results = []
    for func_call in func_call_list:
        # Parse: "ClassName.method(args)" or "method(args)"
        class_name, method_name, args = parse_func_call(func_call)
        instance = instances[class_name]
        result = getattr(instance, method_name)(**args)
        results.append(str(result))

    return results, instances
```

---

## 14. Scoring & Leaderboard

### 14.1 Output Files

```
score/
├── {model_name}/
│   ├── BFCL_v4_simple_python_score.json
│   ├── BFCL_v4_multi_turn_base_score.json
│   └── ...
├── data_overall.csv          # Main leaderboard
├── data_non_live.csv         # Non-live category breakdown
├── data_live.csv             # Live category breakdown
├── data_multi_turn.csv       # Multi-turn breakdown
├── data_agentic.csv          # Agentic breakdown
└── data_format_sensitivity.csv
```

### 14.2 Accuracy Calculation

```python
accuracy = correct_count / total_count

# Overall score is weighted average across categories
# Weights defined in eval_runner_helper.py
```

---

## 15. Debugging Tips

1. **Include input log**: `--include-input-log` shows transformed input to model
2. **Check inference log**: Result files contain `inference_log` with step-by-step details
3. **State log**: Multi-turn results include `state_info` showing instance state after each turn
4. **Score file**: Failed entries listed with detailed error messages and types

### Error Types

```
ast_decoder:decoder_failed           # Failed to parse model output
ast_decoder:decoder_wrong_output_format
simple_function_checker:wrong_func_name
simple_function_checker:missing_required
simple_function_checker:unexpected_param
type_error:simple
type_error:nested
value_error:string
value_error:list/tuple
value_error:dict_key
value_error:dict_value
multi_turn:empty_turn_model_response
multi_turn:force_terminated
multi_turn:instance_state_mismatch
multi_turn:execution_response_mismatch
irrelevance_error:decoder_success
relevance_error:decoder_failed
agentic:inference_error
agentic:no_last_message
```

---

## 16. Adding New Models

1. **Create handler** in `model_handler/api_inference/` or `model_handler/local_inference/`
2. **Implement required methods** (see Section 9.1)
3. **Add config** to `MODEL_CONFIG_MAPPING` in `constants/model_config.py`
4. **Test**: `bfcl generate --model {new_model} --test-category simple_python`

---

## 17. Quick Reference

```bash
# Full evaluation pipeline
bfcl generate --model gpt-4o-FC --test-category all
bfcl evaluate --model gpt-4o-FC --test-category all
bfcl scores

# Single category test
bfcl generate --model gpt-4o-FC --test-category simple_python
bfcl evaluate --model gpt-4o-FC --test-category simple_python --partial-eval

# Local model
bfcl generate --model meta-llama/Llama-3.1-8B-Instruct --num-gpus 1 --backend sglang
```

---

*End of document*
