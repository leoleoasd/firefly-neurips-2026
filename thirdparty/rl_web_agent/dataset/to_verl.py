# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the WebArena dataset to parquet format
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import pandas as pd

from rl_web_agent.prompts import load_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--tasks_dir", default="dataset/train_webarena", help="Directory containing WebArena task JSON files.")
    parser.add_argument("--local_save_dir", default="~/data/webarena", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--sites", default=None, help="Comma-separated list of sites to filter tasks by.")

    args = parser.parse_args()

    data_source = "webarena"

    # Load tool system prompt
    tool_system_prompt_template = load_prompt("tool_system")

    # Load all task JSON files
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    task_files = sorted(glob(str(tasks_dir / "*.json")))
    if not task_files:
        raise ValueError(f"No task JSON files found in {tasks_dir}")

    print(f"Found {len(task_files)} task files in {tasks_dir}")

    # Parse sites filter if provided
    sites_list = None
    if args.sites:
        sites_list = [site.strip() for site in args.sites.split(",")]
        print(f"Filtering tasks by sites: {sites_list}")

    # Process tasks into dataset format
    processed_data = []

    for task_file in task_files:
        with open(task_file) as f:
            task_config = json.load(f)

        # Filter by sites if specified
        if sites_list:
            if len(task_config["sites"]) != 1 or task_config["sites"][0] not in sites_list:
                continue

        task_id = task_config["task_id"]
        intent = task_config["intent"]
        eval_config = task_config["eval"]

        # Format system prompt with objective
        system_prompt = tool_system_prompt_template.format(objective=intent)

        data = {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": "Here is the initial state of the browser. Please start working towards the objective.",
                },
            ],
            "ability": "web_navigation",
            "reward_model": {
                "style": "dummy",
            },
            "extra_info": {
                "task_id": task_id,
                "intent": intent,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "step_browser": {
                        "create_kwargs": {"task_config": json.dumps(task_config)},
                    },
                },
                "task_config": json.dumps(task_config),
            },
        }
        processed_data.append(data)

    # Convert to pandas DataFrame
    df = pd.DataFrame(processed_data)

    # Save to parquet
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    df.to_parquet(os.path.join(local_save_dir, "train.parquet"), index=False)
    if sites_list:
        print(f"Saved {len(processed_data)} WebArena tasks (filtered by sites {sites_list}) to {local_save_dir}/train.parquet")
    else:
        print(f"Saved {len(processed_data)} WebArena tasks to {local_save_dir}/train.parquet")
