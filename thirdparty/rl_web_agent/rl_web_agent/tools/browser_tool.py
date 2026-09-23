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

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Optional
from uuid import uuid4

import hydra

from rl_web_agent.env import WebAgentEnv
from rl_web_agent.utils import format_llm_observation
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BrowserTool(BaseTool):
    """A browser tool for executing web browser actions using WebAgentEnv.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a browser environment instance for a trajectory.
    - `execute`: execute the browser action and return observation.
    - `calc_reward`: calculate the reward from WebAgentEnv evaluation.
    - `release`: release the browser environment instance.

    Uses a class-level semaphore to limit parallel browser operations
    (both environment launches and action executions).
    """

    # Class-level semaphore shared across all BrowserTool instances
    _semaphore: ClassVar[Optional[asyncio.Semaphore]] = None

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Initialize class-level semaphore if not already initialized
        max_parallel = config.get("max_parallel", 2)

        if BrowserTool._semaphore is None:
            BrowserTool._semaphore = asyncio.Semaphore(max_parallel)
            logger.info(f"Initialized browser semaphore with limit: {max_parallel}")

        # Load Hydra config for WebAgentEnv
        # Use initialize_config_dir to load from workspace root config.yaml
        workspace_root = Path.cwd()
        config_path = workspace_root / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # Initialize Hydra with the config directory
        with hydra.initialize_config_dir(config_dir=str(workspace_root), version_base=None):
            self.hydra_cfg = hydra.compose(config_name="config")

        logger.info(f"BrowserTool initialized with config from {config_path}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, task_config: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a browser tool instance with WebAgentEnv.

        Args:
            instance_id: The instance id of the tool.
            task_config: The WebArena task configuration.

        Returns:
            The instance id and tool creation response with initial observation.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        if task_config is None:
            raise ValueError("task_config is required")
        task_config = json.loads(task_config)

        # Use semaphore to limit parallel browser operations
        async with self._semaphore:
            logger.info(f"BrowserTool creating environment: {instance_id=}")

            # Create WebAgentEnv instance
            env = WebAgentEnv(self.hydra_cfg.environment)

            # Setup the environment with task config
            observation = await env.setup(task_config)

            # Get initial observation
            # observation = await env.observation(skip_evaluation=True)
            formatted_obs = format_llm_observation(observation)

            self._instance_dict[instance_id] = {
                "env": env,
                "task_config": task_config,
                "action_history": [],
                "last_observation": observation,
                "reward": 0.0,
            }

            logger.info(f"BrowserTool instance created: {instance_id=}")

            # Return initial observation in ToolResponse
            return instance_id, ToolResponse(text=formatted_obs)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute a browser action using WebAgentEnv.

        Args:
            instance_id: The instance id of the tool.
            parameters: The browser action parameters.

        Returns:
            ToolResponse with observation, reward score, and metrics.
        """
        if instance_id not in self._instance_dict:
            error_msg = f"Instance {instance_id} not found. Call create() first."
            logger.error(error_msg)
            return ToolResponse(text=error_msg), 0.0, {"error": "instance_not_found"}

        instance = self._instance_dict[instance_id]
        env = instance["env"]

        # Build action dict from parameters (use direct key access)
        action_dict = {"action": parameters["action"]}

        # Add optional parameters if present
        if "target" in parameters:
            action_dict["target"] = parameters["target"]
        if "text" in parameters:
            action_dict["text"] = parameters["text"]
        if "enter" in parameters:
            action_dict["enter"] = parameters["enter"]
        if "value" in parameters:
            action_dict["value"] = parameters["value"]
        if "key" in parameters:
            action_dict["key"] = parameters["key"]
        if "url" in parameters:
            action_dict["url"] = parameters["url"]
        if "tab_id" in parameters:
            action_dict["tab_id"] = parameters["tab_id"]
        if "answer" in parameters:
            action_dict["answer"] = parameters["answer"]

        # Store action in history
        instance["action_history"].append(action_dict)

        logger.info(f"BrowserTool execute requested: {instance_id=} action={action_dict['action']}")

        # Use semaphore to limit parallel browser operations
        async with self._semaphore:
            logger.info(f"BrowserTool executing: {instance_id=} action={action_dict}")

            # Execute action in environment
            action_json = json.dumps(action_dict)
            observation = await env.step(action_json)

            # Store observation for reward calculation
            instance["last_observation"] = observation
            instance["reward"] = observation["score"]

            # Format observation for LLM
            formatted_obs = format_llm_observation(observation)

            # Calculate tool reward (0.0 for intermediate steps, actual score for terminal)
            tool_reward = 0.0
            metrics = {
                "terminated": observation["terminated"],
                "score": observation["score"],
                "error": observation.get("error", None),
            }

            logger.info(f"BrowserTool executed: {instance_id=} terminated={observation['terminated']} score={observation['score']}")

            return ToolResponse(text=formatted_obs), tool_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward from WebAgentEnv evaluation.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The evaluation score from WebAgentEnv (0.0 to 1.0).
        """
        if instance_id not in self._instance_dict:
            logger.warning(f"Instance {instance_id} not found for reward calculation")
            return 0.0

        instance = self._instance_dict[instance_id]
        reward = instance["reward"]

        logger.info(f"BrowserTool calc_reward: {instance_id=} {reward=}")
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the browser tool instance and close WebAgentEnv.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            logger.info(f"BrowserTool releasing instance: {instance_id}")
            instance = self._instance_dict[instance_id]
            env = instance["env"]

            # Close the environment (no semaphore needed - cleanup operation)
            await env.close()

            # Remove from instance dict
            del self._instance_dict[instance_id]
            logger.info(f"BrowserTool instance released: {instance_id}")
        else:
            logger.warning(f"BrowserTool release called for unknown instance: {instance_id}")
