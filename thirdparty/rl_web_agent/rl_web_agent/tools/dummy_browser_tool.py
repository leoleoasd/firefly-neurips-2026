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

import logging
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BrowserTool(BaseTool):
    """A browser tool for executing web browser actions.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the browser action.
    - `calc_reward`: calculate the reward (always 0.0 for now).
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, task_config: Optional[dict] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a browser tool instance.

        Args:
            instance_id: The instance id of the tool.
            task_config: The WebArena task configuration.

        Returns:
            The instance id and tool creation response.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        if task_config is None:
            task_config = kwargs.get("create_kwargs", {}).get("task_config", {})

        self._instance_dict[instance_id] = {
            "task_config": task_config,
            "action_history": [],
            "observation": None,
            "reward": 0.0,
        }
        logger.info(f"BrowserTool instance created: {instance_id=} {task_config=}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute a browser action.

        Args:
            instance_id: The instance id of the tool.
            parameters: The browser action parameters.

        Returns:
            ToolResponse with observation, reward score, and metrics.
        """
        action_type = parameters["action"]
        target = parameters["target"] if "target" in parameters else None
        text = parameters["text"] if "text" in parameters else None
        enter = parameters["enter"] if "enter" in parameters else False
        value = parameters["value"] if "value" in parameters else None
        key = parameters["key"] if "key" in parameters else None
        url = parameters["url"] if "url" in parameters else None
        tab_id = parameters["tab_id"] if "tab_id" in parameters else None
        answer = parameters["answer"] if "answer" in parameters else None

        # Build action dict
        action_dict = {"action": action_type}
        if target is not None:
            action_dict["target"] = target
        if text is not None:
            action_dict["text"] = text
        if enter:
            action_dict["enter"] = enter
        if value is not None:
            action_dict["value"] = value
        if key is not None:
            action_dict["key"] = key
        if url is not None:
            action_dict["url"] = url
        if tab_id is not None:
            action_dict["tab_id"] = tab_id
        if answer is not None:
            action_dict["answer"] = answer

        # Store action in history
        self._instance_dict[instance_id]["action_history"].append(action_dict)

        logger.info(f"BrowserTool executed with {instance_id=} {action_type=} {action_dict=}")

        # Dummy implementation - return placeholder observation
        observation_text = f"Browser action '{action_type}' executed. This is a dummy implementation."
        if action_type == "terminate":
            observation_text = f"Task terminated with answer: {answer or ''}"

        self._instance_dict[instance_id]["observation"] = observation_text

        return ToolResponse(text=observation_text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward - always 0.0 for now (dummy implementation).

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward score (0.0 for dummy implementation).
        """
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the browser tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.info(f"BrowserTool instance released: {instance_id}")
