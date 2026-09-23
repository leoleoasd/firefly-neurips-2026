"""
Web Agent that uses LLM providers to complete web-based tasks with chain-of-thought reasoning.
"""

import asyncio
import logging
import re
from typing import Any

from omegaconf import DictConfig

from rl_web_agent.env import WebAgentEnv
from rl_web_agent.llm import get_llm_client
from rl_web_agent.utils import format_llm_observation


class Colors:
    """ANSI color codes for terminal output"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"

    @classmethod
    def highlight_action(cls, text: str) -> str:
        """Highlight action text with colors"""
        return f"{cls.BOLD}{cls.CYAN}🤖 ACTION: {cls.YELLOW}{text}{cls.RESET}"

    @classmethod
    def highlight_step(cls, step: int, text: str) -> str:
        """Highlight step information"""
        return f"{cls.BOLD}{cls.BLUE}📍 Step {step}: {cls.RESET}{text}"

    @classmethod
    def highlight_result(cls, text: str, success: bool = True) -> str:
        """Highlight result text"""
        color = cls.GREEN if success else cls.RED
        icon = "✅" if success else "❌"
        return f"{cls.BOLD}{color}{icon} {text}{cls.RESET}"


class WebAgent:
    """
    LLM-powered web agent that can complete tasks using chain-of-thought reasoning.
    """

    def __init__(self, llm_config: DictConfig, agent_config: DictConfig):
        """
        Initialize the web agent with LLM and agent configuration.

        Args:
            llm_config: Configuration for LLM provider
            agent_config: Configuration for agent behavior
        """
        self.llm_config = llm_config
        self.agent_config = agent_config
        self.llm_provider = None
        self.llm_session = None
        self.logger = logging.getLogger(__name__)
        self.max_steps = agent_config.max_steps
        self.vlm_enabled = agent_config.get("vlm", False)

        # Action history for tracking previous actions
        self.action_history = []

        # Load prompt templates
        from rl_web_agent.prompts import load_prompt

        self.system_prompt_template = load_prompt("system_cot")

        if self.vlm_enabled:
            self.logger.info("VLM mode enabled - screenshots will be sent to LLM")

    async def setup(self):
        """Initialize the LLM provider"""
        self.llm_provider = get_llm_client()

    async def close(self):
        """Clean up resources"""
        pass

    async def _initialize_session(self, objective: str) -> None:
        """
        Initialize LLM session with system message containing objective and instructions.

        Args:
            objective: The task objective
        """
        system_message = self.system_prompt_template.format(objective=objective)
        self.llm_session = await self.llm_provider.create_session(system_message)

    def _format_observation(self, observation: dict[str, Any]) -> str:
        """
        Format observation for LLM consumption.

        Args:
            observation: Current page observation

        Returns:
            Formatted observation string
        """
        # Debug: Check observation structure
        self.logger.debug(f"Observation keys: {list(observation.keys())}")
        if "error" in observation and observation["error"]:
            self.logger.warning(f"Observation contains error: {observation['error']}")

        # Build simplified observation representation
        try:
            obs_text = format_llm_observation(observation)
        except Exception as e:
            self.logger.error(f"Error building observation text: {e}")
            self.logger.error(f"Observation content: {observation}")
            raise

        return obs_text

    def _parse_action(self, response: str) -> str:
        """
        Parse the LLM response to extract the JSON action.

        Args:
            response: LLM response containing thought and action

        Returns:
            JSON action string
        """
        # Look for ACTION: line
        action_match = re.search(r"ACTION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if not action_match:
            # Fallback: look for JSON-like patterns
            json_pattern = r'\{[^}]*"action"[^}]*\}'
            json_match = re.search(json_pattern, response)
            if json_match:
                return json_match.group(0)

            # If no clear action found, return terminate
            raise ValueError("No action found in response")

        action_text = action_match.group(1).strip()

        # Try to extract JSON from the action text
        json_pattern = r'\{[^}]*"action"[^}]*\}'
        json_match = re.search(json_pattern, action_text)
        if json_match:
            return json_match.group(0)

        raise ValueError("No action found in response")

    async def run_task(self, env: WebAgentEnv, objective: str, max_steps: int = None) -> dict[str, Any]:
        """
        Run a complete task using the web agent with conversation context.

        Args:
            env: WebAgentEnv instance (already set up)
            objective: Task objective description
            max_steps: Maximum number of steps (overrides default)

        Returns:
            Dictionary with task results including final score, answer, and step count
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not initialized. Call setup() first.")

        max_steps = max_steps or self.max_steps
        step_count = 0

        # Validate VLM configuration
        if self.vlm_enabled:
            env_config = env.config
            if not hasattr(env_config, "screenshot_path") or env_config["screenshot_path"] is None:
                self.logger.warning("VLM mode is enabled but environment.screenshot_path is not configured. Screenshots will not be available.")
                print(f"{Colors.YELLOW}⚠️  Warning: VLM mode enabled but screenshot_path not configured{Colors.RESET}\n")

        # Highlight task start
        print("\n" + "=" * 60)
        print(f"{Colors.BOLD}{Colors.MAGENTA}🚀 STARTING WEB AGENT TASK{Colors.RESET}")
        print(f"{Colors.BOLD}🎯 Objective:{Colors.RESET} {objective}")
        print(f"{Colors.BOLD}📏 Max Steps:{Colors.RESET} {max_steps}")
        if self.vlm_enabled:
            print(f"{Colors.BOLD}👁️  VLM Mode:{Colors.RESET} {Colors.GREEN}Enabled (screenshots will be sent to LLM){Colors.RESET}")
        print("=" * 60 + "\n")

        self.logger.info(f"Starting task: {objective}")

        # Initialize LLM session with system message
        await self._initialize_session(objective)

        # Get initial observation
        observation = await env.observation()

        try:
            for step in range(max_steps):
                step_count += 1
                print(Colors.highlight_step(step_count, "Processing observation"))
                self.logger.info(f"Step {step_count}: Processing observation")

                # Check if task is already terminated
                if observation.get("terminated", False):
                    print(Colors.highlight_result("Task already terminated by environment"))
                    self.logger.info("Task already terminated by environment")
                    break

                # Format observation for LLM
                formatted_observation = self._format_observation(observation)

                # Get LLM response via session
                print(Colors.highlight_step(step_count, "Querying LLM via session"))
                self.logger.info(f"Step {step_count}: Querying LLM via session")

                # Pass screenshot if VLM mode is enabled and screenshot is available
                screenshot_path = None
                if self.vlm_enabled and observation.get("screenshot_path"):
                    screenshot_path = observation["screenshot_path"]
                    self.logger.info(f"Step {step_count}: Including screenshot: {screenshot_path}")

                response = await self.llm_session.chat_complete(user_message=formatted_observation, image_path=screenshot_path)

                self.logger.info(f"LLM Response: {response['content']}")
                if response.get("reasoning_content"):
                    self.logger.debug(f"Reasoning: {response['reasoning_content']}")

                # Parse action from response
                try:
                    action_json = self._parse_action(response["content"])
                except Exception as e:
                    print(Colors.highlight_result(f"Error parsing action from response: {e}", success=False))
                    self.logger.error(f"Error parsing action from response: {e}")
                    self.logger.error(f"Full LLM response: {response}")
                    raise

                # Highlight the action being executed
                print(Colors.highlight_action(action_json))
                self.logger.info(f"Step {step_count}: Executing action: {action_json}")

                # Track action history
                self.action_history.append(action_json)

                # Execute action and get next observation
                observation = await env.step(action_json)

                # Check if task is terminated after step
                if observation["terminated"]:
                    print(Colors.highlight_result("Task terminated"))
                    self.logger.info("Task terminated")
                    break

                # Brief pause to allow page to update
                await asyncio.sleep(0.5)

            # Get final results
            final_observation = await env.observation()
            final_score = final_observation["score"]
            final_answer = final_observation["model_answer"]
            terminated = final_observation["terminated"]

            result = {"success": terminated and final_score > 0.0, "score": final_score, "answer": final_answer, "steps": step_count, "terminated": terminated, "max_steps_reached": step_count >= max_steps}

            # Highlight final results
            success = result["success"]
            print("\n" + "=" * 60)
            print(Colors.highlight_result(f"TASK COMPLETED - {'SUCCESS' if success else 'FAILED'}", success=success))
            print(f"{Colors.BOLD}📊 Final Score:{Colors.RESET} {final_score}")
            print(f"{Colors.BOLD}📝 Final Answer:{Colors.RESET} {final_answer}")
            print(f"{Colors.BOLD}👣 Steps Taken:{Colors.RESET} {step_count}")
            print(f"{Colors.BOLD}🏁 Terminated:{Colors.RESET} {terminated}")
            if step_count >= max_steps:
                print(f"{Colors.YELLOW}⚠️  Max steps reached{Colors.RESET}")
            print("=" * 60 + "\n")

            self.logger.info(f"Task completed: {result}")
            return result

        except Exception as e:
            import traceback

            # Highlight error
            print("\n" + "=" * 60)
            print(Colors.highlight_result("TASK FAILED WITH ERROR", success=False))
            print(f"{Colors.RED}💥 Error: {str(e)}{Colors.RESET}")
            print("=" * 60 + "\n")

            self.logger.error(f"Error during task execution: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "score": 0.0, "answer": f"Error: {str(e)}", "steps": step_count, "terminated": False, "max_steps_reached": False, "error": str(e)}


async def create_web_agent(llm_config: DictConfig, agent_config: DictConfig) -> WebAgent:
    """
    Create and initialize a WebAgent.

    Args:
        llm_config: LLM configuration
        agent_config: Agent behavior configuration

    Returns:
        Initialized WebAgent instance
    """
    agent = WebAgent(llm_config, agent_config)
    await agent.setup()
    return agent
