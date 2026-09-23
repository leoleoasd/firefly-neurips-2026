#!/usr/bin/env python3
"""
Example usage of the async LLM client with OpenAI and Bedrock providers.
"""

import asyncio
import logging

import hydra
from omegaconf import DictConfig

from rl_web_agent.config_store import ConfigStore
from rl_web_agent.llm import get_llm_client


async def test_completion():
    """Test LLM completion using singleton"""
    print("🔵 Testing LLM completion...")

    async with get_llm_client() as client:
        # Single completion
        from rl_web_agent.prompts import load_prompt

        system_prompt = load_prompt("helpful_assistant")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "What is the capital of France?"}]

        result = await client.complete(messages)
        print(f"Response: {result}")


async def test_with_tools():
    """Test LLM completion with tools"""
    print("🟠 Testing completion with tools...")

    async with get_llm_client() as client:
        # Define a simple tool
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state"}}, "required": ["location"]}}}]

        messages = [{"role": "user", "content": "What's the weather like in Paris?"}]

        try:
            result = await client.complete_with_tools(messages, tools)
            print(f"Tool response: {result}")
        except Exception as e:
            print(f"Tools test failed (may not be supported by provider): {e}")


async def test_concurrent_completions():
    """Test concurrent completions"""
    print("🚀 Testing concurrent completions...")

    async with get_llm_client() as client:
        # Multiple requests
        requests = [{"messages": [{"role": "user", "content": f"Count to {i}"}], "max_tokens": 50} for i in range(1, 6)]

        results = await client.complete_many(requests)

        for i, result in enumerate(results, 1):
            print(f"Request {i}: {result}")


async def run_tests(cfg: DictConfig) -> None:
    """Main test function"""
    logging.basicConfig(level=logging.INFO)

    print("🤖 LLM Client Singleton Example")
    print("=" * 50)
    print(f"Using provider: {cfg.llm.provider}")
    print()

    # Test different functionalities
    try:
        await test_completion()
        print()
        await test_with_tools()
        print()
        await test_concurrent_completions()
        print()
    except Exception as e:
        print(f"Test failed: {e}")

    print("\n✅ Example completed!")


@hydra.main(version_base=None, config_path="../rl_web_agent/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point"""
    # Save config globally for singleton access
    ConfigStore.set(cfg)

    asyncio.run(run_tests(cfg))


if __name__ == "__main__":
    main()
