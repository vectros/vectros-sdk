import unittest
from typing import Any, Dict, Union
from unittest.mock import MagicMock, patch

from vectros_sdk import (
    BaseAgent,
    LLMResponse,
    MemoryResponse,
    clear_agent_registry,
    get_agent,
    list_registered_agents,
    register_agent,
)


class MockMathAgent(BaseAgent):
    """A concrete mock agent for testing BaseAgent functionality."""

    def run(self, task: Union[str, Dict[str, Any]]) -> Any:
        prompt = task if isinstance(task, str) else task.get("prompt", "")
        # Recall context, chat, and remember result
        memories = self.recall(prompt, k=2)
        llm_res = self.chat(f"Solve task: {prompt}")
        self.remember(f"Solved {prompt}: {llm_res.response_message}")
        return llm_res.response_message


class TestAgentAPI(unittest.TestCase):
    """Test BaseAgent abstract class, lifecycle helpers, and Agent Registry."""

    def setUp(self):
        clear_agent_registry()

    def tearDown(self):
        clear_agent_registry()

    def test_base_agent_initialization(self):
        agent = MockMathAgent(
            agent_name="math_bot",
            system_prompt="You are a math tutor.",
            tools=[{"name": "calculator"}],
            llm_config=[{"name": "gpt-4o", "backend": "openai"}],
            base_url="http://custom-kernel:8000",
        )
        self.assertEqual(agent.agent_name, "math_bot")
        self.assertEqual(agent.system_prompt, "You are a math tutor.")
        self.assertEqual(agent.tools, [{"name": "calculator"}])
        self.assertEqual(agent.llm_config, [{"name": "gpt-4o", "backend": "openai"}])
        self.assertEqual(agent.base_url, "http://custom-kernel:8000")

    def test_base_agent_default_prompt(self):
        agent = MockMathAgent(agent_name="default_bot")
        self.assertEqual(agent.system_prompt, "You are an AI assistant named default_bot.")
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.llm_config)

    @patch("vectros_sdk.agent.base.llm_chat")
    def test_agent_chat_helper(self, mock_llm_chat):
        mock_llm_chat.return_value = LLMResponse(
            response_message="The square root of 16 is 4.",
            finished=True,
            status_code=200,
        )

        agent = MockMathAgent(agent_name="math_bot", system_prompt="Math expert")
        resp = agent.chat("What is sqrt(16)?")

        self.assertEqual(resp.response_message, "The square root of 16 is 4.")
        mock_llm_chat.assert_called_once_with(
            agent_name="math_bot",
            messages=[
                {"role": "system", "content": "Math expert"},
                {"role": "user", "content": "What is sqrt(16)?"},
            ],
            base_url=agent.base_url,
            llms=None,
        )

    @patch("vectros_sdk.agent.base.create_memory")
    def test_agent_remember_helper(self, mock_create_memory):
        mock_create_memory.return_value = MemoryResponse(
            success=True,
            memory_id="mem_100",
        )

        agent = MockMathAgent(agent_name="math_bot")
        resp = agent.remember("Formula: a^2 + b^2 = c^2", metadata={"topic": "geometry"})

        self.assertTrue(resp.success)
        self.assertEqual(resp.memory_id, "mem_100")
        mock_create_memory.assert_called_once_with(
            agent_name="math_bot",
            content="Formula: a^2 + b^2 = c^2",
            metadata={"topic": "geometry"},
            base_url=agent.base_url,
        )

    @patch("vectros_sdk.agent.base.search_memories")
    def test_agent_recall_helper(self, mock_search_memories):
        mock_search_memories.return_value = MemoryResponse(
            success=True,
            search_results=[{"memory_id": "mem_1", "content": "Pythagorean theorem", "score": 0.9}],
        )

        agent = MockMathAgent(agent_name="math_bot")
        resp = agent.recall("geometry formulas", k=3)

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.search_results), 1)
        mock_search_memories.assert_called_once_with(
            agent_name="math_bot",
            query="geometry formulas",
            k=3,
            base_url=agent.base_url,
        )

    @patch("vectros_sdk.agent.base.search_memories")
    @patch("vectros_sdk.agent.base.llm_chat")
    @patch("vectros_sdk.agent.base.create_memory")
    def test_agent_run_pipeline(self, mock_create, mock_chat, mock_search):
        mock_search.return_value = MemoryResponse(success=True, search_results=[])
        mock_chat.return_value = LLMResponse(response_message="42", finished=True)
        mock_create.return_value = MemoryResponse(success=True, memory_id="mem_done")

        agent = MockMathAgent(agent_name="math_bot")
        res = agent.run("Calculate answer to universe")

        self.assertEqual(res, "42")
        mock_search.assert_called_once()
        mock_chat.assert_called_once()
        mock_create.assert_called_once()

    def test_agent_registry_operations(self):
        self.assertEqual(len(list_registered_agents()), 0)
        self.assertIsNone(get_agent("math_bot"))

        register_agent("math_bot", MockMathAgent)
        self.assertEqual(len(list_registered_agents()), 1)
        self.assertEqual(get_agent("math_bot"), MockMathAgent)

        clear_agent_registry()
        self.assertEqual(len(list_registered_agents()), 0)


if __name__ == "__main__":
    unittest.main()

