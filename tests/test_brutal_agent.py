import random
import string
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class BrutalWorkerAgent(BaseAgent):
    """Stress test agent subclass."""

    def run(self, task: Union[str, Dict[str, Any]]) -> Any:
        prompt = str(task)
        mem = self.recall(prompt[:50], k=3)
        chat_resp = self.chat(f"Process: {prompt[:100]}")
        self.remember(f"Result for {prompt[:30]}: {chat_resp.response_message}")
        return chat_resp.response_message


class TestBrutalAgentSuite(unittest.TestCase):
    """Brutal stress, chaos, concurrency, and fuzzing tests for Agent API."""

    def setUp(self):
        clear_agent_registry()

    def tearDown(self):
        clear_agent_registry()

    def test_massive_task_payload_handling(self):
        """Handle multi-MB task payload in BaseAgent execution."""
        massive_prompt = "".join(random.choices(string.ascii_letters + string.digits, k=1_000_000))

        with patch("vectros_sdk.agent.base.search_memories") as mock_search, \
             patch("vectros_sdk.agent.base.llm_chat") as mock_chat, \
             patch("vectros_sdk.agent.base.create_memory") as mock_create:

            mock_search.return_value = MemoryResponse(success=True, search_results=[])
            mock_chat.return_value = LLMResponse(response_message="Processed massive prompt", finished=True)
            mock_create.return_value = MemoryResponse(success=True, memory_id="mem_mega")

            agent = BrutalWorkerAgent(agent_name="mega_agent")
            res = agent.run(massive_prompt)

            self.assertEqual(res, "Processed massive prompt")
            mock_search.assert_called_once()
            mock_chat.assert_called_once()
            mock_create.assert_called_once()

    def test_hostile_agent_configuration_and_fuzzing(self):
        """Fuzz agent initialization with hostile strings, Unicode, null bytes."""
        hostiles = [
            "agent/\x00/danger",
            "🤖_super_agent_🌟",
            "<script>document.cookie</script>",
            "'; DROP TABLE agents; --",
            "\u200B\u200C\u200Dinvisible_agent",
            "A" * 4096,
        ]

        for h in hostiles:
            agent = BrutalWorkerAgent(
                agent_name=h,
                system_prompt=f"System {h}",
                tools=[{"name": h}],
                llm_config=[{"name": h, "backend": h}],
            )
            self.assertEqual(agent.agent_name, h)
            self.assertEqual(agent.system_prompt, f"System {h}")
            self.assertEqual(agent.tools[0]["name"], h)

    def test_concurrent_agent_execution(self):
        """Run 100 concurrent agent workflows across 16 worker threads."""
        with patch("vectros_sdk.agent.base.search_memories") as mock_search, \
             patch("vectros_sdk.agent.base.llm_chat") as mock_chat, \
             patch("vectros_sdk.agent.base.create_memory") as mock_create:

            mock_search.side_effect = lambda **kwargs: MemoryResponse(success=True, search_results=[])
            mock_chat.side_effect = lambda **kwargs: LLMResponse(
                response_message=f"Answer for {kwargs.get('agent_name')}",
                finished=True,
                status_code=200,
            )
            mock_create.side_effect = lambda **kwargs: MemoryResponse(success=True, memory_id="mem_id")

            def worker_task(idx):
                agent = BrutalWorkerAgent(agent_name=f"worker_{idx}")
                res = agent.run(f"Task query #{idx}")
                return res == f"Answer for worker_{idx}"

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(worker_task, i) for i in range(100)]
                results = [f.result() for f in as_completed(futures)]

            self.assertEqual(len(results), 100)
            self.assertTrue(all(results))

    def test_concurrent_agent_registry_hammering(self):
        """Hammer local agent registry concurrently with register/get/list calls."""
        class DynamicAgent(BaseAgent):
            def run(self, task):
                return "ok"

        def hammer_task(idx):
            name = f"agent_{idx}"
            register_agent(name, DynamicAgent)
            agent_cls = get_agent(name)
            agents = list_registered_agents()
            return agent_cls is DynamicAgent and name in agents

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(hammer_task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 100)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()

