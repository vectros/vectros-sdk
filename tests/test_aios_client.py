import unittest
from unittest.mock import MagicMock, patch

from vectros_sdk import AIOSClient, CerebrumClient
from vectros_sdk.llm.models import LLMResponse
from vectros_sdk.memory.models import MemoryResponse
from vectros_sdk.post.models import PostResponse
from vectros_sdk.storage.models import StorageResponse
from vectros_sdk.tool.models import ToolResponse


class TestAIOSClient(unittest.TestCase):
    """Test unified AIOSClient and CerebrumClient orchestrator."""

    def test_client_initialization_and_alias(self):
        client = AIOSClient(
            base_url="http://kernel:8000",
            agent_name="orchestrator",
            default_llms=[{"name": "gpt-4o", "backend": "openai"}],
        )
        self.assertEqual(client.base_url, "http://kernel:8000")
        self.assertEqual(client.agent_name, "orchestrator")
        self.assertEqual(client.default_llms, [{"name": "gpt-4o", "backend": "openai"}])

        self.assertIs(CerebrumClient, AIOSClient)
        cerebrum = CerebrumClient(base_url="http://localhost:8000", agent_name="cerebrum_agent")
        self.assertEqual(cerebrum.agent_name, "cerebrum_agent")

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_subclient_and_convenience_chat(self, mock_send):
        mock_send.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": "Unified client chat response",
                "finished": True,
            }
        }

        client = AIOSClient(agent_name="chat_agent")
        resp1 = client.llm.chat([{"role": "user", "content": "Hello"}])
        self.assertEqual(resp1.response_message, "Unified client chat response")

        # Top level convenience
        resp2 = client.chat("What is 1+1?", system_prompt="Math tutor")
        self.assertEqual(resp2.response_message, "Unified client chat response")

    @patch("vectros_sdk.memory.api.send_request")
    def test_memory_subclient_and_convenience(self, mock_send):
        mock_send.return_value = {
            "response": {
                "response_class": "memory",
                "success": True,
                "memory_id": "mem_uni_1",
                "search_results": [{"memory_id": "mem_uni_1", "content": "Fact 1", "score": 0.9}],
            }
        }

        client = AIOSClient(agent_name="mem_agent")
        resp1 = client.memory.create("Important fact", metadata={"tag": "vital"})
        self.assertEqual(resp1.memory_id, "mem_uni_1")

        resp2 = client.memory.get("mem_uni_1")
        self.assertTrue(resp2.success)

        resp3 = client.memory.update("mem_uni_1", content="Updated fact")
        self.assertTrue(resp3.success)

        resp4 = client.memory.delete("mem_uni_1")
        self.assertTrue(resp4.success)

        resp5 = client.memory.create_agentic("Cognitive memory")
        self.assertTrue(resp5.success)

        # Top level convenience
        resp6 = client.remember("Short memory")
        self.assertEqual(resp6.memory_id, "mem_uni_1")

        resp7 = client.recall("search query", k=3)
        self.assertEqual(len(resp7.search_results), 1)

    @patch("vectros_sdk.storage.api.send_request")
    def test_storage_subclient(self, mock_send):
        mock_send.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Storage operation ok",
                "finished": True,
            }
        }

        client = AIOSClient(agent_name="storage_agent")
        self.assertTrue(client.storage.mount("/root").finished)
        self.assertTrue(client.storage.create_file("a.txt").finished)
        self.assertTrue(client.storage.create_dir("dir").finished)
        self.assertTrue(client.storage.write_file("a.txt", "content").finished)
        self.assertTrue(client.storage.retrieve_file("query", n=5).finished)
        self.assertTrue(client.storage.rollback_file("a.txt", n=1).finished)
        self.assertTrue(client.storage.share_file("a.txt").finished)

    @patch("vectros_sdk.tool.api.send_request")
    def test_tool_subclient(self, mock_send):
        mock_send.return_value = {
            "response": {
                "response_class": "tool",
                "response_message": "Tool ok",
                "finished": True,
            }
        }

        client = AIOSClient(agent_name="tool_agent")
        resp = client.tool.call([{"name": "test_tool", "parameters": {}}])
        self.assertEqual(resp.response_message, "Tool ok")

        class MockToolCls: pass
        client.tool.register("mock_tool", MockToolCls)
        self.assertEqual(client.tool.get("mock_tool"), MockToolCls)
        self.assertIn("mock_tool", client.tool.list())

    @patch("vectros_sdk.post.api.send_request")
    def test_post_subclient_and_convenience(self, mock_send):
        mock_send.return_value = {
            "response": {
                "response_class": "post",
                "response_message": "Post ok",
                "message_id": "msg_uni_1",
                "messages": [{"id": 1}],
                "finished": True,
            }
        }

        client = AIOSClient(agent_name="post_agent")
        self.assertEqual(client.post.send("bob", "hello").message_id, "msg_uni_1")
        self.assertEqual(len(client.post.receive().messages), 1)
        self.assertTrue(client.post.broadcast("announce").finished)
        self.assertTrue(client.post.publish("news", "headline").finished)
        self.assertTrue(client.post.subscribe("news").finished)

        # Top level convenience
        resp_msg = client.send_message("bob", "Direct message")
        self.assertEqual(resp_msg.message_id, "msg_uni_1")

    def test_agent_subclient(self):
        client = AIOSClient()
        class SampleAgent: pass
        client.agent.register("sample_agent", SampleAgent)
        self.assertEqual(client.agent.get("sample_agent"), SampleAgent)
        self.assertIn("sample_agent", client.agent.list())


if __name__ == "__main__":
    unittest.main()

