import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from vectros_sdk import (
    MemoryQuery,
    MemoryResponse,
    create_agentic_memory,
    create_memory,
    delete_memory,
    get_memory,
    search_memories,
    update_memory,
    AIOSKernelError,
)


class TestMemoryModels(unittest.TestCase):
    """Test Task 3.1 & 3.2: MemoryQuery and MemoryResponse models."""

    def test_memory_query_defaults(self):
        q = MemoryQuery(agent_name="bot_1", action_type="create", content="Sample note")
        self.assertEqual(q.query_class, "memory")
        self.assertEqual(q.agent_name, "bot_1")
        self.assertEqual(q.action_type, "create")
        self.assertEqual(q.content, "Sample note")
        self.assertIsNone(q.memory_id)
        self.assertIsNone(q.metadata)

    def test_memory_query_invalid_action_type(self):
        with self.assertRaises(ValidationError):
            MemoryQuery(agent_name="bot_1", action_type="invalid_action")

    def test_memory_response_crud_shape(self):
        resp = MemoryResponse(
            success=True,
            memory_id="mem_123",
            content="Strategy meeting notes",
            metadata={"tags": ["planning", "urgent"]},
        )
        self.assertEqual(resp.response_class, "memory")
        self.assertTrue(resp.success)
        self.assertEqual(resp.memory_id, "mem_123")
        self.assertEqual(resp.content, "Strategy meeting notes")
        self.assertEqual(resp.metadata["tags"], ["planning", "urgent"])
        self.assertIsNone(resp.error)

        # Dual access
        self.assertEqual(resp["memory_id"], "mem_123")
        self.assertEqual(resp["response"]["memory_id"], "mem_123")
        self.assertEqual(resp["response"]["content"], "Strategy meeting notes")

    def test_memory_response_search_shape(self):
        results = [
            {"memory_id": "mem_1", "content": "Note 1", "score": 0.95, "metadata": {"tag": "A"}},
            {"memory_id": "mem_2", "content": "Note 2", "score": 0.88, "metadata": {"tag": "B"}},
        ]
        resp = MemoryResponse(success=True, search_results=results)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.search_results), 2)
        self.assertEqual(resp["search_results"][0]["memory_id"], "mem_1")
        self.assertEqual(resp["response"]["search_results"][1]["score"], 0.88)


class TestMemoryAPIFunctions(unittest.TestCase):
    """Test Tasks 3.3 to 3.8: All 6 Memory API functions."""

    @patch("vectros_sdk.memory.api.send_request")
    def test_create_memory(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "memory_id": "mem_abc123",
                "content": None,
                "metadata": None,
                "error": None,
            }
        }

        metadata = {"tags": ["project-planning", "urgent"], "context": "Q4 strategy meeting"}
        resp = create_memory(
            agent_name="project_bot",
            content="Decided to accelerate feature rollout timeline by 2 weeks",
            metadata=metadata,
        )

        self.assertIsInstance(resp, MemoryResponse)
        self.assertTrue(resp.success)
        self.assertEqual(resp.memory_id, "mem_abc123")
        self.assertEqual(resp["response"]["memory_id"], "mem_abc123")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "create")
        self.assertEqual(called_query.agent_name, "project_bot")
        self.assertEqual(called_query.metadata, metadata)

    @patch("vectros_sdk.memory.api.send_request")
    def test_get_memory(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "memory_id": "mem_abc123",
                "content": "Decided to accelerate rollout",
                "metadata": {"priority": "high"},
                "error": None,
            }
        }

        resp = get_memory(agent_name="project_bot", memory_id="mem_abc123")

        self.assertTrue(resp.success)
        self.assertEqual(resp.content, "Decided to accelerate rollout")
        self.assertEqual(resp.metadata["priority"], "high")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "get")
        self.assertEqual(called_query.memory_id, "mem_abc123")

    @patch("vectros_sdk.memory.api.send_request")
    def test_update_memory(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "memory_id": "mem_abc123",
                "content": None,
                "metadata": None,
                "error": None,
            }
        }

        resp = update_memory(
            agent_name="project_bot",
            memory_id="mem_abc123",
            content="Updated timeline decision",
            metadata={"priority": "critical"},
            base_url="http://custom-kernel:8000",
        )

        self.assertTrue(resp.success)
        self.assertEqual(resp.memory_id, "mem_abc123")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "update")
        self.assertEqual(called_query.content, "Updated timeline decision")
        self.assertEqual(called_query.metadata["priority"], "critical")
        self.assertEqual(mock_send_request.call_args[1]["base_url"], "http://custom-kernel:8000")

    @patch("vectros_sdk.memory.api.send_request")
    def test_delete_memory(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "error": None,
            }
        }

        resp = delete_memory(agent_name="project_bot", memory_id="mem_abc123")

        self.assertTrue(resp.success)
        self.assertIsNone(resp.error)

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "delete")
        self.assertEqual(called_query.memory_id, "mem_abc123")

    @patch("vectros_sdk.memory.api.send_request")
    def test_search_memories(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "search_results": [
                    {
                        "memory_id": "mem_abc123",
                        "content": "Updated timeline decision...",
                        "score": 0.92,
                        "metadata": {"priority": "critical"},
                    }
                ],
                "error": None,
            }
        }

        resp = search_memories(agent_name="project_bot", query="timeline changes", k=3)

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.search_results), 1)
        self.assertEqual(resp.search_results[0]["score"], 0.92)

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "search")
        self.assertEqual(called_query.query, "timeline changes")
        self.assertEqual(called_query.k, 3)

    @patch("vectros_sdk.memory.api.send_request")
    def test_create_agentic_memory(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "memory_id": "mem_agentic_999",
                "content": None,
                "metadata": None,
                "error": None,
            }
        }

        resp = create_agentic_memory(
            agent_name="research_bot",
            content="Breakthrough in protein folding: Achieved 92% accuracy",
            metadata={"system": "scientific_discovery"},
        )

        self.assertTrue(resp.success)
        self.assertEqual(resp.memory_id, "mem_agentic_999")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "create_agentic")
        self.assertEqual(called_query.agent_name, "research_bot")
        self.assertEqual(called_query.metadata["system"], "scientific_discovery")

    @patch("vectros_sdk.memory.api.send_request")
    def test_memory_api_error_propagation(self, mock_send_request):
        mock_send_request.side_effect = AIOSKernelError("Memory node unavailable", status_code=503)

        with self.assertRaises(AIOSKernelError) as ctx:
            get_memory("broken_bot", "mem_404")
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
