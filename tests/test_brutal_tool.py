import json
import random
import string
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from vectros_sdk import (
    AIOSKernelError,
    BaseTool,
    ToolQuery,
    ToolResponse,
    call_tool,
    clear_registry,
    get_tool,
    list_registered_tools,
    register_tool,
)


class TestBrutalToolSuite(unittest.TestCase):
    """Brutal stress, chaos, concurrency, and fuzzing tests for Tool API."""

    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    def test_massive_tool_call_payload(self):
        """Handle 5MB tool arguments and deeply nested parameter trees."""
        massive_blob = "".join(random.choices(string.ascii_letters + string.digits, k=1_000_000))
        deep_nesting = {"nested": {f"level_{i}": {"data": massive_blob[:100]} for i in range(100)}}

        query = ToolQuery(
            agent_name="mega_agent",
            tool_calls=[
                {"name": "heavy_tool", "parameters": {"blob": massive_blob, "tree": deep_nesting}}
            ],
        )
        dumped = query.model_dump()
        self.assertEqual(len(dumped["tool_calls"][0]["parameters"]["blob"]), 1_000_000)
        serialized = json.dumps(dumped)
        self.assertTrue(len(serialized) >= 1_000_000)

        # ToolResponse handling with large payload
        resp = ToolResponse(
            response_message=massive_blob[:50_000],
            finished=True,
            status_code=200,
        )
        self.assertEqual(len(resp.response_message), 50_000)
        self.assertEqual(resp["response"]["response_message"][:10], massive_blob[:10])

    def test_hostile_tool_names_and_fuzzing(self):
        """Hostile strings, null bytes, unicode, script injection in tool names and params."""
        hostiles = [
            "tool/\x00/danger",
            "🛠️_super_tool_🚀",
            "<script>eval('evil')</script>",
            "'; DROP TABLE tools; --",
            "\u200B\u200C\u200Dinvisible_tool",
            "A" * 2048,
        ]

        for h in hostiles:
            query = ToolQuery(
                agent_name=f"agent_{h[:5]}",
                tool_calls=[{"name": h, "parameters": {h: h}}],
            )
            data = query.model_dump()
            self.assertEqual(data["tool_calls"][0]["name"], h)
            self.assertEqual(query["tool_calls"][0]["name"], h)

    @patch("vectros_sdk.tool.api.send_request")
    def test_concurrent_tool_execution(self, mock_send_request):
        """Execute 100 concurrent tool requests across worker threads."""
        def mock_dispatcher(query, base_url=None, timeout=60):
            return {
                "response": {
                    "response_class": "tool",
                    "response_message": f"Executed tool calls for {query.agent_name}",
                    "finished": True,
                    "status_code": 200,
                }
            }

        mock_send_request.side_effect = mock_dispatcher

        def worker_task(idx):
            resp = call_tool(
                agent_name=f"agent_{idx}",
                tool_calls=[{"name": f"tool_{idx}", "parameters": {"id": idx}}],
            )
            return resp.finished and resp.status_code == 200

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker_task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 100)
        self.assertTrue(all(results))

    def test_concurrent_registry_hammering(self):
        """Hammer local tool registry with concurrent register/get/list calls."""
        class DynamicTool(BaseTool):
            def get_tool_call_format(self):
                return {}
            def run(self, params):
                return "ok"

        def hammer_task(idx):
            name = f"tool_{idx}"
            register_tool(name, DynamicTool)
            tool = get_tool(name)
            tools = list_registered_tools()
            return tool is DynamicTool and name in tools

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(hammer_task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 100)
        self.assertTrue(all(results))

    @patch("vectros_sdk.tool.api.send_request")
    def test_malformed_and_raw_responses(self, mock_send_request):
        """Handle raw string and shallow dict responses safely."""
        # Flat dict
        mock_send_request.return_value = {
            "response_class": "tool",
            "response_message": "Flat tool result",
            "finished": True,
        }
        resp = call_tool("bot", [{"name": "t"}])
        self.assertEqual(resp.response_message, "Flat tool result")
        self.assertTrue(resp.finished)

        # Raw string
        mock_send_request.return_value = "Kernel plain text tool response"
        resp2 = call_tool("bot", [{"name": "t"}])
        self.assertEqual(resp2.response_message, "Kernel plain text tool response")


if __name__ == "__main__":
    unittest.main()

