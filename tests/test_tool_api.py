import unittest
from typing import Any, Dict
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


class MockCalculatorTool(BaseTool):
    """Simple calculator tool for testing BaseTool abstraction."""

    def get_tool_call_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic math arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expr"],
                },
            },
        }

    def run(self, params: Dict[str, Any]) -> Any:
        expr = params.get("expr", "0")
        return eval(expr)  # safe test mock


class TestToolModels(unittest.TestCase):
    """Test ToolQuery and ToolResponse models."""

    def test_tool_query_defaults(self):
        tool_calls = [{"name": "weather_service/get_forecast", "parameters": {"location": "NYC"}}]
        query = ToolQuery(agent_name="weather_bot", tool_calls=tool_calls)
        self.assertEqual(query.query_class, "tool")
        self.assertEqual(query.agent_name, "weather_bot")
        self.assertEqual(query.tool_calls, tool_calls)
        # Dict subscripting
        self.assertEqual(query["agent_name"], "weather_bot")
        self.assertEqual(query["tool_calls"], tool_calls)

    def test_tool_response_defaults(self):
        resp = ToolResponse(response_message="Tool executed successfully")
        self.assertEqual(resp.response_class, "tool")
        self.assertEqual(resp.response_message, "Tool executed successfully")
        self.assertFalse(resp.finished)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.status_code, 200)

        # Dual access
        self.assertEqual(resp["response_message"], "Tool executed successfully")
        self.assertEqual(resp["response"]["response_message"], "Tool executed successfully")

    def test_tool_response_with_extra_fields(self):
        resp = ToolResponse(
            response_message="Calculated",
            finished=True,
            status_code=200,
            result=42,
        )
        self.assertEqual(resp.result, 42)
        self.assertEqual(resp["result"], 42)
        self.assertEqual(resp["response"]["result"], 42)


class TestBaseToolAndRegistry(unittest.TestCase):
    """Test BaseTool interface and Tool Registry operations."""

    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    def test_base_tool_contract(self):
        tool = MockCalculatorTool()
        schema = tool.get_tool_call_format()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculator")
        res = tool.run({"expr": "2 + 3"})
        self.assertEqual(res, 5)

    def test_tool_registry_operations(self):
        self.assertEqual(len(list_registered_tools()), 0)
        self.assertIsNone(get_tool("calculator"))

        register_tool("calculator", MockCalculatorTool)
        self.assertEqual(len(list_registered_tools()), 1)
        self.assertEqual(get_tool("calculator"), MockCalculatorTool)

        clear_registry()
        self.assertEqual(len(list_registered_tools()), 0)


class TestToolAPIFunctions(unittest.TestCase):
    """Test Tool API execution functions."""

    @patch("vectros_sdk.tool.api.send_request")
    def test_call_tool_success(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "tool",
                "response_message": "Weather is sunny in New York",
                "finished": True,
                "status_code": 200,
            }
        }

        tool_calls = [{
            "name": "weather_service/get_forecast",
            "parameters": {"location": "New York", "units": "metric"},
        }]

        resp = call_tool(
            agent_name="weather_agent",
            tool_calls=tool_calls,
            base_url="http://custom-kernel:8000",
        )

        self.assertIsInstance(resp, ToolResponse)
        self.assertEqual(resp.response_message, "Weather is sunny in New York")
        self.assertEqual(resp["response"]["response_message"], "Weather is sunny in New York")
        self.assertTrue(resp.finished)

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "weather_agent")
        self.assertEqual(called_query.query_class, "tool")
        self.assertEqual(called_query.tool_calls, tool_calls)
        self.assertEqual(mock_send_request.call_args[1]["base_url"], "http://custom-kernel:8000")

    @patch("vectros_sdk.tool.api.send_request")
    def test_call_tool_kernel_error_propagation(self, mock_send_request):
        mock_send_request.side_effect = AIOSKernelError("Tool service crashed", status_code=500)

        with self.assertRaises(AIOSKernelError) as ctx:
            call_tool(
                agent_name="error_agent",
                tool_calls=[{"name": "fail_tool", "parameters": {}}],
            )
        self.assertEqual(ctx.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()

