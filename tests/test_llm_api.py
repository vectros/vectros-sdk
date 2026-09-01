import json
import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from vectros_sdk import (
    LLMQuery,
    LLMResponse,
    llm_call_tool,
    llm_chat,
    llm_chat_with_json_output,
    llm_chat_with_tool_call_output,
    llm_operate_file,
    AIOSKernelError,
)


class TestLLMModels(unittest.TestCase):
    """Test Task 2.1 & 2.2: LLMQuery and LLMResponse model definitions and validations."""

    def test_llm_query_defaults(self):
        query = LLMQuery(
            agent_name="agent_1",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(query.query_class, "llm")
        self.assertEqual(query.agent_name, "agent_1")
        self.assertEqual(query.action_type, "chat")
        self.assertEqual(query.message_return_type, "text")
        self.assertEqual(query.tools, [])
        self.assertIsNone(query.llms)
        self.assertIsNone(query.response_format)

    def test_llm_query_invalid_action_type(self):
        with self.assertRaises(ValidationError):
            LLMQuery(
                messages=[{"role": "user", "content": "Hello"}],
                action_type="invalid_action",
            )

    def test_llm_response_defaults(self):
        resp = LLMResponse(response_message="Test output")
        self.assertEqual(resp.response_class, "llm")
        self.assertEqual(resp.response_message, "Test output")
        self.assertFalse(resp.finished)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.status_code, 200)

        # Dual access tests
        self.assertEqual(resp["response_message"], "Test output")
        self.assertEqual(resp["response"]["response_message"], "Test output")

    def test_llm_response_tool_calls_json_string_parsing(self):
        raw_tools = '[{"name": "calculator", "parameters": {"expression": "2+2"}}]'
        resp = LLMResponse(response_message="Calculated", tool_calls=raw_tools)
        self.assertIsInstance(resp.tool_calls, list)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0]["name"], "calculator")


class TestLLMAPIFunctions(unittest.TestCase):
    """Test Tasks 2.3 to 2.7: All 5 LLM API functions."""

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_chat(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": "Python is a high-level programming language.",
                "finished": True,
                "status_code": 200,
            }
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]
        llms = [{"name": "gpt-4o-mini", "backend": "openai"}]

        resp = llm_chat("test_assistant", messages=messages, llms=llms)

        self.assertIsInstance(resp, LLMResponse)
        self.assertEqual(resp.response_message, "Python is a high-level programming language.")
        self.assertEqual(resp["response"]["response_message"], "Python is a high-level programming language.")
        self.assertTrue(resp.finished)

        # Verify query sent
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "chat")
        self.assertEqual(called_query.message_return_type, "text")
        self.assertEqual(called_query.llms, llms)
        self.assertEqual(called_query.agent_name, "test_assistant")

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_chat_with_json_output(self, mock_send_request):
        json_output = '{"keywords": ["AIOS", "operating system"], "summary": "AIOS is an OS for agents."}'
        mock_send_request.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": json_output,
                "finished": True,
                "status_code": 200,
            }
        }

        schema = {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
        }

        resp = llm_chat_with_json_output(
            "json_bot",
            messages=[{"role": "user", "content": "Extract info"}],
            response_format={"type": "json_object", "schema": schema},
        )

        self.assertEqual(resp.response_message, json_output)
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "chat")
        self.assertEqual(called_query.message_return_type, "json")
        self.assertIsNotNone(called_query.response_format)

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_chat_with_tool_call_output(self, mock_send_request):
        tool_calls = [{"name": "scholar_search", "parameters": {"query": "transformers"}}]
        mock_send_request.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": None,
                "tool_calls": tool_calls,
                "finished": True,
                "status_code": 200,
            }
        }

        tools = [{
            "name": "scholar_search",
            "description": "Search academic papers",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }]

        resp = llm_chat_with_tool_call_output(
            "researcher",
            messages=[{"role": "user", "content": "Find papers"}],
            tools=tools,
        )

        self.assertEqual(resp.tool_calls, tool_calls)
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "tool_use")
        self.assertEqual(called_query.tool_choice, "auto")
        self.assertEqual(called_query.tools, tools)

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_call_tool(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": "Weather in NYC is sunny",
                "tool_calls": [{"name": "get_weather", "parameters": {"location": "NYC"}}],
                "finished": True,
            }
        }

        tools = [{"name": "get_weather", "parameters": {"location": {"type": "string"}}}]
        resp = llm_call_tool(
            "weather_agent",
            messages=[{"role": "user", "content": "Weather in NYC"}],
            tools=tools,
        )

        self.assertEqual(resp.response_message, "Weather in NYC is sunny")
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "tool_use")
        self.assertEqual(called_query.tool_choice, "required")

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_operate_file(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": "File notes.txt created successfully.",
                "finished": True,
            }
        }

        resp = llm_operate_file(
            "file_bot",
            messages=[{"role": "user", "content": "Create notes.txt"}],
        )

        self.assertEqual(resp.response_message, "File notes.txt created successfully.")
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "operate_file")

    @patch("vectros_sdk.llm.api.send_request")
    def test_llm_api_kernel_error_propagation(self, mock_send_request):
        mock_send_request.side_effect = AIOSKernelError("Kernel unavailable", status_code=503)

        with self.assertRaises(AIOSKernelError) as ctx:
            llm_chat("error_bot", messages=[{"role": "user", "content": "Hello"}])
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
