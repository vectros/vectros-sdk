import os
import unittest
from unittest.mock import patch, MagicMock
import requests
from pydantic import BaseModel, Field

import vectros_sdk
from vectros_sdk import send_request, aios_kernel_url, Query, Response
from vectros_sdk.client.config import get_kernel_url, set_kernel_url, DEFAULT_AIOS_KERNEL_URL
from vectros_sdk.client.send_request import AIOSKernelError
import vectros_sdk.client
import vectros_sdk.core
import vectros_sdk.llm
import vectros_sdk.memory
import vectros_sdk.storage
import vectros_sdk.tool
import vectros_sdk.tool.core
import vectros_sdk.agent
import vectros_sdk.commands


class TestPackageImports(unittest.TestCase):
    """Test Task 1.1: Package structure and imports."""

    def test_imports(self):
        self.assertIsNotNone(vectros_sdk)
        self.assertIsNotNone(vectros_sdk.client)
        self.assertIsNotNone(vectros_sdk.core)
        self.assertIsNotNone(vectros_sdk.llm)
        self.assertIsNotNone(vectros_sdk.memory)
        self.assertIsNotNone(vectros_sdk.storage)
        self.assertIsNotNone(vectros_sdk.tool)
        self.assertIsNotNone(vectros_sdk.tool.core)
        self.assertIsNotNone(vectros_sdk.agent)
        self.assertIsNotNone(vectros_sdk.commands)

    def test_exposed_symbols(self):
        self.assertTrue(hasattr(vectros_sdk, "send_request"))
        self.assertTrue(hasattr(vectros_sdk, "aios_kernel_url"))
        self.assertTrue(hasattr(vectros_sdk, "Query"))
        self.assertTrue(hasattr(vectros_sdk, "Response"))


class TestConfig(unittest.TestCase):
    """Test Task 1.3: Kernel URL configuration."""

    def test_default_url(self):
        self.assertEqual(DEFAULT_AIOS_KERNEL_URL, "http://localhost:8000")

    def test_env_override(self):
        with patch.dict(os.environ, {"AIOS_KERNEL_URL": "http://custom-kernel:9000"}):
            self.assertEqual(get_kernel_url(), "http://custom-kernel:9000")

    def test_runtime_override(self):
        original_url = vectros_sdk.client.config.aios_kernel_url
        try:
            set_kernel_url("http://runtime-override:8888")
            self.assertEqual(vectros_sdk.client.config.aios_kernel_url, "http://runtime-override:8888")
        finally:
            set_kernel_url(original_url)


class TestModels(unittest.TestCase):
    """Test Tasks 1.4 and 1.5: Base Query and Response models."""

    def test_base_query(self):
        q = Query(query_class="custom", agent_name="agent_1")
        self.assertEqual(q.query_class, "custom")
        self.assertEqual(q.agent_name, "agent_1")
        # Subscripting
        self.assertEqual(q["query_class"], "custom")
        self.assertEqual(q.get("agent_name"), "agent_1")
        self.assertIsNone(q.get("non_existent"))

    def test_query_subclass(self):
        class CustomQuery(Query):
            query_class: str = "custom_child"
            extra_param: int = Field(default=42)

        cq = CustomQuery(agent_name="sub_agent", extra_param=100)
        self.assertEqual(cq.query_class, "custom_child")
        self.assertEqual(cq.extra_param, 100)
        self.assertEqual(cq["extra_param"], 100)

    def test_base_response(self):
        r = Response(
            response_class="base",
            response_message="success",
            finished=True,
            status_code=200
        )
        self.assertEqual(r.response_class, "base")
        self.assertEqual(r.response_message, "success")
        self.assertTrue(r.finished)
        self.assertIsNone(r.error)
        self.assertEqual(r.status_code, 200)

        # Direct subscript access
        self.assertEqual(r["response_message"], "success")

        # Nested response["response"]["response_message"] access pattern from sdk.md
        self.assertEqual(r["response"]["response_message"], "success")
        self.assertTrue(r["response"]["finished"])

    def test_response_subclass(self):
        class ChildResponse(Response):
            response_class: str = "child"
            custom_data: str = Field(default="hello")

        cr = ChildResponse(response_message="done", custom_data="world")
        self.assertEqual(cr.custom_data, "world")
        self.assertEqual(cr["custom_data"], "world")
        self.assertEqual(cr["response"]["custom_data"], "world")


class TestSendRequest(unittest.TestCase):
    """Test Task 1.6: HTTP dispatcher send_request."""

    @patch("requests.post")
    def test_send_request_success_with_query_object(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": {
                "response_class": "llm",
                "response_message": "Hello from mock kernel",
                "finished": True,
                "status_code": 200
            }
        }
        mock_post.return_value = mock_resp

        q = Query(query_class="llm", agent_name="test_bot")
        res = send_request(q, base_url="http://localhost:8000")

        self.assertIn("response", res)
        self.assertEqual(res["response"]["response_message"], "Hello from mock kernel")
        mock_post.assert_called_once_with(
            "http://localhost:8000/query",
            json={"query_class": "llm", "agent_name": "test_bot"},
            headers={"Content-Type": "application/json"},
            timeout=300
        )

    @patch("requests.post")
    def test_send_request_with_dict(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_resp

        res = send_request({"query_class": "ping"}, base_url="http://localhost:8000/query")
        self.assertEqual(res, {"status": "ok"})
        mock_post.assert_called_once_with(
            "http://localhost:8000/query",
            json={"query_class": "ping"},
            headers={"Content-Type": "application/json"},
            timeout=300
        )

    def test_send_request_invalid_query_type(self):
        with self.assertRaises(ValueError):
            send_request("invalid_string_query")

    @patch("requests.post")
    def test_send_request_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.json.return_value = {"error": "Kernel failure"}
        mock_post.return_value = mock_resp

        with self.assertRaises(AIOSKernelError) as ctx:
            send_request({"query_class": "fail"})
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("500", str(ctx.exception))

    @patch("requests.post")
    def test_send_request_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        with self.assertRaises(AIOSKernelError) as ctx:
            send_request({"query_class": "timeout"})
        self.assertIn("timed out", str(ctx.exception))

    @patch("requests.post")
    def test_send_request_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")
        with self.assertRaises(AIOSKernelError) as ctx:
            send_request({"query_class": "conn_error"})
        self.assertIn("Failed to connect", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
