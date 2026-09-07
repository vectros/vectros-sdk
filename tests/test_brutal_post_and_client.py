import json
import random
import string
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from vectros_sdk import (
    AIOSClient,
    AIOSKernelError,
    CerebrumClient,
    PostQuery,
    PostResponse,
    broadcast_post,
    publish_to_topic,
    receive_posts,
    send_post,
    subscribe_topic,
)


class TestBrutalPostAndClientSuite(unittest.TestCase):
    """Brutal stress, chaos, concurrency, and fuzzing tests for Post API and AIOSClient."""

    def test_massive_post_payload(self):
        """Handle 5MB message content and deep nested dictionaries in PostQuery."""
        massive_text = "".join(random.choices(string.ascii_letters + string.digits, k=1_000_000))
        nested_tree = {f"k_{i}": {"data": massive_text[:50]} for i in range(100)}

        query = PostQuery(
            agent_name="mega_sender",
            action_type="send",
            recipient="mega_receiver",
            message={"text": massive_text, "tree": nested_tree},
            metadata={"source": "stress_test"},
        )
        dumped = query.model_dump()
        self.assertEqual(len(dumped["message"]["text"]), 1_000_000)
        serialized = json.dumps(dumped)
        self.assertTrue(len(serialized) >= 1_000_000)

        resp = PostResponse(
            response_message=massive_text[:50_000],
            message_id="msg_mega",
            finished=True,
            status_code=200,
        )
        self.assertEqual(len(resp.response_message), 50_000)
        self.assertEqual(resp["response"]["message_id"], "msg_mega")

    def test_hostile_strings_and_fuzzing(self):
        """Hostile strings, null bytes, unicode, script injection in post fields."""
        hostiles = [
            "agent/\x00/danger",
            "📬_super_post_🚀",
            "<script>alert('post_xss')</script>",
            "'; DROP TABLE mailbox; --",
            "\u200B\u200C\u200Dtopic_stealth",
            "X" * 4096,
        ]

        for h in hostiles:
            query = PostQuery(
                agent_name=f"sender_{h[:5]}",
                recipient=f"recip_{h[:5]}",
                topic=h,
                message=h,
                metadata={h[:10]: h},
            )
            data = query.model_dump()
            self.assertEqual(data["topic"], h)
            self.assertEqual(data["message"], h)

    @patch("vectros_sdk.post.api.send_request")
    def test_concurrent_post_operations(self, mock_send_request):
        """Execute 100 concurrent Post operations across worker threads."""
        def mock_dispatcher(query, base_url=None, timeout=60):
            return {
                "response": {
                    "response_class": "post",
                    "response_message": f"Processed {query.action_type} for {query.agent_name}",
                    "message_id": f"msg_{random.randint(1000, 9999)}",
                    "messages": [{"id": 1, "body": "test"}],
                    "finished": True,
                    "status_code": 200,
                }
            }

        mock_send_request.side_effect = mock_dispatcher

        def worker_task(idx):
            ops = [
                lambda: send_post(f"sender_{idx}", f"recip_{idx}", f"msg_{idx}"),
                lambda: receive_posts(f"agent_{idx}", limit=5),
                lambda: broadcast_post(f"sender_{idx}", f"broadcast_{idx}"),
                lambda: publish_to_topic(f"sender_{idx}", f"topic_{idx}", f"data_{idx}"),
                lambda: subscribe_topic(f"agent_{idx}", f"topic_{idx}"),
            ]
            op = random.choice(ops)
            resp = op()
            return resp.finished and resp.status_code == 200

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker_task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 100)
        self.assertTrue(all(results))

    @patch("vectros_sdk.client.send_request.requests.post")
    def test_concurrent_aios_client_orchestration(self, mock_requests_post):
        """Hammer all AIOSClient subclients concurrently with mocked HTTP backend."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": {
                "response_message": "Kernel unified ok",
                "success": True,
                "finished": True,
                "status_code": 200,
            }
        }
        mock_requests_post.return_value = mock_resp

        client = AIOSClient(base_url="http://mock-kernel:8000", agent_name="hammer_bot")

        def client_task(idx):
            actions = [
                lambda: client.chat(f"Question {idx}"),
                lambda: client.remember(f"Memory {idx}"),
                lambda: client.recall(f"Query {idx}"),
                lambda: client.storage.create_file(f"file_{idx}.txt"),
                lambda: client.tool.call([{"name": f"tool_{idx}", "parameters": {}}]),
                lambda: client.send_message(f"target_{idx}", f"Hello {idx}"),
            ]
            action = random.choice(actions)
            res = action()
            return res is not None

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(client_task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 100)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()

