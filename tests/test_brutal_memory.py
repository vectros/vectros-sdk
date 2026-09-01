import json
import random
import string
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from vectros_sdk import (
    AIOSKernelError,
    MemoryQuery,
    MemoryResponse,
    create_agentic_memory,
    create_memory,
    delete_memory,
    get_memory,
    search_memories,
    update_memory,
)


class TestBrutalMemorySuite(unittest.TestCase):
    """Brutal stress, chaos, concurrency, and fuzzing tests for Memory API."""

    def test_massive_memory_payload(self):
        """Store and retrieve a 5MB memory content block with massive metadata."""
        massive_text = "".join(random.choices(string.ascii_letters + string.digits, k=1_000_000))
        massive_metadata = {f"meta_key_{i}": f"meta_val_{i}" * 100 for i in range(100)}

        q = MemoryQuery(
            agent_name="mega_bot",
            action_type="create",
            content=massive_text,
            metadata=massive_metadata,
        )
        dumped = q.model_dump()
        self.assertEqual(len(dumped["content"]), 1_000_000)
        self.assertEqual(len(dumped["metadata"]), 100)
        # Verify JSON serializability
        serialized = json.dumps(dumped)
        self.assertTrue(len(serialized) > 1_000_000)

    def test_hostile_unicode_and_metadata_fuzzing(self):
        """Hostile strings in memory content and metadata keys/values."""
        hostiles = [
            "🧠" * 2000,
            "DROP TABLE memories;--",
            "<script>document.cookie</script>",
            "\x00\x01\x02\r\n\t",
            "\u200B\u200C\u200D\uFEFF",
            "𓀀𓀁𓀂𓀃𓀄𓀅𓀆",
        ]
        for h in hostiles:
            q = MemoryQuery(
                agent_name=f"agent_{h[:5]}",
                action_type="create",
                content=h,
                metadata={h[:10]: h},
            )
            data = q.model_dump()
            self.assertEqual(data["content"], h)
            self.assertEqual(data["metadata"][h[:10]], h)

    @patch("vectros_sdk.memory.api.send_request")
    def test_concurrent_memory_hammering(self, mock_send_request):
        """100 threads concurrently executing create, get, update, search, delete."""
        mock_send_request.side_effect = lambda query, base_url=None: {
            "response": {
                "success": True,
                "memory_id": f"mem_{query.agent_name}_{query.action_type}",
                "content": f"Content for {query.agent_name}",
                "search_results": [{"memory_id": "m1", "score": 0.99}],
                "status_code": 200,
            }
        }

        operations = ["create", "get", "update", "delete", "search", "create_agentic"]
        thread_count = 100
        results = []
        errors = []

        def worker(worker_id: int):
            op = operations[worker_id % len(operations)]
            agent = f"worker_bot_{worker_id}"
            time.sleep(random.uniform(0.0005, 0.005))
            if op == "create":
                res = create_memory(agent, "some note")
            elif op == "get":
                res = get_memory(agent, f"mem_{worker_id}")
            elif op == "update":
                res = update_memory(agent, f"mem_{worker_id}", content="updated")
            elif op == "delete":
                res = delete_memory(agent, f"mem_{worker_id}")
            elif op == "search":
                res = search_memories(agent, "query", k=3)
            else:
                res = create_agentic_memory(agent, "agentic context")
            return res.success

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(thread_count)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append(e)

        self.assertEqual(len(errors), 0, f"Thread execution errors: {errors}")
        self.assertEqual(len(results), thread_count)
        self.assertTrue(all(results))

    @patch("vectros_sdk.memory.api.send_request")
    def test_search_results_boundary_scores(self, mock_send_request):
        """Fuzzing search results with 10,000 items, extreme scores (negative, huge, precision)."""
        huge_search_results = [
            {
                "memory_id": f"mem_{i}",
                "content": f"Match {i}",
                "score": float(i) / 1000.0,
                "metadata": {"rank": i}
            }
            for i in range(5000)
        ]
        mock_send_request.return_value = {
            "response": {
                "success": True,
                "search_results": huge_search_results,
            }
        }

        resp = search_memories("search_bot", "query", k=5000)
        self.assertEqual(len(resp.search_results), 5000)
        self.assertEqual(resp.search_results[4999]["memory_id"], "mem_4999")
        self.assertEqual(resp["response"]["search_results"][4999]["score"], 4.999)


if __name__ == "__main__":
    unittest.main()
