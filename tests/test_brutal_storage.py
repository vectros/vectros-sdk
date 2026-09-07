import json
import random
import string
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from vectros_sdk import (
    AIOSKernelError,
    StorageQuery,
    StorageResponse,
    create_dir,
    create_file,
    mount,
    retrieve_file,
    rollback_file,
    share_file,
    write_file,
)


class TestBrutalStorageSuite(unittest.TestCase):
    """Brutal stress, chaos, concurrency, and fuzzing tests for Storage API."""

    def test_massive_file_payload(self):
        """Write and handle a 5MB text payload in StorageQuery and StorageResponse."""
        massive_text = "".join(random.choices(string.ascii_letters + string.digits, k=5_000_000))
        query = StorageQuery(
            agent_name="mega_agent",
            operation_type="write_file",
            params=[{"file_path": "huge_dump.txt", "content": massive_text}],
        )
        dumped = query.model_dump()
        self.assertEqual(len(dumped["params"][0]["content"]), 5_000_000)
        serialized = json.dumps(dumped)
        self.assertTrue(len(serialized) >= 5_000_000)

        # Response handling with large payload
        resp = StorageResponse(
            response_message=massive_text[:100_000],
            finished=True,
            status_code=200,
        )
        self.assertEqual(len(resp.response_message), 100_000)
        self.assertEqual(resp["response"]["response_message"][:10], massive_text[:10])

    def test_hostile_paths_and_fuzzing(self):
        """Hostile strings, path traversal, control chars, and Unicode in paths/queries."""
        hostiles = [
            "../../../../../../etc/shadow",
            "..\\..\\..\\windows\\system32\\config\\SAM",
            "/dev/null\x00extra",
            "📁/📂/📄_unicode_test_🎉.txt",
            "<script>alert('xss')</script>.json",
            "'; DROP TABLE files; --",
            "\u200B\u200C\u200D\uFEFFinvisible.dat",
            "A" * 4096,
        ]

        for h in hostiles:
            query = StorageQuery(
                agent_name=f"agent_{h[:5]}",
                operation_type="create_file",
                params=[{"file_path": h}],
            )
            data = query.model_dump()
            self.assertEqual(data["params"][0]["file_path"], h)
            self.assertEqual(query["params"][0]["file_path"], h)

    @patch("vectros_sdk.storage.api.send_request")
    def test_concurrent_storage_operations(self, mock_send_request):
        """Execute 100 concurrent storage operations across worker threads."""
        def mock_dispatcher(query, base_url=None, timeout=60):
            return {
                "response": {
                    "response_class": "storage",
                    "response_message": f"Processed {query.operation_type} for {query.agent_name}",
                    "finished": True,
                    "status_code": 200,
                }
            }

        mock_send_request.side_effect = mock_dispatcher

        def worker_task(idx):
            ops = [
                lambda: mount(f"agent_{idx}", f"/mnt/drive_{idx}"),
                lambda: create_file(f"agent_{idx}", f"path/file_{idx}.txt"),
                lambda: create_dir(f"agent_{idx}", f"path/dir_{idx}"),
                lambda: write_file(f"agent_{idx}", f"file_{idx}.txt", f"data_{idx}"),
                lambda: retrieve_file(f"agent_{idx}", f"query_{idx}", n=5, keywords=["k1", "k2"]),
                lambda: rollback_file(f"agent_{idx}", f"file_{idx}.txt", n=1),
                lambda: share_file(f"agent_{idx}", f"file_{idx}.txt"),
            ]
            op = random.choice(ops)
            resp = op()
            return resp.finished and resp.status_code == 200

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker_task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 100)
        self.assertTrue(all(results))

    @patch("vectros_sdk.storage.api.send_request")
    def test_malformed_and_raw_responses(self, mock_send_request):
        """Handle raw string, shallow dict, and unconventional kernel outputs safely."""
        # Flat dict response
        mock_send_request.return_value = {
            "response_class": "storage",
            "response_message": "Flat response",
            "finished": True,
            "status_code": 200,
        }
        resp = mount("agent_test", "/tmp")
        self.assertEqual(resp.response_message, "Flat response")
        self.assertTrue(resp.finished)

        # Raw string response
        mock_send_request.return_value = "Kernel plain text response"
        resp2 = create_file("agent_test", "a.txt")
        self.assertEqual(resp2.response_message, "Kernel plain text response")

        # Nested dict with extra custom payload
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Custom payload",
                "custom_token": "xyz123",
                "finished": True,
                "status_code": 200,
            }
        }
        resp3 = share_file("agent_test", "shared.bin")
        self.assertEqual(resp3.custom_token, "xyz123")
        self.assertEqual(resp3["custom_token"], "xyz123")


if __name__ == "__main__":
    unittest.main()

