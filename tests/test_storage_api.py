import unittest
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


class TestStorageModels(unittest.TestCase):
    """Test StorageQuery and StorageResponse model definitions and behaviors."""

    def test_storage_query_defaults(self):
        query = StorageQuery()
        self.assertEqual(query.query_class, "storage")
        self.assertEqual(query.operation_type, "text")
        self.assertIsNone(query.agent_name)
        self.assertIsNone(query.params)

    def test_storage_query_custom_values(self):
        query = StorageQuery(
            agent_name="fs_agent",
            operation_type="mount",
            params=[{"root_dir": "/tmp/test"}],
        )
        self.assertEqual(query.query_class, "storage")
        self.assertEqual(query.agent_name, "fs_agent")
        self.assertEqual(query.operation_type, "mount")
        self.assertEqual(query.params, [{"root_dir": "/tmp/test"}])
        # Dict subscripting
        self.assertEqual(query["agent_name"], "fs_agent")
        self.assertEqual(query["operation_type"], "mount")
        self.assertEqual(query.get("non_existent", "default"), "default")

    def test_storage_response_defaults(self):
        resp = StorageResponse(response_message="Directory mounted successfully")
        self.assertEqual(resp.response_class, "storage")
        self.assertEqual(resp.response_message, "Directory mounted successfully")
        self.assertFalse(resp.finished)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.status_code, 200)

        # Dual subscript access
        self.assertEqual(resp["response_message"], "Directory mounted successfully")
        self.assertEqual(resp["response"]["response_message"], "Directory mounted successfully")

    def test_storage_response_with_extra_fields(self):
        resp = StorageResponse(
            response_message="Retrieved 2 files",
            finished=True,
            status_code=200,
            matched_files=["/a/b.txt", "/a/c.txt"],
        )
        self.assertEqual(resp.matched_files, ["/a/b.txt", "/a/c.txt"])
        self.assertEqual(resp["matched_files"], ["/a/b.txt", "/a/c.txt"])
        self.assertEqual(resp["response"]["matched_files"], ["/a/b.txt", "/a/c.txt"])


class TestStorageAPIFunctions(unittest.TestCase):
    """Test all 7 Storage API functions."""

    @patch("vectros_sdk.storage.api.send_request")
    def test_mount(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Successfully mounted /data/research_projects",
                "finished": True,
                "status_code": 200,
            }
        }

        resp = mount("research_agent", "/data/research_projects")

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "Successfully mounted /data/research_projects")
        self.assertEqual(resp["response"]["response_message"], "Successfully mounted /data/research_projects")
        self.assertTrue(resp.finished)

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "research_agent")
        self.assertEqual(called_query.operation_type, "mount")
        self.assertEqual(called_query.params, [{"root_dir": "/data/research_projects"}])

    @patch("vectros_sdk.storage.api.send_request")
    def test_create_file(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "File projects/data_analyzer/main.py created",
                "finished": True,
                "status_code": 200,
            }
        }

        resp = create_file("developer_agent", "projects/data_analyzer/main.py")

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "File projects/data_analyzer/main.py created")
        self.assertEqual(resp["response"]["response_message"], "File projects/data_analyzer/main.py created")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "developer_agent")
        self.assertEqual(called_query.operation_type, "create_file")
        self.assertEqual(called_query.params, [{"file_path": "projects/data_analyzer/main.py"}])

    @patch("vectros_sdk.storage.api.send_request")
    def test_create_dir(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Directory projects/new_webapp/src created",
                "finished": True,
                "status_code": 200,
            }
        }

        resp = create_dir("project_manager", "projects/new_webapp/src")

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "Directory projects/new_webapp/src created")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "project_manager")
        self.assertEqual(called_query.operation_type, "create_dir")
        self.assertEqual(called_query.params, [{"dir_path": "projects/new_webapp/src"}])

    @patch("vectros_sdk.storage.api.send_request")
    def test_write_file(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Successfully written to projects/website/index.html",
                "finished": True,
                "status_code": 200,
            }
        }

        html_content = "<html><body>Hello AIOS</body></html>"
        resp = write_file("web_developer", "projects/website/index.html", html_content)

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "Successfully written to projects/website/index.html")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "web_developer")
        self.assertEqual(called_query.operation_type, "write_file")
        self.assertEqual(
            called_query.params,
            [{"file_path": "projects/website/index.html", "content": html_content}],
        )

    @patch("vectros_sdk.storage.api.send_request")
    def test_retrieve_file_with_keywords(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Found 3 matching files",
                "finished": True,
                "status_code": 200,
            }
        }

        keywords = ["pandas", "numpy", "dataframe"]
        resp = retrieve_file(
            agent_name="data_scientist",
            query_text="data processing pipeline",
            n=5,
            keywords=keywords,
            base_url="http://custom-kernel:9000",
        )

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "Found 3 matching files")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "data_scientist")
        self.assertEqual(called_query.operation_type, "retrieve_file")
        self.assertEqual(
            called_query.params,
            [{"query_text": "data processing pipeline", "n": 5, "keywords": keywords}],
        )
        self.assertEqual(mock_send_request.call_args[1]["base_url"], "http://custom-kernel:9000")

    @patch("vectros_sdk.storage.api.send_request")
    def test_retrieve_file_without_keywords(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Found 1 matching file",
                "finished": True,
            }
        }

        resp = retrieve_file(
            agent_name="data_scientist",
            query_text="model config",
            n=1,
        )

        self.assertIsInstance(resp, StorageResponse)
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.params, [{"query_text": "model config", "n": 1}])

    @patch("vectros_sdk.storage.api.send_request")
    def test_rollback_file(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Rolled back config/app_settings.json by 1 version",
                "finished": True,
                "status_code": 200,
            }
        }

        resp = rollback_file("system_agent", "config/app_settings.json", n=1)

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "Rolled back config/app_settings.json by 1 version")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "system_agent")
        self.assertEqual(called_query.operation_type, "rollback_file")
        self.assertEqual(called_query.params, [{"file_path": "config/app_settings.json", "n": 1}])

    @patch("vectros_sdk.storage.api.send_request")
    def test_share_file(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "storage",
                "response_message": "Shared datasets/processed_data.csv successfully",
                "finished": True,
                "status_code": 200,
            }
        }

        resp = share_file("data_engineer", "datasets/processed_data.csv")

        self.assertIsInstance(resp, StorageResponse)
        self.assertEqual(resp.response_message, "Shared datasets/processed_data.csv successfully")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "data_engineer")
        self.assertEqual(called_query.operation_type, "share_file")
        self.assertEqual(called_query.params, [{"file_path": "datasets/processed_data.csv"}])

    @patch("vectros_sdk.storage.api.send_request")
    def test_storage_api_kernel_error_propagation(self, mock_send_request):
        mock_send_request.side_effect = AIOSKernelError("Storage node offline", status_code=503)

        with self.assertRaises(AIOSKernelError) as ctx:
            mount("failing_agent", "/bad/path")
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()

