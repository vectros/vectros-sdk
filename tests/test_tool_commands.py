import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vectros_sdk.commands.download_tool import download_tool, main as download_tool_main
from vectros_sdk.commands.list_local_tools import list_local_tools, main as list_local_tools_main
from vectros_sdk.commands.list_toolhub import list_toolhub_tools, main as list_toolhub_main
from vectros_sdk.commands.upload_tool import upload_tool, main as upload_tool_main
from vectros_sdk.tool.core.registry import clear_registry, register_tool


class MockSampleTool:
    """A sample registered mock tool."""
    pass


class TestToolCommands(unittest.TestCase):
    """Test Tool CLI commands and underlying programmatic APIs."""

    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    # 1. list-toolhub-tools tests
    @patch("requests.get")
    def test_list_toolhub_tools_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {
                "name": "wikipedia",
                "description": ["Search wikipedia pages"],
                "meta": {"author": "wikimedia", "version": "1.0.0"},
            }
        ]
        mock_get.return_value = mock_resp

        tools = list_toolhub_tools("https://app.aios.foundation")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "wikipedia")

        # Test CLI entry point
        with patch("sys.stdout"):
            exit_code = list_toolhub_main(["--toolhub_url", "https://app.aios.foundation"])
            self.assertEqual(exit_code, 0)

    @patch("requests.get")
    def test_list_toolhub_tools_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 502
        mock_resp.text = "Bad Gateway"
        mock_get.return_value = mock_resp

        tools = list_toolhub_tools("https://app.aios.foundation")
        self.assertEqual(tools, [])

    # 2. list-local-tools tests
    def test_list_local_tools_registry_and_filesystem(self):
        register_tool("mock_tool", MockSampleTool)

        with tempfile.TemporaryDirectory() as tmpdir:
            tool_dir = Path(tmpdir) / "demo_author" / "demo_tool"
            tool_dir.mkdir(parents=True)
            config_data = {
                "name": "demo_tool",
                "description": "Demo local tool",
                "meta": {"author": "demo_author", "version": "0.1.0"},
                "build": {"entry": "entry.py", "module": "DemoTool"},
            }
            with open(tool_dir / "config.json", "w") as f:
                json.dump(config_data, f)

            tools = list_local_tools(tools_dir=tmpdir)
            names = [t["name"] for t in tools]
            self.assertIn("mock_tool", names)
            self.assertIn("demo_tool", names)

            # Test CLI entry point
            with patch("sys.stdout"):
                exit_code = list_local_tools_main(["--tools_dir", tmpdir])
                self.assertEqual(exit_code, 0)

    # 3. download-tool tests
    @patch("requests.get")
    def test_download_tool_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "name": "calc_tool",
            "version": "1.2.0",
            "description": "Calculates numbers",
            "code": "class Calc:\n    def run(self, params):\n        return 42\n",
            "config": {
                "name": "calc_tool",
                "meta": {"author": "math_corp", "version": "1.2.0"},
                "build": {"entry": "entry.py", "module": "Calc"},
            },
        }
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            res = download_tool(
                tool_author="math_corp",
                tool_name="calc_tool",
                tool_version="1.2.0",
                target_dir=tmpdir,
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["tool_name"], "calc_tool")

            # Verify downloaded files
            cfg_path = Path(tmpdir) / "config.json"
            entry_path = Path(tmpdir) / "entry.py"
            self.assertTrue(cfg_path.exists())
            self.assertTrue(entry_path.exists())

            with open(entry_path, "r") as f:
                self.assertIn("class Calc", f.read())

    @patch("requests.get")
    def test_download_tool_cli(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "name": "search_tool",
            "code": "print('search')",
        }
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.stdout"):
                exit_code = download_tool_main([
                    "--tool_author", "search_corp",
                    "--tool_name", "search_tool",
                    "--target_dir", tmpdir,
                ])
                self.assertEqual(exit_code, 0)

    # 4. upload-tool tests
    @patch("requests.post")
    def test_upload_tool_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"status": "published", "tool_id": "tool_123"}
        mock_post.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            tool_dir = Path(tmpdir) / "my_tool"
            tool_dir.mkdir()
            config_data = {
                "name": "my_tool",
                "description": "My custom tool",
                "meta": {"author": "developer", "version": "0.0.1"},
                "build": {"entry": "entry.py", "module": "MyTool"},
            }
            with open(tool_dir / "config.json", "w") as f:
                json.dump(config_data, f)
            with open(tool_dir / "entry.py", "w") as f:
                f.write("class MyTool:\n    pass\n")

            res = upload_tool(str(tool_dir), toolhub_url="https://app.aios.foundation")
            self.assertTrue(res["success"])
            self.assertEqual(res["tool_name"], "my_tool")

            # Test CLI entry point
            with patch("sys.stdout"):
                exit_code = upload_tool_main([
                    "--tool_path", str(tool_dir),
                    "--toolhub_url", "https://app.aios.foundation",
                ])
                self.assertEqual(exit_code, 0)

    def test_upload_tool_invalid_path_or_missing_config(self):
        # Non-existent path
        res1 = upload_tool("/non/existent/path")
        self.assertFalse(res1["success"])
        self.assertIn("not found", res1["error"])

        # Missing config.json
        with tempfile.TemporaryDirectory() as tmpdir:
            res2 = upload_tool(tmpdir)
            self.assertFalse(res2["success"])
            self.assertIn("Missing config.json", res2["error"])


if __name__ == "__main__":
    unittest.main()

