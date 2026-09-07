import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vectros_sdk.agent.registry import clear_agent_registry, register_agent
from vectros_sdk.commands.download_agent import download_agent, main as download_agent_main
from vectros_sdk.commands.list_agenthub import list_agenthub_agents, main as list_agenthub_main
from vectros_sdk.commands.list_local_agents import list_local_agents, main as list_local_agents_main
from vectros_sdk.commands.upload_agent import upload_agent, main as upload_agent_main


class MockSampleAgent:
    """A sample registered mock agent."""
    pass


class TestAgentCommands(unittest.TestCase):
    """Test Agent CLI commands and underlying programmatic APIs."""

    def setUp(self):
        clear_agent_registry()

    def tearDown(self):
        clear_agent_registry()

    # 1. list-agenthub-agents tests
    @patch("requests.get")
    def test_list_agenthub_agents_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {
                "name": "academic_researcher",
                "description": ["Conducts literature reviews"],
                "meta": {"author": "aios_lab", "version": "1.0.0"},
            }
        ]
        mock_get.return_value = mock_resp

        agents = list_agenthub_agents("https://app.aios.foundation")
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["name"], "academic_researcher")

        # Test CLI entry point
        with patch("sys.stdout"):
            exit_code = list_agenthub_main(["--agenthub_url", "https://app.aios.foundation"])
            self.assertEqual(exit_code, 0)

    @patch("requests.get")
    def test_list_agenthub_agents_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        agents = list_agenthub_agents("https://app.aios.foundation")
        self.assertEqual(agents, [])

    # 2. list-local-agents tests
    def test_list_local_agents_registry_and_filesystem(self):
        register_agent("mock_agent", MockSampleAgent)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_dir = Path(tmpdir) / "demo_author" / "demo_agent"
            agent_dir.mkdir(parents=True)
            config_data = {
                "name": "demo_agent",
                "description": "Demo local agent",
                "meta": {"author": "demo_author", "version": "0.1.0"},
                "build": {"entry": "entry.py", "module": "DemoAgent"},
            }
            with open(agent_dir / "config.json", "w") as f:
                json.dump(config_data, f)

            agents = list_local_agents(agents_dir=tmpdir)
            names = [a["name"] for a in agents]
            self.assertIn("mock_agent", names)
            self.assertIn("demo_agent", names)

            # Test CLI entry point
            with patch("sys.stdout"):
                exit_code = list_local_agents_main(["--agents_dir", tmpdir])
                self.assertEqual(exit_code, 0)

    # 3. download-agent tests
    @patch("requests.get")
    def test_download_agent_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "name": "math_solver",
            "version": "1.0.0",
            "description": "Solves complex equations",
            "code": "class MathSolver:\n    def run(self, task):\n        return 'solved'\n",
            "config": {
                "name": "math_solver",
                "meta": {"author": "science_team", "version": "1.0.0"},
                "build": {"entry": "entry.py", "module": "MathSolver"},
            },
        }
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            res = download_agent(
                agent_author="science_team",
                agent_name="math_solver",
                agent_version="1.0.0",
                target_dir=tmpdir,
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["agent_name"], "math_solver")

            # Verify downloaded files
            cfg_path = Path(tmpdir) / "config.json"
            entry_path = Path(tmpdir) / "entry.py"
            self.assertTrue(cfg_path.exists())
            self.assertTrue(entry_path.exists())

            with open(entry_path, "r") as f:
                self.assertIn("class MathSolver", f.read())

    @patch("requests.get")
    def test_download_agent_cli(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "name": "coder_agent",
            "code": "print('code')",
        }
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.stdout"):
                exit_code = download_agent_main([
                    "--agent_author", "coder_corp",
                    "--agent_name", "coder_agent",
                    "--target_dir", tmpdir,
                ])
                self.assertEqual(exit_code, 0)

    # 4. upload-agent tests
    @patch("requests.post")
    def test_upload_agent_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"status": "published", "agent_id": "agent_123"}
        mock_post.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_dir = Path(tmpdir) / "my_agent"
            agent_dir.mkdir()
            config_data = {
                "name": "my_agent",
                "description": "My custom agent",
                "meta": {"author": "developer", "version": "0.0.1"},
                "build": {"entry": "entry.py", "module": "MyAgent"},
            }
            with open(agent_dir / "config.json", "w") as f:
                json.dump(config_data, f)
            with open(agent_dir / "entry.py", "w") as f:
                f.write("class MyAgent:\n    pass\n")

            res = upload_agent(str(agent_dir), agenthub_url="https://app.aios.foundation")
            self.assertTrue(res["success"])
            self.assertEqual(res["agent_name"], "my_agent")

            # Test CLI entry point
            with patch("sys.stdout"):
                exit_code = upload_agent_main([
                    "--agent_path", str(agent_dir),
                    "--agenthub_url", "https://app.aios.foundation",
                ])
                self.assertEqual(exit_code, 0)

    def test_upload_agent_invalid_path_or_missing_config(self):
        # Non-existent path
        res1 = upload_agent("/non/existent/path")
        self.assertFalse(res1["success"])
        self.assertIn("not found", res1["error"])

        # Missing config.json
        with tempfile.TemporaryDirectory() as tmpdir:
            res2 = upload_agent(tmpdir)
            self.assertFalse(res2["success"])
            self.assertIn("Missing config.json", res2["error"])


if __name__ == "__main__":
    unittest.main()

