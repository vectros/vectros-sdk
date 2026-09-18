"""
SDK.9 — proves the real operator control-plane client
(`vectros_sdk.transport.control_protocol.ControlProtocolClient`, and the
high-level `AIOSClient.control` sub-client wrapping it) drives a real,
standing kernel server's dedicated control socket (CTL.11) end to end: Tier
1 read verbs (CTL.7) and Tier 2 intervene verbs (CTL.8).

No mocks: starts the actual `aiosctl serve-kernel` process built from this
repository and talks to its real control socket over a real Unix socket,
mirroring `test_real_kernel_client.py`'s own harness rather than importing
it, matching this suite's existing convention of self-contained test files.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vectros_sdk.client.client import AIOSClient  # noqa: E402
from vectros_sdk.transport.control_protocol import (  # noqa: E402
    ControlProtocolClient,
    ControlProtocolError,
)

REPO_ROOT = Path(__file__).resolve().parents[1].parent
CARGO_AVAILABLE = shutil.which("cargo") is not None


def _wait_for_socket(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(str(path))
                return
            except OSError:
                pass
        time.sleep(0.1)
    raise TimeoutError(f"kernel server never bound {path}")


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real kernel server")
class TestControlProtocol(unittest.TestCase):
    """Every test here talks to a real `aiosd` kernel's real control socket."""

    @classmethod
    def setUpClass(cls) -> None:
        pid = os.getpid()
        cls.agent_id = f"agent_sdk9_{pid}"
        cls.run_id = f"run_sdk9_{pid}"
        cls.socket_path = Path(f"/tmp/aios-py-sdk9-{pid}.sock")
        cls.control_socket_path = cls.socket_path.with_suffix(".control")
        cls.telemetry_dir = Path(f"/tmp/aios-py-sdk9-{pid}-telemetry")
        cls.socket_path.unlink(missing_ok=True)
        cls.control_socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)
        cls.telemetry_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["AIOS_AGENT_ID"] = cls.agent_id
        env["AIOS_RUN_ID"] = cls.run_id
        cls.server = subprocess.Popen(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "aiosctl",
                "--",
                "serve-kernel",
                str(cls.socket_path),
                str(cls.telemetry_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_socket(cls.socket_path)
            _wait_for_socket(cls.control_socket_path)
        except TimeoutError:
            cls.server.terminate()
            output = cls.server.stdout.read() if cls.server.stdout else ""
            cls.server.wait(timeout=5)
            raise AssertionError(f"server never started; output:\n{output}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
        cls.socket_path.unlink(missing_ok=True)
        cls.control_socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)

    def client(self) -> ControlProtocolClient:
        return ControlProtocolClient(str(self.control_socket_path))

    # Numbered rather than left to unittest's default alphabetical
    # ordering: every test here shares the one agent/run this class
    # bootstraps (the standing server has no runtime "register a fresh
    # agent" verb), so `terminate_*` must run strictly last -- explicit
    # ordering is safer than relying on method names happening to sort
    # correctly.

    def test_01_ping_and_tier1_read_verbs_answer_from_the_real_bootstrapped_tenant(self) -> None:
        client = self.client()
        client.ping()  # raises on any unexpected response

        agents = client.list_agents()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_id"], self.agent_id)
        self.assertEqual(agents[0]["state"], "Ready")

        agent = client.describe_agent(self.agent_id)
        self.assertEqual(agent["agent_id"], self.agent_id)

        runs = client.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], self.run_id)
        self.assertEqual(runs[0]["state"], "Running")

        run = client.describe_run(self.run_id)
        self.assertEqual(run["run_id"], self.run_id)

        depths = client.queue_depths()
        self.assertEqual(depths, {
            "model": 0, "memory": 0, "context": 0,
            "tool": 0, "storage": 0, "ipc": 0,
        })

        usage = client.budget_usage(self.agent_id)
        self.assertIn("reserved", usage)
        self.assertIn("settled", usage)
        self.assertIn("budget", usage)

        self.assertEqual(client.list_requests(), [])

    def test_02_unknown_agent_and_run_are_rejected_not_fabricated(self) -> None:
        client = self.client()
        with self.assertRaises(ControlProtocolError):
            client.describe_agent("agent_never_registered")
        with self.assertRaises(ControlProtocolError):
            client.describe_run("run_never_registered")

    def test_03_high_level_aiosclient_control_subclient_drives_the_same_real_verbs(self) -> None:
        client = AIOSClient(socket_path=str(self.socket_path))
        agents = client.control.list_agents()
        self.assertTrue(any(a["agent_id"] == self.agent_id for a in agents))

        with self.assertRaises(ControlProtocolError):
            client.control.terminate_run(self.run_id)  # confirm defaults to False

    def test_04_suspend_and_resume_run_really_change_the_live_lifecycle(self) -> None:
        client = self.client()
        suspended = client.suspend_run(self.run_id)
        self.assertEqual(suspended["state"], "Quiescing")
        self.assertEqual(client.describe_run(self.run_id)["state"], "Quiescing")

        resumed = client.resume_run(self.run_id)
        self.assertEqual(resumed["state"], "Running")
        self.assertEqual(client.describe_run(self.run_id)["state"], "Running")

    def test_05_terminate_agent_refuses_without_confirm_and_really_terminates_with_it(self) -> None:
        client = self.client()
        with self.assertRaises(ControlProtocolError):
            client.terminate_agent(self.agent_id, confirm=False)

        # Confirm the agent is still real and untouched by the refused call.
        self.assertEqual(client.describe_agent(self.agent_id)["state"], "Ready")

        outcome = client.terminate_agent(self.agent_id, confirm=True)
        self.assertIn(outcome["final_state"], ("Terminated", "Failed"))
        self.assertTrue(outcome["fully_clean"])
        self.assertTrue(outcome["run_also_terminated"])

        self.assertEqual(
            client.describe_agent(self.agent_id)["state"], outcome["final_state"]
        )
