"""
SDK.7 -- proves `vectros_sdk.tool.call_tool_with_approval` really drives the
CTL.10 Tier 3 approval flow end to end against a real, standing kernel
server: a deferred call, a callback deciding for real, the decision driven
through the real control socket, and a real retry.

No mocks: starts the actual `aiosctl serve-kernel` process built from this
repository (with `AIOS_REQUIRE_TOOL_APPROVAL=1` for the gated tests, unset
for the ungated one) and talks to it over real Unix sockets, mirroring
`test_real_kernel_client.py`'s own harness rather than importing it.
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
from vectros_sdk.tool import ApprovalRefused, call_tool_with_approval  # noqa: E402

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


def _spawn_server(pid_tag: str, *, require_tool_approval: bool):
    agent_id = f"agent_sdk7_{pid_tag}"
    socket_path = Path(f"/tmp/aios-py-sdk7-{pid_tag}.sock")
    control_socket_path = socket_path.with_suffix(".control")
    telemetry_dir = Path(f"/tmp/aios-py-sdk7-{pid_tag}-telemetry")
    socket_path.unlink(missing_ok=True)
    control_socket_path.unlink(missing_ok=True)
    shutil.rmtree(telemetry_dir, ignore_errors=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["AIOS_AGENT_ID"] = agent_id
    env["AIOS_RUN_ID"] = f"run_sdk7_{pid_tag}"
    if require_tool_approval:
        env["AIOS_REQUIRE_TOOL_APPROVAL"] = "1"
    else:
        env.pop("AIOS_REQUIRE_TOOL_APPROVAL", None)
    server = subprocess.Popen(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "aiosctl",
            "--",
            "serve-kernel",
            str(socket_path),
            str(telemetry_dir),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_socket(socket_path)
        _wait_for_socket(control_socket_path)
    except TimeoutError:
        server.terminate()
        output = server.stdout.read() if server.stdout else ""
        server.wait(timeout=5)
        raise AssertionError(f"server never started; output:\n{output}")
    agent_suffix = agent_id[len("agent_") :]
    return server, socket_path, telemetry_dir, agent_suffix


def _stop_server(server, socket_path: Path, telemetry_dir: Path) -> None:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
    socket_path.unlink(missing_ok=True)
    socket_path.with_suffix(".control").unlink(missing_ok=True)
    shutil.rmtree(telemetry_dir, ignore_errors=True)


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real kernel server")
class TestToolApprovalRealBackend(unittest.TestCase):
    def _tool_calls(self, text: str):
        return [{"name": "uppercase", "parameters": {"input": text}}]

    def test_gate_off_the_callback_is_never_invoked_and_the_call_succeeds_immediately(self) -> None:
        server, socket_path, telemetry_dir, agent = _spawn_server(
            "gate-off", require_tool_approval=False
        )
        try:
            client = AIOSClient(agent_name=agent, socket_path=str(socket_path))
            invoked = []

            def callback(entry):
                invoked.append(entry)
                return True

            response = call_tool_with_approval(client, self._tool_calls("no gate here"), callback)
            self.assertTrue(response.finished)
            self.assertEqual(response.response_message, "NO GATE HERE")
            self.assertEqual(invoked, [], "callback must not run when nothing was deferred")
        finally:
            _stop_server(server, socket_path, telemetry_dir)

    def test_approved_call_really_executes_after_a_real_round_trip(self) -> None:
        server, socket_path, telemetry_dir, agent = _spawn_server(
            "approve", require_tool_approval=True
        )
        try:
            client = AIOSClient(agent_name=agent, socket_path=str(socket_path))
            seen = []

            def callback(entry):
                seen.append(entry)
                return True

            response = call_tool_with_approval(
                client, self._tool_calls("approve me"), callback
            )
            self.assertTrue(response.finished)
            self.assertEqual(response.response_message, "APPROVE ME")
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["agent_id"], f"agent_{agent}")
            self.assertEqual(
                seen[0]["arguments"], [["input", {"String": "approve me"}]]
            )

            # The approval was consumed by that one retry.
            self.assertEqual(client.control.list_pending_approvals(), [])
        finally:
            _stop_server(server, socket_path, telemetry_dir)

    def test_denied_call_raises_approval_refused_and_does_not_execute(self) -> None:
        server, socket_path, telemetry_dir, agent = _spawn_server(
            "deny", require_tool_approval=True
        )
        try:
            client = AIOSClient(agent_name=agent, socket_path=str(socket_path))

            with self.assertRaises(ApprovalRefused):
                call_tool_with_approval(
                    client, self._tool_calls("deny me"), lambda entry: False
                )

            # The denial was consumed by that one report -- nothing lingers.
            self.assertEqual(client.control.list_pending_approvals(), [])
        finally:
            _stop_server(server, socket_path, telemetry_dir)


if __name__ == "__main__":
    unittest.main()
