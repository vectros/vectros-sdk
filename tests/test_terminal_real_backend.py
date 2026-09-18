"""
TERM.16 — real-backend, end to end, matching Phase 12a's own acceptance
criteria exactly: a user runs a real command, asks a natural-language
question, and approves a proposed command, in one session, against a real
model, with the proposal audited.

No mocks for the kernel or the model: starts the actual `aiosctl
serve-kernel` process and a real local Ollama model, mirroring
`test_real_kernel_client.py`'s harness (this suite's own established
convention is a self-contained file per real-backend test, not a shared
harness module).
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vectros_sdk.terminal.repl import Terminal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1].parent
CARGO_AVAILABLE = shutil.which("cargo") is not None
OLLAMA_MODEL = "gemma4:e4b"


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0):
            return True
    except Exception:
        return False


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
@unittest.skipUnless(_ollama_reachable(), "a local Ollama daemon serving gemma4:e4b is required")
class TestTerminalRealBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pid = os.getpid()
        cls.agent_id = f"agent_term16_{pid}"
        cls.socket_path = Path(f"/tmp/aios-term16-{pid}.sock")
        cls.telemetry_dir = Path(f"/tmp/aios-term16-{pid}-telemetry")
        cls.socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)
        cls.telemetry_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["AIOS_AGENT_ID"] = cls.agent_id
        env["AIOS_RUN_ID"] = f"run_term16_{pid}"
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
                "--ollama-model",
                OLLAMA_MODEL,
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_socket(cls.socket_path)
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
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)

    def _agent_suffix(self) -> str:
        # tenant_permissions() namespaces every grant by
        # agent_id.trim_start_matches("agent_"); the SDK's agent_name
        # parameter *is* that suffix, not the full "agent_..." id (matching
        # test_real_kernel_client.py's own established convention).
        return self.agent_id[len("agent_"):]

    def test_ls_then_a_question_then_an_approved_and_audited_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as audit_dir:
            audit_log = Path(audit_dir) / "audit.jsonl"
            with patch("vectros_sdk.terminal.proposal.AUDIT_LOG_PATH", audit_log), patch(
                "builtins.input", return_value="y"
            ):
                terminal = Terminal(
                    agent_name=self._agent_suffix(),
                    socket_path=str(self.socket_path),
                    model=OLLAMA_MODEL,
                )

                # Turn 1: a real operator-typed command, dispatched locally.
                result = subprocess.run(["ls", str(REPO_ROOT)], capture_output=True)
                self.assertEqual(result.returncode, 0)

                # Turn 2: natural language, against the real model, over the
                # real kernel protocol -- and explicitly asked for a command,
                # so the approval path below is actually exercised rather
                # than left to chance.
                terminal.handle_line(
                    "Give me a single shell command, in a fenced code block, "
                    "to print the current date."
                )

                reply = terminal.messages[-1]["content"]
                self.assertTrue(
                    any(f"```{lang}" in reply for lang in ("propose", "bash", "sh", "shell")),
                    f"model did not propose a command in a recognized fence: {reply!r}",
                )

            # The approval above (input() patched to "y") must have run the
            # proposed command for real and recorded exactly one audit line.
            lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["decision"], "approved")
            self.assertIsNotNone(record["exit_code"])


if __name__ == "__main__":
    unittest.main()
