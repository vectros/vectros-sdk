"""
ASBX.13: `AgentSandbox` — a real, ephemeral, spec-scoped sandbox kernel for
tests, cheap enough to use like a unit-test fixture.

Launches the real `aiosctl serve-sandbox` process (Rust, this same
repository) against a freshly-built `SandboxSpec` JSON document -- the exact
schema `aios_core::SandboxSpec`'s own `Serialize` impl produces (ASBX.3),
never a shape this module invents -- and hands back a real
`AIOSClient(socket_path=...)` wired to it. That client is the same
real-kernel translation layer (`vectros_sdk.client.real_kernel`)
`test_real_kernel_client.py` already proves works end to end; this module
adds no new wire-protocol code of its own.

Grants are auto-generated to mirror `kernel_server.rs::tenant_permissions`'s
exact naming convention (`res_memory_<suffix>`, `res_storage_<suffix>`,
`res_context_<suffix>`, `res_tool_uppercase_<suffix>`, and
`res_model_<model>` when a model is given) so `real_kernel.py`'s existing,
untouched calls work against a sandbox exactly as they do against the real
production bootstrap -- a test author gets the familiar `AIOSClient` API for
free, not a second, sandbox-specific client surface.

No mocks: `with AgentSandbox() as sandbox: sandbox.client.memory.create(...)`
talks to a real `ScopedKernel` process over a real Unix socket, whose
`AccessManager` ceiling is exactly this spec's own grants -- nothing more.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from vectros_sdk.client.client import AIOSClient

REPO_ROOT = Path(__file__).resolve().parents[2].parent
SANDBOX_SPEC_VERSION = 1
DEFAULT_SOCKET_WAIT_SECONDS = 30.0


class AgentSandboxUnavailable(Exception):
    """Raised when the real `aiosctl` binary cannot be built/run at all, or
    exits before ever binding its socket. Never falls back to a mock
    client -- a test that needs a real sandbox either gets one or fails
    loudly."""


def model_resource_id(model_name: str) -> str:
    """Mirrors `aiosd::model_resource_id` exactly (model names commonly
    contain `:`, which the kernel's `ResourceId` rejects)."""
    return f"res_model_{model_name.replace(':', '-')}"


def _default_grants(suffix: str, model: Optional[str]) -> List[dict]:
    def resource(kind: str) -> str:
        return f"res_{kind}_{suffix}"

    grants = [
        {"operation": "memory.get", "resource": resource("memory")},
        {"operation": "memory.put", "resource": resource("memory")},
        {"operation": "memory.delete", "resource": resource("memory")},
        {"operation": "storage.get", "resource": resource("storage")},
        {"operation": "storage.put", "resource": resource("storage")},
        {"operation": "storage.delete", "resource": resource("storage")},
        {"operation": "storage.list", "resource": resource("storage")},
        {"operation": "context.create", "resource": resource("context")},
        {"operation": "context.append", "resource": resource("context")},
        {"operation": "context.prepare", "resource": resource("context")},
        {"operation": "context.checkpoint", "resource": resource("context")},
        {"operation": "context.restore", "resource": resource("context")},
        {"operation": "tool.invoke", "resource": f"res_tool_uppercase_{suffix}"},
        {"operation": "tool.describe", "resource": f"res_tool_uppercase_{suffix}"},
    ]
    if model:
        resource_id = model_resource_id(model)
        grants.append({"operation": "model.generate", "resource": resource_id})
        grants.append({"operation": "model.capabilities", "resource": resource_id})
    return grants


class AgentSandbox:
    """A real sandboxed kernel process, scoped to exactly this instance's
    own `SandboxSpec`. Use as a context manager; `sandbox.client` is a real
    `AIOSClient` once entered.

    Args:
        agent_name: bare suffix (no `agent_` prefix) identifying this
            sandbox's agent/run; a fresh random one is used if omitted, so
            concurrent sandboxes never collide.
        model: an Ollama model name (e.g. `"gemma4:e4b"`) to wire a real
            backend for, or `None` (the default) for a fixture-only
            sandbox -- most tests that only exercise memory/storage/tool
            grants need no model at all.
        extra_grants: additional `(operation, resource)` pairs beyond the
            conventional default set, for a test that needs to name its own
            resource.
        cassette_dir / cassette_mode: ASBX.10 record/replay, only
            meaningful together with `model`. `cassette_mode` is `"record"`
            or `"replay"`.
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        extra_grants: Optional[Sequence[Tuple[str, str]]] = None,
        max_cpu_millis: int = 1_000_000,
        max_memory_bytes: int = 64 * 1024 * 1024,
        max_tasks: int = 1_000,
        cassette_dir: Optional[str] = None,
        cassette_mode: Optional[str] = None,
    ) -> None:
        if not shutil.which("cargo"):
            raise AgentSandboxUnavailable(
                "cargo is required to build/run the real aiosctl serve-sandbox process"
            )
        self.agent_name = agent_name or f"sbx{uuid.uuid4().hex[:12]}"
        self._model = model
        self._extra_grants = list(extra_grants or [])
        self._budget = (max_cpu_millis, max_memory_bytes, max_tasks)
        self._cassette_dir = cassette_dir
        self._cassette_mode = cassette_mode
        self._process: Optional[subprocess.Popen] = None
        self._spec_path: Optional[Path] = None
        self._socket_path: Optional[Path] = None
        self.client: Optional[AIOSClient] = None

    def _build_spec(self) -> dict:
        suffix = self.agent_name
        cpu, memory, tasks = self._budget
        grants = _default_grants(suffix, self._model) + [
            {"operation": operation, "resource": resource}
            for operation, resource in self._extra_grants
        ]
        return {
            "version": SANDBOX_SPEC_VERSION,
            "agent": f"agent_{suffix}",
            "run": f"run_{suffix}",
            "budget": {
                "max_cpu_millis": cpu,
                "max_memory_bytes": memory,
                "max_tasks": tasks,
            },
            "grants": grants,
            "quotas": {
                "max_memory_bytes": memory,
                "max_storage_bytes": memory,
                "max_tokens": 100_000,
                "max_cost_millicents": 100_000,
                "max_wall_clock_ms": 60_000,
            },
            "tool_allowlist": [],
            "model_allowlist": [],
            "network_policy": "DenyAll",
            "filesystem_policy": {"read_only_paths": [], "read_write_paths": []},
            "determinism": "Fixture",
            "isolation_tier": "Process",
            "seed": {"memory": [], "storage": []},
        }

    def __enter__(self) -> "AgentSandbox":
        unique = f"{self.agent_name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._spec_path = Path(f"/tmp/aios-sandbox-{unique}-spec.json")
        self._socket_path = Path(f"/tmp/aios-sandbox-{unique}.sock")
        self._spec_path.write_text(json.dumps(self._build_spec()))

        args = [
            "cargo",
            "run",
            "-q",
            "-p",
            "aiosctl",
            "--",
            "serve-sandbox",
            str(self._spec_path),
            str(self._socket_path),
        ]
        if self._model:
            args += ["--ollama-model", self._model]
            if self._cassette_dir and self._cassette_mode:
                args += [
                    "--cassette-dir",
                    self._cassette_dir,
                    "--cassette-mode",
                    self._cassette_mode,
                ]

        self._process = subprocess.Popen(
            args,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_for_socket()
        self.client = AIOSClient(agent_name=self.agent_name, socket_path=str(self._socket_path))
        return self

    def _wait_for_socket(self, timeout: float = DEFAULT_SOCKET_WAIT_SECONDS) -> None:
        assert self._process is not None and self._socket_path is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._socket_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.settimeout(0.5)
                        probe.connect(str(self._socket_path))
                    return
                except OSError:
                    pass
            if self._process.poll() is not None:
                output = self._process.stdout.read() if self._process.stdout else ""
                raise AgentSandboxUnavailable(f"serve-sandbox exited early:\n{output}")
            time.sleep(0.05)
        self._process.terminate()
        raise TimeoutError(f"serve-sandbox never bound {self._socket_path}")

    def __exit__(self, *_exc_info: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._spec_path is not None:
            self._spec_path.unlink(missing_ok=True)
        if self._socket_path is not None:
            self._socket_path.unlink(missing_ok=True)
