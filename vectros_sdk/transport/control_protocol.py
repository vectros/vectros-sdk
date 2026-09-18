"""
Real AIOS operator control-protocol client (SDK.9).

Speaks the exact framed wire format `aios-sdk`'s Rust `read_frame`/`write_frame`
use (a 4-byte big-endian length prefix followed by externally-tagged JSON) —
the same framing `vectros_sdk.transport.execution_protocol` already proved —
against the standing kernel server's *dedicated control socket* (CTL.11,
``<socket-path>.control``, not the protocol socket itself), for the Tier 1
(read) and Tier 2 (intervene) operator verbs `aiosd::transport::owner`
implements (see ``crates/aiosd/src/transport/owner.rs`` and
``crates/aios-sdk/src/control.rs``).

This gives the Python SDK real agent lifecycle, status, and cancellation
APIs matching AGT.4/CTL.7/CTL.8 — not a client-side reimplementation of any
of that logic, purely a typed wire adapter to the real server.

Verified against a real ``aiosctl serve-kernel`` instance, not guessed:
every builder here was checked against the actual JSON the Rust server
sends/expects for that exact ``ControlRequest``/``ControlResponse`` variant.

Example
-------
>>> with ControlProtocolClient("/tmp/aios.sock.control") as client:
...     client.list_agents()
...     client.terminate_agent("agent_default", confirm=True)
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, List, Optional

MAX_FRAME_BYTES = 1024 * 1024


class ControlProtocolError(Exception):
    """Raised for framing, I/O, or server-rejection failures. Never silent."""


def _write_frame(sock: socket.socket, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    if not body or len(body) > MAX_FRAME_BYTES:
        raise ControlProtocolError(f"frame size {len(body)} out of bounds")
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ControlProtocolError("connection closed while reading a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(sock: socket.socket) -> Any:
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise ControlProtocolError(f"frame length {length} out of bounds")
    body = _recv_exact(sock, length)
    return json.loads(body)


class ControlProtocolClient:
    """One connection to a standing AIOS kernel server's control socket.

    Not a persistent session: matching ``OwnerSession::serve_once``'s own
    one-request-per-connection design, this opens a fresh Unix socket
    connection for every call rather than reusing one — there is no
    protocol-level reason to keep a connection open between operator verbs,
    and it keeps this client's own state trivial (there is none).
    """

    def __init__(self, control_socket_path: str, timeout: float = 30.0) -> None:
        self._path = control_socket_path
        self._timeout = timeout

    def __enter__(self) -> "ControlProtocolClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def _call(self, request: Any) -> Any:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        try:
            sock.connect(self._path)
            _write_frame(sock, request)
            response = _read_frame(sock)
        finally:
            sock.close()
        if isinstance(response, dict) and "Rejected" in response:
            code = response["Rejected"].get("code", "unknown")
            raise ControlProtocolError(f"control request rejected: {code}")
        return response

    # --- Tier 1: read verbs (operator.observe) -----------------------------

    def ping(self) -> None:
        response = self._call("Ping")
        if response != "Pong":
            raise ControlProtocolError(f"unexpected ping response: {response!r}")

    def list_agents(self) -> List[Dict[str, Any]]:
        return _unwrap(self._call("ListAgents"), "Agents")

    def describe_agent(self, agent_id: str) -> Dict[str, Any]:
        return _unwrap(self._call({"DescribeAgent": agent_id}), "Agent")

    def list_runs(self) -> List[Dict[str, Any]]:
        return _unwrap(self._call("ListRuns"), "Runs")

    def describe_run(self, run_id: str) -> Dict[str, Any]:
        return _unwrap(self._call({"DescribeRun": run_id}), "Run")

    def queue_depths(self) -> Dict[str, int]:
        return _unwrap(self._call("QueueDepths"), "QueueDepths")

    def budget_usage(self, agent_id: str) -> Dict[str, Any]:
        return _unwrap(self._call({"BudgetUsage": agent_id}), "BudgetUsage")

    def list_requests(self) -> List[Dict[str, Any]]:
        return _unwrap(self._call("ListRequests"), "Requests")

    # --- Tier 2: intervene verbs (operator.intervene) -----------------------
    #
    # `terminate_run`/`terminate_agent` require `confirm=True` -- a real kill
    # button is a foot-gun (plan.md's own Phase 22 note); this mirrors
    # `aiosctl control terminate-*`'s own `--yes` requirement rather than
    # letting a bare method call silently do something irreversible.

    def cancel_request(self, request_id: str) -> str:
        """Returns the request's status afterward (``"Cancelled"`` for a
        queued request; still ``"Dispatched"`` for a cooperative-only signal
        on an already-dispatched one — see the Rust server's own doc
        comment on `SystemCallDispatcher::cancel_any` for exactly why)."""
        return _unwrap(self._call({"CancelRequest": request_id}), "RequestCancelled")

    def suspend_run(self, run_id: str) -> Dict[str, Any]:
        return _unwrap(self._call({"SuspendRun": run_id}), "RunSuspended")

    def resume_run(self, run_id: str) -> Dict[str, Any]:
        return _unwrap(self._call({"ResumeRun": run_id}), "RunResumed")

    def terminate_run(self, run_id: str, *, confirm: bool) -> Dict[str, Any]:
        if not confirm:
            raise ControlProtocolError(
                "refusing to terminate a run without confirm=True"
            )
        return _unwrap(self._call({"TerminateRun": run_id}), "RunTerminated")

    def terminate_agent(self, agent_id: str, *, confirm: bool) -> Dict[str, Any]:
        if not confirm:
            raise ControlProtocolError(
                "refusing to terminate an agent without confirm=True"
            )
        return _unwrap(self._call({"TerminateAgent": agent_id}), "AgentTerminated")


def _unwrap(response: Any, variant: str) -> Any:
    if isinstance(response, dict) and variant in response:
        return response[variant]
    raise ControlProtocolError(f"expected {variant!r} response, got {response!r}")
