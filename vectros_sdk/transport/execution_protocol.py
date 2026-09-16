"""
Real AIOS execution-protocol client (M2: SDK.3/SDK.6 foundation).

Speaks the exact framed wire format `aios-sdk`'s Rust `ExecutionClient` uses —
a 4-byte big-endian length prefix followed by externally-tagged JSON — against
the standing kernel server (``aiosctl serve-kernel``, backed by
``aiosd::serve_forever``). Verified byte-for-byte against the Rust side by
serializing real ``ExecutionRequest``/``ExecutionResponse`` values and
comparing the JSON, not by guessing serde's conventions.

This is the first Python client that speaks the *real* protocol. Two other
Python transports exist and are narrower or non-functional by comparison:

- ``vectros_sdk.operation.OperationClient`` — a real, working, but
  intentionally limited *development* protocol (lifecycle observation only;
  it explicitly refuses to submit tool invocations or carry a grant).
- ``vectros_sdk.client.send_request`` — POSTs to ``http://localhost:8000/query``,
  which nothing in this workspace serves (see docs/architecture.md SDK.2).

Example
-------
>>> with ExecutionProtocolClient("/tmp/aios.sock") as client:
...     client.execute("req_1", memory_put("res_memory_agent_default", "hello"))
...     client.execute("req_2", memory_get("res_memory_agent_default"))
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, Optional

MAX_FRAME_BYTES = 1024 * 1024
# Must track aios_core::ProtocolVersion::CURRENT (crates/aios-core/src/kernel/system_call/protocol.rs).
# There is no negotiation yet (ARCH.27/SYS.5 track that); a mismatch is
# refused by the server, not silently accepted.
PROTOCOL_VERSION = {"major": 1, "minor": 3}


class ExecutionProtocolError(Exception):
    """Raised for framing, I/O, or server-rejection failures. Never silent."""


def _write_frame(sock: socket.socket, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    if not body or len(body) > MAX_FRAME_BYTES:
        raise ExecutionProtocolError(f"frame size {len(body)} out of bounds")
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ExecutionProtocolError("connection closed while reading a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(sock: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise ExecutionProtocolError(f"frame length {length} out of bounds")
    body = _recv_exact(sock, length)
    return json.loads(body)


class ExecutionProtocolClient:
    """One connection to a standing AIOS kernel server (`aiosctl serve-kernel`).

    Not thread-safe; one socket serves one logical caller, matching the
    server's current sequential-connection design (concurrent connections are
    Phase 5/SCH.6 work, not yet built).
    """

    def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(socket_path)

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "ExecutionProtocolClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def execute(
        self,
        request_id: str,
        kernel_request: Dict[str, Any],
        deadline_ms: int = 30_000,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sends one ``ExecutionRequest`` and returns the ``KernelValue`` payload.

        ``kernel_request`` is the externally-tagged ``KernelRequest`` JSON,
        e.g. ``{"Data": {"MemoryGet": {"key": "res_memory_x"}}}``. Use the
        builder functions below rather than hand-building these where
        possible — they exist precisely to avoid drift from the Rust shapes.

        Raises ``ExecutionProtocolError`` if the server rejects the request
        or the response has an unexpected shape. Never returns a fabricated
        result for a rejected or malformed response.
        """
        envelope = {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "deadline": deadline_ms,
            "request": kernel_request,
            "trace_id": trace_id,
            "span_id": span_id,
        }
        _write_frame(self._sock, envelope)
        response = _read_frame(self._sock)
        if "Rejected" in response:
            code = response["Rejected"].get("code", "unknown")
            raise ExecutionProtocolError(f"server rejected {request_id}: {code}")
        if "Completed" not in response:
            raise ExecutionProtocolError(f"unexpected response shape: {response!r}")
        return response["Completed"]["value"]


# ---------------------------------------------------------------------------
# KernelRequest builders. Each mirrors one variant of aios_sdk::KernelRequest
# (crates/aios-sdk/src/execution.rs) exactly — verified against the Rust
# side's own serde output, not guessed.
# ---------------------------------------------------------------------------


def memory_get(key: str) -> Dict[str, Any]:
    return {"Data": {"MemoryGet": {"key": key}}}


def memory_put(
    key: str,
    text: str,
    expected_revision: int = 0,
    expires_at_ms: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "Data": {
            "MemoryPut": {
                "key": key,
                "expected_revision": expected_revision,
                "value": {"Text": text},
                "expires_at_ms": expires_at_ms,
            }
        }
    }


def context_create(context: str) -> Dict[str, Any]:
    return {"ContextCreate": {"context": context}}


def context_append(
    context: str, revision: int, role: str, text: str, tool_call_id: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "ContextAppend": {
            "context": context,
            "revision": revision,
            "message": {"role": role, "text": text, "tool_call_id": tool_call_id},
        }
    }


def context_prepare(
    context: str, revision: int, model: str, output_bytes: int, window_bytes: int
) -> Dict[str, Any]:
    return {
        "ContextPrepare": {
            "context": context,
            "revision": revision,
            "model": model,
            "output_bytes": output_bytes,
            "window_bytes": window_bytes,
        }
    }


def model_generate(
    preparation: str, model: str, input_bytes: int, max_output_tokens: int
) -> Dict[str, Any]:
    return {
        "ModelGenerate": {
            "preparation": preparation,
            "model": model,
            "input_bytes": input_bytes,
            "max_output_tokens": max_output_tokens,
        }
    }
