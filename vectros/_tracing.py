"""Small best-effort OpenInference span emitter for AIOS client processes.

Spans are redacted before they enter the bounded sender queue. Sending happens
on a daemon thread; a missing or slow collector never blocks a syscall.
"""

from __future__ import annotations

import json
import ctypes
import os
import queue
import re
import secrets
import socket
import array
import fcntl
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any


MAX_CONTENT_BYTES = 64 * 1024
MAX_SPAN_BYTES = 1024 * 1024
SOCKET_PATH = os.environ.get("AIOS_TRACE_SOCKET", "/run/aios-traced/ingest.sock")

_SECRET_KEY = re.compile(r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret)", re.I)
_EMAIL = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_NAMED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret)"
    r"([\"']?\s*[:=]\s*[\"']?)[^\s,;\"']+"
)
_KEYS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)


class AIOSTraceContextC(ctypes.Structure):
    _fields_ = [
        ("trace_id", ctypes.c_uint8 * 16),
        ("span_id", ctypes.c_uint8 * 8),
        ("parent_span_id", ctypes.c_uint8 * 8),
    ]


def _truncate_text(value: str) -> tuple[str, bool, int]:
    raw = value.encode("utf-8", "replace")
    if len(raw) <= MAX_CONTENT_BYTES:
        return value, False, len(raw)
    clipped = raw[:MAX_CONTENT_BYTES].decode("utf-8", "ignore")
    return clipped, True, len(raw)


def _redact_text(value: str) -> str:
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _NAMED_SECRET.sub(r"\1\2[REDACTED_SECRET]", value)
    for pattern in _KEYS:
        value = pattern.sub("[REDACTED_SECRET]", value)
    return value


def _sanitize(value: Any, depth: int = 0) -> tuple[Any, bool, int]:
    """Redact common secrets, bound shape, and return truncation metadata."""
    if depth > 12:
        return "[TRUNCATED_DEPTH]", True, 0
    if isinstance(value, str):
        clean = _redact_text(value)
        clean, truncated, original_bytes = _truncate_text(clean)
        return clean, truncated, original_bytes
    if isinstance(value, dict):
        result, any_truncated, total = {}, False, 0
        for index, (key, item) in enumerate(value.items()):
            if index >= 512:
                result["_aios_truncated_items"] = len(value) - index
                any_truncated = True
                break
            safe_key = str(key)[:256]
            if _SECRET_KEY.search(safe_key):
                result[safe_key] = "[REDACTED_SECRET]"
                continue
            safe_value, truncated, size = _sanitize(item, depth + 1)
            result[safe_key] = safe_value
            any_truncated |= truncated
            total += size
        return result, any_truncated, total
    if isinstance(value, (list, tuple)):
        result, any_truncated, total = [], False, 0
        for item in value[:512]:
            safe_value, truncated, size = _sanitize(item, depth + 1)
            result.append(safe_value)
            any_truncated |= truncated
            total += size
        if len(value) > 512:
            result.append("[TRUNCATED_ITEMS]")
            any_truncated = True
        return result, any_truncated, total
    if value is None or isinstance(value, (bool, int, float)):
        return value, False, 0
    return str(value)[:1024], True, 0


class _BestEffortSender:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=512)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0
        self._telemetry_fds: dict[int, int] = {}

    def send(self, record: dict[str, Any]) -> None:
        try:
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            self._drop()
            return
        if len(encoded) > MAX_SPAN_BYTES:
            self._drop()
            return
        try:
            agent_id = record.get("attributes", {}).get("aios.agent.id")
            if isinstance(agent_id, bool) or not isinstance(agent_id, int) or agent_id <= 0:
                self._drop()
                return
            self._queue.put_nowait((encoded + b"\n", agent_id))
        except queue.Full:
            self._drop()
            return
        if self._thread is None:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="aios-trace-sender", daemon=True)
                    self._thread.start()

    def _drop(self) -> None:
        with self._lock:
            self.dropped += 1

    def flush(self, timeout: float) -> bool:
        """Wait up to timeout seconds for queued spans to be sent."""
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def _run(self) -> None:
        while True:
            record, agent_id = self._queue.get()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.02)
                    sock.connect(SOCKET_PATH)
                    fd = self._telemetry_fds.get(agent_id)
                    if fd is None:
                        try:
                            fd = os.open("/dev/aios_telemetry", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
                            # AIOS ABI 4.1: _IOW('A', 0xFB, struct aios_trace_scope).
                            scope = struct.pack("@HHIQ", 4, 16, 0, agent_id)
                            fcntl.ioctl(fd, 0x401041FB, scope)
                            self._telemetry_fds[agent_id] = fd
                        except OSError:
                            if fd is not None:
                                os.close(fd)
                            fd = None
                    if fd is not None:
                        rights = array.array("i", [fd])
                        sent = sock.sendmsg([record], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
                        if sent < len(record):
                            sock.sendall(record[sent:])
                    else:
                        sock.sendall(record)
                    # The collector closes the connection once it has checked the
                    # span against the live kernel agent; wait for that, so the
                    # agent does not unregister first.
                    sock.shutdown(socket.SHUT_WR)
                    sock.settimeout(1.0)
                    while sock.recv(64):
                        pass
            except (OSError, TimeoutError):
                self._drop()
            finally:
                self._queue.task_done()


_SENDER = _BestEffortSender()


def flush_spans(timeout: float = 2.0) -> bool:
    """Send queued spans while their agent is still registered.

    The collector checks each span against the live kernel agent, so spans
    must leave before the agent unregisters or the process exits.
    """
    return _SENDER.flush(timeout)


@dataclass
class TraceSpan:
    trace_id: bytes
    span_id: bytes
    parent_span_id: bytes
    name: str
    kind: str
    agent_id: int
    session_id: str | None = None
    started_ns: int = field(default_factory=time.time_ns)
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def root(cls, name: str, kind: str, agent_id: int,
             session_id: str | None = None) -> "TraceSpan":
        return cls(secrets.token_bytes(16), secrets.token_bytes(8), bytes(8),
                   name, kind, agent_id, session_id)

    def child(self, name: str, kind: str) -> "TraceSpan":
        return TraceSpan(self.trace_id, secrets.token_bytes(8), self.span_id,
                         name, kind, self.agent_id, self.session_id)

    def context(self) -> AIOSTraceContextC:
        context = AIOSTraceContextC()
        context.trace_id[:] = self.trace_id
        context.span_id[:] = self.span_id
        context.parent_span_id[:] = self.parent_span_id
        return context

    def set_input(self, value: Any, mime_type: str = "application/json") -> None:
        safe, truncated, original = _sanitize(value)
        self.attributes["input.value"] = safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False)
        self.attributes["input.mime_type"] = mime_type
        if truncated:
            self.attributes["aios.content.truncated"] = True
            self.attributes["aios.content.original_bytes"] = original

    def finish(self, output: Any = None, error: str | None = None,
               extra: dict[str, Any] | None = None) -> None:
        ended_ns = time.time_ns()
        attributes = dict(self.attributes)
        if output is not None:
            safe, truncated, original = _sanitize(output)
            attributes["output.value"] = safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False)
            attributes["output.mime_type"] = "application/json"
            if truncated:
                attributes["aios.content.truncated"] = True
                attributes["aios.content.original_bytes"] = max(
                    attributes.get("aios.content.original_bytes", 0), original)
        if extra:
            safe_extra, _, _ = _sanitize(extra)
            attributes.update(safe_extra)
        if error:
            safe_error, _, _ = _sanitize(error)
            attributes["exception.type"] = (
                type(error).__name__ if isinstance(error, BaseException) else "AIOSWorkerError"
            )
            attributes["exception.message"] = safe_error
        if self.session_id:
            attributes["session.id"] = self.session_id
        attributes["aios.agent.id"] = self.agent_id
        attributes, truncated, _ = _sanitize(attributes)
        if truncated:
            attributes["aios.content.truncated"] = True
        _SENDER.send({
            "name": self.name,
            "trace_id": self.trace_id.hex(),
            "span_id": self.span_id.hex(),
            "parent_span_id": self.parent_span_id.hex() if any(self.parent_span_id) else None,
            "start_time_unix_nano": self.started_ns,
            "end_time_unix_nano": ended_ns,
            "status": "ERROR" if error else "OK",
            "attributes": {"openinference.span.kind": self.kind, **attributes},
            "events": self.events,
        })
