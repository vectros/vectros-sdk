"""Typed client for the current development Operation Interface.

The HTTP ``/query`` client remains a compatibility layer for the SDK's legacy
module APIs.  This client speaks the daemon's deliberately small Unix-domain
development protocol.  It is not an authority channel: in particular it does
not expose tool invocation, approval, or terminal-success commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
from typing import Final, Iterable


MAX_COMMAND_BYTES: Final = 4 * 1024
MAX_RESPONSE_BYTES: Final = 1024 * 1024
_KINDS: Final = frozenset({"infer", "read_memory", "write_memory", "import_artifact", "share_artifact"})
_SIDE_EFFECTS: Final = frozenset(
    {"read_only", "idempotent_write", "non_idempotent_write", "destructive", "privilege_changing"}
)


class OperationInterfaceError(RuntimeError):
    """A stable, payload-free error from the development Operation Interface."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OperationView:
    operation_id: str
    state: str


@dataclass(frozen=True)
class OperationEvent:
    cursor: int
    event_type: str
    # Journal payloads are protocol internals. SDK callers receive no payload
    # body here, preventing accidental logging of future sensitive fields.


class OperationClient:
    """Client for development lifecycle observation, not a production authority API.

    The daemon must already own the Unix socket. This client never creates a
    socket path, follows redirects, or falls back to HTTP. Production agents
    must use the authenticated, versioned Operation Interface once available.
    """

    def __init__(self, socket_path: str, timeout_seconds: float = 5.0) -> None:
        if not socket_path.startswith("/") or "\x00" in socket_path:
            raise ValueError("socket_path must be an absolute Unix path")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._socket_path = socket_path
        self._timeout_seconds = timeout_seconds

    def submit(
        self,
        *,
        agent_id: str,
        instance_id: str,
        run_id: str,
        operation_id: str,
        kind: str,
        side_effect: str,
        idempotency_key: str | None = None,
    ) -> OperationView:
        if kind not in _KINDS:
            raise ValueError("development OperationClient does not submit tool invocations")
        if side_effect not in _SIDE_EFFECTS:
            raise ValueError("unsupported side-effect class")
        values = [agent_id, instance_id, run_id, operation_id, kind, side_effect]
        if idempotency_key is not None:
            values.append(idempotency_key)
        return self._view("SUBMIT", *values)

    def admit_next(self) -> OperationView:
        return self._view("ADMIT")

    def start(self, operation_id: str) -> OperationView:
        return self._view("START", operation_id)

    def cancel(self, operation_id: str) -> OperationView:
        return self._view("CANCEL", operation_id)

    def get(self, operation_id: str) -> OperationView:
        return self._view("GET", operation_id)

    def watch(self, agent_id: str, run_id: str, cursor: int = 0) -> list[OperationEvent]:
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        lines = self._request("WATCH", agent_id, run_id, str(cursor))
        if lines == ["OK END"]:
            return []
        events: list[OperationEvent] = []
        for line in lines:
            parts = line.split(" ", 3)
            if len(parts) < 3 or parts[0] != "EVENT" or not parts[1].isdigit():
                raise OperationInterfaceError("MALFORMED_RESPONSE")
            events.append(OperationEvent(cursor=int(parts[1]), event_type=parts[2]))
        return events

    def _view(self, command: str, *fields: str) -> OperationView:
        lines = self._request(command, *fields)
        if len(lines) != 1:
            raise OperationInterfaceError("MALFORMED_RESPONSE")
        parts = lines[0].split(" ")
        if len(parts) != 3 or parts[0] != "OK":
            raise OperationInterfaceError("MALFORMED_RESPONSE")
        return OperationView(operation_id=parts[2], state=parts[1])

    def _request(self, command: str, *fields: str) -> list[str]:
        if any(not _safe_token(value) for value in (command, *fields)):
            raise ValueError("Operation Interface fields must be bounded non-whitespace tokens")
        wire = " ".join((command, *fields)).encode("utf-8") + b"\n"
        if len(wire) > MAX_COMMAND_BYTES:
            raise ValueError("Operation Interface command exceeds its bound")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout_seconds)
                connection.connect(self._socket_path)
                connection.sendall(wire)
                chunks: list[bytes] = []
                size = 0
                while True:
                    data = connection.recv(8192)
                    if not data:
                        break
                    size += len(data)
                    if size > MAX_RESPONSE_BYTES:
                        raise OperationInterfaceError("RESPONSE_TOO_LARGE")
                    chunks.append(data)
        except (OSError, TimeoutError) as error:
            raise OperationInterfaceError("UNAVAILABLE") from error
        try:
            text = b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as error:
            raise OperationInterfaceError("MALFORMED_RESPONSE") from error
        lines = [line for line in text.splitlines() if line]
        if len(lines) == 1 and lines[0].startswith("ERR "):
            code = lines[0][4:]
            if code in {"INVALID_ARGUMENT", "UNAVAILABLE", "NOT_FOUND", "FAILED_PRECONDITION"}:
                raise OperationInterfaceError(code)
            raise OperationInterfaceError("MALFORMED_RESPONSE")
        return lines


def _safe_token(value: str) -> bool:
    return bool(value) and len(value) <= 128 and not any(character.isspace() for character in value) and "\x00" not in value
