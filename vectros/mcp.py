"""Use Model Context Protocol servers as agent tools.

    github = MCPServer("github", url="https://api.githubcopilot.com/mcp/",
                       headers={"Authorization": f"Bearer {token}"})
    files = MCPServer("files", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])
    agent = Agent("ops", tools=[github, files])

The agent's process talks to the server (stdio or Streamable HTTP) with only
the standard library. Each MCP tool becomes ``<server>__<tool>`` and runs as a
kernel client tool call, so the kernel checks permission, applies approvals
and records it like any ``@tool`` function. Tools that the server does not
mark read-only need approval, as in the AIOS tool worker.

For servers that every agent on a host shares, the administrator registers
them in the AIOS tool worker instead; agents then name those tools by string.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import urllib.error
import urllib.request
import weakref
from pathlib import Path
from typing import Iterable, Sequence

from .errors import VectrosError
from .tools import Tool

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
SEPARATOR = "__"
# Model APIs accept function names from this set; kernel tool names are
# at most 63 bytes.
_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]")


class MCPError(VectrosError):
    """The MCP server failed to start, answered with an error, or a tool
    reported isError."""


class _Stdio:
    """Newline-delimited JSON-RPC over a child process's stdin/stdout."""

    def __init__(self, name: str, command: list[str], env: dict, cwd: str | None):
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env={**os.environ, **env}, cwd=cwd, bufsize=0)
        except FileNotFoundError as exc:
            raise MCPError(f"MCP server {name!r}: command not found: {command[0]}") from exc
        self.name = name
        self.stderr_tail: list[str] = []
        self._lock = threading.Lock()
        self._closed = threading.Event()

    def start(self, on_message) -> None:
        threading.Thread(target=self._read, args=(on_message,), daemon=True,
                         name=f"mcp-{self.name}").start()
        threading.Thread(target=self._read_stderr, daemon=True,
                         name=f"mcp-{self.name}-stderr").start()

    def _read(self, on_message) -> None:
        try:
            while True:
                try:
                    line = self.process.stdout.readline(MAX_MESSAGE_BYTES + 1)
                except (OSError, ValueError):
                    break
                if not line:
                    break
                if len(line) > MAX_MESSAGE_BYTES or not line.strip():
                    continue
                try:
                    on_message(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Servers must not print other text on stdout.
        finally:
            self._closed.set()
            on_message(None)

    def _read_stderr(self) -> None:
        try:
            for raw in iter(self.process.stderr.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip()
                self.stderr_tail = (self.stderr_tail + [text])[-20:]
        except (OSError, ValueError):
            pass

    def send(self, message: dict) -> None:
        data = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        with self._lock:
            if self._closed.is_set() or self.process.poll() is not None:
                raise ConnectionError(f"MCP server {self.name!r} exited")
            try:
                self.process.stdin.write(data)
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise ConnectionError(f"MCP server {self.name!r} closed its input") from exc

    @property
    def alive(self) -> bool:
        return not self._closed.is_set() and self.process.poll() is None

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self._closed.set()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


class _Http:
    """Streamable HTTP transport: POST each message; replies are JSON or SSE."""

    def __init__(self, name: str, url: str, headers: dict, timeout: float):
        self.name = name
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self.alive = True
        self._on_message = None

    def start(self, on_message) -> None:
        self._on_message = on_message

    def send(self, message: dict) -> None:
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream", **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        request = urllib.request.Request(self.url, data=json.dumps(message).encode(),
                                         headers=headers, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code == 404 and self.session_id:
                self.alive = False
                raise ConnectionError(f"MCP server {self.name!r} session expired") from exc
            raise ConnectionError(f"MCP server {self.name!r} returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"MCP server {self.name!r} unreachable: {exc.reason}") from exc
        with response:
            self.session_id = response.headers.get("Mcp-Session-Id") or self.session_id
            if response.status == 202 or "id" not in message:
                return
            if response.headers.get("Content-Type", "").startswith("text/event-stream"):
                self._read_sse(response)
                return
            body = response.read(MAX_MESSAGE_BYTES + 1)
            if len(body) > MAX_MESSAGE_BYTES:
                raise ConnectionError(f"MCP server {self.name!r} reply exceeded 8 MiB")
            if body.strip():
                self._dispatch(json.loads(body))

    def _read_sse(self, response) -> None:
        data: list[str] = []
        received = 0
        for raw in response:
            received += len(raw)
            if received > MAX_MESSAGE_BYTES:
                raise ConnectionError(f"MCP server {self.name!r} stream exceeded 8 MiB")
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                self._dispatch(json.loads("\n".join(data)))
                data = []
        if data:
            self._dispatch(json.loads("\n".join(data)))

    def _dispatch(self, payload) -> None:
        for message in payload if isinstance(payload, list) else [payload]:
            self._on_message(message)

    def close(self) -> None:
        if self.session_id:
            request = urllib.request.Request(
                self.url, method="DELETE",
                headers={"Mcp-Session-Id": self.session_id, **self.headers})
            try:
                urllib.request.urlopen(request, timeout=2).close()
            except (urllib.error.URLError, OSError):
                pass
        self.alive = False


def _close(transport) -> None:
    if transport is not None:
        transport.close()


def _render(result: dict):
    """A CallToolResult as text, or its structured content when it has no text."""
    parts = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "resource":
            resource = item.get("resource") or {}
            parts.append(str(resource["text"]) if "text" in resource else
                         f"[resource {resource.get('uri', '')} {resource.get('mimeType', 'binary')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link {item.get('uri', '')} {item.get('name', '')}]".strip())
        elif kind in ("image", "audio"):
            size = len(item.get("data", "")) * 3 // 4
            parts.append(f"[{kind} {item.get('mimeType', '')} {size} bytes]")
    if not parts and result.get("structuredContent") is not None:
        return result["structuredContent"]
    return "\n".join(parts)


class MCPServer:
    """An MCP server whose tools an agent can use.

    Args:
        name: Prefix for the server's tools (``<name>__<tool>``).
        command: Command that starts a stdio server, as a list or a string.
        url: Streamable HTTP endpoint, instead of a command.
        args: Extra arguments for ``command``.
        env: Extra environment for the stdio server.
        cwd: Working directory for the stdio server.
        headers: HTTP headers, for example ``Authorization``.
        tools: Only offer these server tool names.
        approval: True or False for every tool, a list of tool names that need
            approval, or None: approval unless the server marks the tool
            read-only (``readOnlyHint``).
        timeout: Seconds allowed for each request.
    """

    def __init__(self, name: str, command: str | Sequence[str] | None = None, *,
                 url: str | None = None, args: Iterable[str] = (),
                 env: dict[str, str] | None = None, cwd: str | None = None,
                 headers: dict[str, str] | None = None, tools: Iterable[str] | None = None,
                 approval: bool | Iterable[str] | None = None, timeout: float = 60):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name) or SEPARATOR in name:
            raise ValueError(f"invalid MCP server name {name!r}: use 1-32 of A-Z a-z 0-9 _ - "
                             "without '__'")
        if (command is None) == (url is None):
            raise ValueError("MCPServer needs exactly one of command or url")
        if isinstance(command, str):
            command = shlex.split(command)
        self.name = name
        self.command = [*command, *args] if command is not None else None
        self.url = url
        self.env = dict(env or {})
        self.cwd = cwd
        self.headers = dict(headers or {})
        self.allowed = set(tools) if tools is not None else None
        self.approval = approval if approval is None or isinstance(approval, bool) \
            else set(approval)
        self.timeout = timeout
        self.instructions = ""
        self.server_info: dict = {}
        self._transport = None
        self._ids = itertools.count(1)
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._finalizer = None
        self._tools: list[Tool] | None = None

    @classmethod
    def from_config(cls, path: str | os.PathLike) -> list["MCPServer"]:
        """Servers from an ``{"mcpServers": {...}}`` file, the format other
        MCP hosts and the AIOS tool worker use. ``${VAR}`` in env, headers and
        url is read from the environment."""
        data = json.loads(Path(path).read_text())
        servers = data.get("mcpServers", data) if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            raise ValueError("MCP config must be an object with an mcpServers object")
        result = []
        for name, raw in servers.items():
            if not isinstance(raw, dict) or raw.get("disabled"):
                continue
            result.append(cls(
                name, raw.get("command"), url=_expand(raw["url"]) if raw.get("url") else None,
                args=[str(a) for a in raw.get("args", [])],
                env={k: _expand(str(v)) for k, v in raw.get("env", {}).items()},
                cwd=raw.get("cwd"),
                headers={k: _expand(str(v)) for k, v in raw.get("headers", {}).items()},
                tools=raw.get("tools"), approval=raw.get("side_effect"),
                timeout=float(raw.get("timeout", 60))))
        return result

    # --- connection ---

    def connect(self) -> "MCPServer":
        """Start the server and run the MCP handshake (done on first use)."""
        if self._transport is not None and self._transport.alive:
            return self
        self.close()
        if self.command:
            transport = _Stdio(self.name, self.command, self.env, self.cwd)
        else:
            transport = _Http(self.name, self.url, self.headers, self.timeout)
        self._transport = transport
        self._finalizer = weakref.finalize(self, _close, transport)
        transport.start(self._on_message)
        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": {"name": "vectros-sdk", "version": "2"}},
                timeout=max(30.0, self.timeout))
        except Exception as exc:
            tail = "; ".join(getattr(transport, "stderr_tail", [])[-3:])
            self.close()
            raise MCPError(f"MCP server {self.name!r} failed to start: {exc}"
                           + (f" ({tail})" if tail else "")) from exc
        version = result.get("protocolVersion")
        if version not in SUPPORTED_PROTOCOL_VERSIONS:
            self.close()
            raise MCPError(f"MCP server {self.name!r} speaks unsupported protocol {version!r}")
        if isinstance(transport, _Http):
            transport.protocol_version = version
        self.server_info = result.get("serverInfo") or {}
        self.instructions = result.get("instructions") or ""
        self._notify("notifications/initialized")
        return self

    def close(self) -> None:
        """Stop the server (or end the HTTP session)."""
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._transport = None

    def __enter__(self) -> "MCPServer":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    def _on_message(self, message) -> None:
        if message is None:  # Transport closed: fail every waiter.
            with self._lock:
                waiters = list(self._pending.values())
                self._pending.clear()
            for waiter in waiters:
                waiter.put(ConnectionError(f"MCP server {self.name!r} disconnected"))
            return
        if not isinstance(message, dict):
            return
        if "method" in message:
            if "id" in message:  # Server request: answer ping, refuse the rest.
                reply = {"jsonrpc": "2.0", "id": message["id"]}
                if message["method"] == "ping":
                    reply["result"] = {}
                elif message["method"] == "roots/list":
                    reply["result"] = {"roots": []}
                else:
                    reply["error"] = {"code": -32601,
                                      "message": f"client does not support {message['method']}"}
                try:
                    self._transport.send(reply)
                except (ConnectionError, AttributeError):
                    pass
            return
        with self._lock:
            waiter = self._pending.pop(message.get("id"), None)
        if waiter is not None:
            waiter.put(message)

    def _notify(self, method: str, params: dict | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._transport.send(message)

    def _request(self, method: str, params: dict | None = None,
                 timeout: float | None = None) -> dict:
        request_id = next(self._ids)
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[request_id] = waiter
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            self._transport.send(message)
            reply = waiter.get(timeout=timeout or self.timeout)
        except queue.Empty:
            try:
                self._notify("notifications/cancelled",
                             {"requestId": request_id, "reason": "timeout"})
            except ConnectionError:
                pass
            raise TimeoutError(f"MCP server {self.name!r} did not answer {method} in time") \
                from None
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        if isinstance(reply, Exception):
            raise reply
        if "error" in reply:
            error = reply["error"] or {}
            raise MCPError(f"MCP server {self.name!r}: {error.get('message', 'error')} "
                           f"(code {error.get('code')})")
        result = reply.get("result")
        if not isinstance(result, dict):
            raise MCPError(f"MCP server {self.name!r}: {method} returned a non-object result")
        return result

    # --- tools ---

    def list_tools(self) -> list[dict]:
        """The server's tool descriptions, as MCP returns them."""
        self.connect()
        tools, cursor = [], None
        for _ in range(100):  # Bound pagination against a misbehaving server.
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(t for t in result.get("tools", [])
                         if isinstance(t, dict) and isinstance(t.get("name"), str))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return [t for t in tools if self.allowed is None or t["name"] in self.allowed]

    def call(self, tool: str, arguments: dict | None = None):
        """Call one server tool; returns text, or structured content."""
        self.connect()
        result = self._request("tools/call", {"name": tool, "arguments": arguments or {}})
        output = _render(result)
        if result.get("isError"):
            raise MCPError(output if isinstance(output, str) else json.dumps(output))
        return output

    def _needs_approval(self, tool: dict) -> bool:
        if isinstance(self.approval, bool):
            return self.approval
        if isinstance(self.approval, set):
            return tool["name"] in self.approval
        return not (tool.get("annotations") or {}).get("readOnlyHint", False)

    def tools(self) -> list[Tool]:
        """The server's tools as agent tools (listed once, then cached)."""
        if self._tools is None:
            self._tools = [self._as_tool(spec) for spec in self.list_tools()]
        return self._tools

    def _as_tool(self, spec: dict) -> Tool:
        remote = spec["name"]
        name = f"{self.name}{SEPARATOR}{_TOOL_NAME.sub('_', remote)}"[:63]
        schema = spec.get("inputSchema") if isinstance(spec.get("inputSchema"), dict) else {}
        schema = {"type": "object", "properties": {}, **schema}
        description = " ".join(str(spec.get("description") or spec.get("title") or remote).split())

        def run(**arguments):
            return self.call(remote, arguments)

        return Tool(name, description, schema, run, self._needs_approval(spec))


_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str) -> str:
    def replace(match):
        if match.group(1) not in os.environ:
            raise ValueError(f"MCP config references unset environment variable {match.group(1)}")
        return os.environ[match.group(1)]
    return _VARIABLE.sub(replace, value)
