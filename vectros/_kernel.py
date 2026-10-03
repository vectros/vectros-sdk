"""ctypes binding to libaios.so, the userspace half of the AIOS kernel ABI.

Everything here talks to /dev/aios. The higher layers (Agent, tool) only use
the methods of :class:`Kernel`, so tests can swap in a fake with the same
surface.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import fcntl
import json
import os
import struct
import time
from pathlib import Path
from typing import Callable, Generator

from ._tracing import AIOSTraceContextC, TraceSpan
from .errors import KernelError, KernelUnavailable, ToolDenied

ABI_VERSION = 4

SYSCALL_LLM, SYSCALL_MEMORY, SYSCALL_STORAGE, SYSCALL_TOOL = 0, 1, 2, 3

STATE_CREATED, STATE_QUEUED, STATE_PROCESSING, STATE_DONE = 0, 1, 2, 3
STATE_FAILED, STATE_CANCELLED = 4, 5
STATE_HELD_APPROVAL = 9
TERMINAL_STATES = (STATE_DONE, STATE_FAILED, STATE_CANCELLED)

STO_WRITE, STO_RETRIEVE = 3, 4

OWNER_APPROVE_TOOL, OWNER_DENY_TOOL = 3, 4
_IOCTL_OWNER_ACTION = 0x40A04173  # _IOW('A', 0x73, struct aios_owner_action)
_IOCTL_LIST_TOOLS_NR = 0x54        # _IOWR('A', 0x54, ...), size computed below

_TOOL_HEADER = struct.Struct("@HHIQ64sII")
_STORAGE_HEADER = struct.Struct("@I256sQI4xQQ")
_TOOL_ENTRY = struct.Struct("@HHI64s32s256sIII")
_TOOL_PAGE_HEADER = struct.Struct("@HHIIIII")

RESULT_MAX = 256 * 1024

_LIB_CANDIDATES = (
    "/usr/lib/libaios.so",
    "/usr/lib64/libaios.so",
    "/usr/local/lib/libaios.so",
)


def find_library() -> str:
    """Locate libaios.so: $VECTROS_LIBAIOS, system paths, then the linker."""
    explicit = os.environ.get("VECTROS_LIBAIOS") or os.environ.get("AIOS_LIB")
    if explicit:
        return explicit
    for candidate in _LIB_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = ctypes.util.find_library("aios")
    if found:
        return found
    raise KernelUnavailable(
        "libaios.so not found; install the aios package or set VECTROS_LIBAIOS")


def _errno_error(what: str) -> KernelError:
    code = ctypes.get_errno()
    return KernelError(code, f"{what}: {os.strerror(code)}")


class Kernel:
    """One open handle on /dev/aios."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or find_library()
        try:
            self.lib = ctypes.CDLL(path, use_errno=True)
        except OSError as exc:
            raise KernelUnavailable(f"cannot load {path}: {exc}") from exc
        self._bind()
        self.h = self.lib.aios_open()
        if not self.h:
            code = ctypes.get_errno()
            hints = {
                errno.ENOENT: "aios.ko is not loaded",
                errno.EACCES: "add your user to group aios and log in again",
            }
            detail = hints.get(code, os.strerror(code) if code else "unknown error")
            raise KernelUnavailable(f"cannot open /dev/aios: {detail}")
        # Buffers whose addresses the kernel reads after submit; freed on wait.
        self._inflight: dict[int, object] = {}

    def _bind(self) -> None:
        void_p, u64, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
        int_p = ctypes.POINTER(ctypes.c_int)
        signatures = {
            "aios_register_agent": [void_p, ctypes.c_char_p, ctypes.POINTER(u64)],
            "aios_unregister_agent": [void_p, u64],
            "aios_setup_core": [void_p, u64, u32, u32, u32, ctypes.c_char_p],
            "aios_submit": [void_p, u64, u64, u32, u32, u32, void_p, u32, u64,
                            ctypes.POINTER(u64)],
            "aios_submit_trace": [void_p, u64, u64, u32, u32, u32, void_p, u32, u64,
                                  ctypes.POINTER(AIOSTraceContextC), ctypes.POINTER(u64)],
            "aios_wait": [void_p, u64, void_p, ctypes.POINTER(u32), int_p, int_p],
            "aios_query_status": [void_p, u64, int_p, int_p],
            "aios_read_stream": [void_p, u64, void_p, ctypes.POINTER(u32), ctypes.c_int,
                                 int_p, int_p],
            "aios_cancel": [void_p, u64],
            "aios_trace_get_enabled": [void_p, u64, int_p],
        }
        self.lib.aios_open.restype = void_p
        self.lib.aios_close.argtypes = [void_p]
        self.lib.aios_close.restype = None
        for name, argtypes in signatures.items():
            function = getattr(self.lib, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = ctypes.c_int

    @property
    def fd(self) -> int:
        # struct aios_handle is opaque; its first member is the /dev/aios fd.
        return ctypes.cast(self.h, ctypes.POINTER(ctypes.c_int))[0]

    def close(self) -> None:
        if self.h:
            self.lib.aios_close(self.h)
            self.h = None

    # --- agents ---

    def register(self, name: str) -> int:
        agent_id = ctypes.c_uint64(0)
        if self.lib.aios_register_agent(self.h, name.encode(), ctypes.byref(agent_id)) < 0:
            raise _errno_error("register agent")
        return agent_id.value

    def unregister(self, agent_id: int) -> None:
        if self.lib.aios_unregister_agent(self.h, agent_id) < 0:
            raise _errno_error("unregister agent")

    def setup_core(self, agent_id: int, core: int, pstr: str = "") -> None:
        if self.lib.aios_setup_core(self.h, agent_id, core, 0, 0, pstr.encode()) < 0:
            raise _errno_error(f"setup core {core}")

    def trace_enabled(self, agent_id: int) -> bool:
        function = getattr(self.lib, "aios_trace_get_enabled", None)
        enabled = ctypes.c_int(0)
        if function is None or function(self.h, agent_id, ctypes.byref(enabled)) < 0:
            return False
        return bool(enabled.value)

    # --- raw syscalls ---

    def submit(self, agent_id: int, type_id: int, payload: bytes, timeout_ms: int,
               span: TraceSpan | None = None, keepalive: object = None) -> int:
        buf = ctypes.create_string_buffer(payload, len(payload))
        syscall_id = ctypes.c_uint64(0)
        if span is not None and hasattr(self.lib, "aios_submit_trace"):
            context = span.context()
            query_id = int.from_bytes(span.trace_id[:8], "big") or 1
            ret = self.lib.aios_submit_trace(
                self.h, agent_id, query_id, type_id, 0, 1, ctypes.cast(buf, ctypes.c_void_p),
                len(payload), timeout_ms, ctypes.byref(context), ctypes.byref(syscall_id))
        else:
            ret = self.lib.aios_submit(
                self.h, agent_id, 0, type_id, 0, 1, ctypes.cast(buf, ctypes.c_void_p),
                len(payload), timeout_ms, ctypes.byref(syscall_id))
        if ret < 0:
            raise _errno_error("submit syscall")
        self._inflight[syscall_id.value] = (buf, keepalive)
        return syscall_id.value

    def wait(self, syscall_id: int) -> bytes:
        result = ctypes.create_string_buffer(RESULT_MAX)
        length = ctypes.c_uint32(RESULT_MAX)
        state, error = ctypes.c_int(0), ctypes.c_int(0)
        ret = self.lib.aios_wait(self.h, syscall_id, result, ctypes.byref(length),
                                 ctypes.byref(state), ctypes.byref(error))
        self._inflight.pop(syscall_id, None)
        if ret < 0:
            raise _errno_error("wait syscall")
        data = result.raw[:length.value]
        if error.value:
            message = data.split(b"\0", 1)[0].decode("utf-8", "replace")
            raise KernelError(-error.value if error.value < 0 else error.value,
                              message or os.strerror(abs(error.value)))
        return data

    def status(self, syscall_id: int) -> tuple[int, int]:
        state, error = ctypes.c_int(0), ctypes.c_int(0)
        if self.lib.aios_query_status(self.h, syscall_id, ctypes.byref(state),
                                      ctypes.byref(error)) < 0:
            raise _errno_error("query status")
        return state.value, error.value

    def read_stream(self, syscall_id: int, size: int = 4096) -> tuple[bytes, int]:
        buf = ctypes.create_string_buffer(size)
        length = ctypes.c_uint32(size)
        state, error = ctypes.c_int(0), ctypes.c_int(0)
        if self.lib.aios_read_stream(self.h, syscall_id, buf, ctypes.byref(length), 1,
                                     ctypes.byref(state), ctypes.byref(error)) < 0:
            raise _errno_error("read stream")
        return buf.raw[:length.value], state.value

    def cancel(self, syscall_id: int) -> None:
        if self.lib.aios_cancel(self.h, syscall_id) < 0:
            raise _errno_error("cancel syscall")

    def owner_action(self, agent_id: int, action: int, syscall_id: int,
                     reason: str = "") -> None:
        request = struct.pack("@HHIIQQ128s", ABI_VERSION, 160, action, 0, agent_id,
                              syscall_id, reason.encode()[:127])
        fcntl.ioctl(self.fd, _IOCTL_OWNER_ACTION, request)

    # --- services used by Agent ---

    def llm_stream(self, agent_id: int, messages: list[dict], model: str | None = None,
                   json_mode: bool = False, timeout_ms: int = 120_000,
                   span: TraceSpan | None = None) -> Generator[str, None, str]:
        """LLM syscall. Yields text chunks as the worker streams them and
        returns the complete text."""
        payload = json.dumps({
            "messages": messages, "tools": None,
            "format": "json" if json_mode else None, "model": model,
        }).encode()
        syscall_id = self.submit(agent_id, SYSCALL_LLM, payload, timeout_ms, span)
        try:
            while True:
                chunk, state = self.read_stream(syscall_id)
                if chunk:
                    yield chunk.decode("utf-8", "replace")
                if state in TERMINAL_STATES and not chunk:
                    break
        except GeneratorExit:  # Caller stopped reading: free the kernel slot.
            try:
                self.cancel(syscall_id)
            except KernelError:
                pass
            raise
        return self.wait(syscall_id).decode("utf-8", "replace").rstrip("\0")

    def tool(self, agent_id: int, name: str, args: dict, timeout_ms: int = 60_000,
             approve: Callable[[str, dict], bool] | None = None,
             span: TraceSpan | None = None) -> str:
        """Run a kernel-registered tool. Held side-effect tools go to approve();
        with no approver they wait for the owner to decide in AIOS Manager."""
        raw_name = name.encode()
        if not raw_name or len(raw_name) >= 64 or b"\0" in raw_name:
            raise ValueError("tool name must be 1-63 UTF-8 bytes without NUL")
        raw_args = json.dumps(args, separators=(",", ":")).encode()
        payload = _TOOL_HEADER.pack(ABI_VERSION, _TOOL_HEADER.size, 0, 0,
                                    raw_name.ljust(64, b"\0"), len(raw_args), 0) + raw_args
        syscall_id = self.submit(agent_id, SYSCALL_TOOL, payload, timeout_ms, span)
        if approve is not None:
            while True:
                state, _ = self.status(syscall_id)
                if state == STATE_HELD_APPROVAL:
                    allowed = approve(name, args)
                    action = OWNER_APPROVE_TOOL if allowed else OWNER_DENY_TOOL
                    self.owner_action(agent_id, action, syscall_id,
                                      "approved by agent owner" if allowed else "denied by agent owner")
                    break
                if state not in (STATE_CREATED, STATE_QUEUED):
                    break
                time.sleep(0.05)
        try:
            return self.wait(syscall_id).decode("utf-8", "replace").rstrip("\0")
        except KernelError as exc:
            if exc.errno == errno.EACCES:
                raise ToolDenied(name, str(exc)) from exc
            raise

    def list_tools(self) -> list[dict]:
        """Kernel tool registry, all pages."""
        size = _TOOL_PAGE_HEADER.size + 8 * _TOOL_ENTRY.size
        request = 0xC0000000 | (size << 16) | (ord("A") << 8) | _IOCTL_LIST_TOOLS_NR
        tools, offset = [], 0
        while True:
            buf = bytearray(_TOOL_PAGE_HEADER.pack(ABI_VERSION, size, 0, offset, 0, 0, 0)
                            + bytes(8 * _TOOL_ENTRY.size))
            fcntl.ioctl(self.fd, request, buf, True)
            _, _, _, _, total, count, _ = _TOOL_PAGE_HEADER.unpack_from(buf)
            for index in range(count):
                fields = _TOOL_ENTRY.unpack_from(buf, _TOOL_PAGE_HEADER.size + index * _TOOL_ENTRY.size)
                schema_text = fields[5].split(b"\0", 1)[0].decode("utf-8", "replace")
                try:
                    schema = json.loads(schema_text) if schema_text else {}
                except json.JSONDecodeError:
                    schema = {}
                tools.append({
                    "name": fields[3].split(b"\0", 1)[0].decode(),
                    "schema": schema if isinstance(schema, dict) else {},
                    "approval_required": bool(fields[2] & 1),
                })
            offset += count
            if not count or offset >= total:
                return tools

    def storage_write(self, agent_id: int, path: str, data: bytes,
                      timeout_ms: int = 10_000) -> None:
        self._storage(agent_id, STO_WRITE, path, data, timeout_ms)

    def storage_read(self, agent_id: int, path: str, timeout_ms: int = 10_000) -> bytes:
        return self._storage(agent_id, STO_RETRIEVE, path, b"", timeout_ms)

    def _storage(self, agent_id: int, op: int, path: str, data: bytes,
                 timeout_ms: int) -> bytes:
        raw_path = path.encode()
        if not raw_path or len(raw_path) >= 256 or b"\0" in raw_path:
            raise ValueError("storage path must be 1-255 UTF-8 bytes without NUL")
        source = ctypes.create_string_buffer(data, len(data)) if data else None
        pointer = ctypes.addressof(source) if source is not None else 0
        payload = _STORAGE_HEADER.pack(op, raw_path, pointer, len(data), 0, 0)
        return self.wait(self.submit(agent_id, SYSCALL_STORAGE, payload, timeout_ms,
                                     keepalive=source))
