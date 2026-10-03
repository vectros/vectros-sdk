"""Kernel.client_tool control flow, with the ioctl layer replaced."""

import errno
import struct

import pytest

from vectros import KernelError, ToolDenied
from vectros import _kernel as k


class Recorder(k.Kernel):
    def __init__(self, states, wait_error=None):
        self.states = list(states)
        self.wait_error = wait_error
        self.log = []
        self._inflight = {}

    def submit(self, agent_id, type_id, payload, timeout_ms, span=None, keepalive=None):
        self.log.append(("submit", struct.unpack_from("@HHIQ64sII", payload)[2]))
        return 7

    def status(self, syscall_id):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return state, 0

    def owner_action(self, agent_id, action, syscall_id, reason=""):
        self.log.append(("owner", action))

    def _client_complete(self, syscall_id, data, error):
        self.log.append(("complete", data, error))

    def wait(self, syscall_id):
        if self.wait_error:
            raise self.wait_error
        return b"result"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(k.time, "sleep", lambda _: None)


def test_runs_when_leased_and_reports_result():
    kernel = Recorder([k.STATE_PROCESSING])
    assert kernel.client_tool(1, "add", {"a": 1}, lambda: "3") == "result"
    assert kernel.log == [("submit", k.TOOL_CALL_CLIENT), ("complete", b"3", 0)]


def test_approval_flag_and_approver_decision():
    kernel = Recorder([k.STATE_HELD_APPROVAL, k.STATE_PROCESSING])
    kernel.client_tool(1, "rm", {}, lambda: "ok", approval=True, approve=lambda n, a: True)
    assert kernel.log[0] == ("submit", k.TOOL_CALL_CLIENT | k.TOOL_CALL_APPROVAL)
    assert ("owner", k.OWNER_APPROVE_TOOL) in kernel.log
    assert kernel.log[-1] == ("complete", b"ok", 0)


def test_denied_call_never_runs():
    ran = []
    kernel = Recorder([k.STATE_HELD_APPROVAL, k.STATE_CANCELLED],
                      wait_error=KernelError(errno.ECANCELED, "cancelled"))
    with pytest.raises(ToolDenied):
        kernel.client_tool(1, "rm", {}, lambda: ran.append(1) or "x", approval=True,
                           approve=lambda n, a: False)
    assert ran == [] and ("owner", k.OWNER_DENY_TOOL) in kernel.log


def test_without_approver_waits_for_manager():
    kernel = Recorder([k.STATE_HELD_APPROVAL] * 3 + [k.STATE_PROCESSING])
    kernel.client_tool(1, "rm", {}, lambda: "ok", approval=True)
    assert not any(entry[0] == "owner" for entry in kernel.log)
    assert kernel.log[-1] == ("complete", b"ok", 0)


def test_tool_exception_is_reported_then_raised():
    kernel = Recorder([k.STATE_PROCESSING], wait_error=KernelError(errno.EIO, "x"))

    def boom():
        raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError, match="disk on fire"):
        kernel.client_tool(1, "t", {}, boom)
    assert kernel.log[-1] == ("complete", b"RuntimeError: disk on fire", -errno.EIO)


def test_timeout_surfaces_as_kernel_error():
    kernel = Recorder([k.STATE_FAILED], wait_error=KernelError(errno.ETIMEDOUT, "timed out"))
    with pytest.raises(KernelError):
        kernel.client_tool(1, "t", {}, lambda: "x")


def test_tool_payload_validates_name():
    with pytest.raises(ValueError):
        k._tool_payload("x" * 64, {}, 0)
    payload = k._tool_payload("add", {"a": 1}, k.TOOL_CALL_CLIENT)
    assert len(payload) == 88 + len(b'{"a":1}')
