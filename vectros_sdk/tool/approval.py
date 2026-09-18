"""
SDK.7 -- human-in-the-loop tool-call approval, client side. Wraps a single
tool call so a caller-supplied callback decides whether to approve or deny
a call the real kernel server has deferred (CTL.10's Tier 3
`ListPendingApprovals`/`ApproveTool`/`DenyTool`), then drives that decision
through the real control socket and retries.

Real, no mocks: this is a thin client-side orchestration over already-real,
already-tested primitives -- `ToolClient.call` (which raises
`ExecutionProtocolError(code="pending_approval")` when the server defers a
call) and `ControlClient.list_pending_approvals`/`approve_tool`/`deny_tool`
(CTL.10). It never fabricates an approval decision or a tool result: if the
server was never configured with `AIOS_REQUIRE_TOOL_APPROVAL=1`, the first
call just succeeds immediately and `approval_callback` is never invoked --
there is nothing to approve.

Deliberately refuses rather than guesses when more than one pending
approval matches: this single-tenant server (ARCH.30) has exactly one
Agent, so a real caller's own retry should find exactly its own one
deferred call. Finding zero or several is a real, honest "cannot safely
identify which one this retry deferred", never a silent pick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

from vectros_sdk.transport.execution_protocol import ExecutionProtocolError

if TYPE_CHECKING:
    from vectros_sdk.client.client import AIOSClient

ApprovalCallback = Callable[[Dict[str, Any]], bool]


class ApprovalRefused(Exception):
    """Raised when the approval callback declines a deferred tool call, or
    when the pending approval this retry deferred cannot be safely
    identified."""


def call_tool_with_approval(
    client: "AIOSClient",
    tool_calls: List[Dict[str, Any]],
    approval_callback: ApprovalCallback,
) -> Any:
    """Calls ``client.tool.call(tool_calls)``; if the real kernel defers it
    pending operator review, finds the matching pending approval, asks
    ``approval_callback`` to decide, drives that decision through the real
    control socket, and retries once.

    ``approval_callback`` receives the real ``PendingApproval`` dict (the
    exact ``agent_id``/``run_id``/``tool``/``arguments``/``requested_at_ms``
    the server recorded) and returns ``True`` to approve, ``False`` to deny.

    Raises ``ApprovalRefused`` if the callback declines, or if the deferred
    call cannot be safely identified. Raises whatever ``client.tool.call``
    would raise for any other rejection -- a non-approval failure is never
    treated as an approval question.
    """
    try:
        return client.tool.call(tool_calls)
    except ExecutionProtocolError as error:
        if error.code != "pending_approval":
            raise

    pending = client.control.list_pending_approvals()
    if client.agent_name:
        expected_agent = f"agent_{client.agent_name}"
        pending = [entry for entry in pending if entry["agent_id"] == expected_agent]
    if len(pending) != 1:
        raise ApprovalRefused(
            f"expected exactly one pending approval for this retry, found {len(pending)} "
            "-- cannot safely identify which one this call deferred"
        )
    entry = pending[0]

    if approval_callback(entry):
        client.control.approve_tool(entry["approval_id"])
    else:
        client.control.deny_tool(entry["approval_id"])
        raise ApprovalRefused(f"approval callback declined {entry['approval_id']}")

    return client.tool.call(tool_calls)
