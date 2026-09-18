"""
Unified KernelClient for the AIOS kernel's development Operation Interface
(submit/get/cancel/watch a long-running Operation).

This is a **different** capability from the Query API (`llm_chat`,
`create_memory`, ...) that `vectros_sdk.client.send_request` and
`vectros_sdk.client.real_kernel` serve: the real `KernelRequest`/`KernelValue`
protocol (`aios-sdk`'s `ExecutionClient`) has no Operation-lifecycle
observation variant at all — no submit/get/cancel/watch — so there is
nothing here for `ExecutionProtocolClient` to consolidate onto yet. This
class exists to serve that one gap until a real, authorized lifecycle API is
built server-side.

SDK.6 (2026-09-17): this class used to also try a gRPC transport
(`vectros_sdk.transport.grpc_transport.GrpcTransport`) against
`aiosd::grpc_server` on port 50051 *first*, silently preferring it over UDS
on every construction. That server is Path B: no authorization, no
accounting, no auditing, zero test coverage (see CLAUDE.md,
docs/architecture.md §1-§2) — "do not treat the second path as production
code, and do not add to it." A client that defaults to it is already doing
exactly that, so this class no longer imports or tries it at all; it now
only ever speaks the small, deliberately-scoped UDS development protocol
(`vectros_sdk.operation.OperationClient`), which is real but explicitly not
an authority channel (no tool invocation, no approval, no terminal-success
commands — see that module's docstring).

`GrpcTransport` itself, its vendored proto stubs, and the standalone demo
scripts that call it directly (`test_grpc.py`, `test_tools.py`) are
untouched: plan.md Principle 3 ("delete only after replacement") ties their
removal to something else, not to this consolidation. This module simply
stops being one more thing that reaches for Path B by default.

TERM.6/TERM.12/SEC.15 (2026-09-17): `host_agent.py` -- the unsandboxed
prototype this comment used to list here alongside the demo scripts above --
has been deleted now that the real AIOS Terminal (`vectros_sdk.terminal`,
plan.md Phase 12a) ships a real proposal+approval path. It was never one of
the demo scripts kept for a reason; it was kept only until this replacement
existed.
"""

import logging
from typing import Any, Dict, Iterator, Optional

from vectros_sdk.domain import KIND_NAMES_UDS
from vectros_sdk.operation import OperationClient

logger = logging.getLogger(__name__)

DEFAULT_UDS_PATH = "/tmp/aiosd-dev.sock"


class KernelClient:
    """Thin, typed wrapper over the UDS development Operation Interface.

    Example::

        from vectros_sdk import KernelClient
        client = KernelClient()
        print("mode:", client.mode)   # "uds"
    """

    def __init__(
        self,
        uds_path: str = DEFAULT_UDS_PATH,
        timeout: float = 5.0,
    ):
        # OperationClient's constructor only validates the path string; it
        # does not connect. Connection failures surface from the first real
        # call, same as every other transport in this SDK — never swallowed
        # into a construction-time "mode" guess.
        self._client = OperationClient(uds_path, timeout_seconds=timeout)
        self._mode = "uds"

    @property
    def mode(self) -> str:
        """Returns "uds" — the only transport this client speaks."""
        return self._mode

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def submit(
        self,
        agent_id: str,
        instance_id: str,
        run_id: str,
        operation_id: str,
        kind_int: int,
        payload: str = "{}",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit an Operation to the kernel. Returns the OperationView dict."""
        kind_str = KIND_NAMES_UDS.get(kind_int, "infer")
        view = self._client.submit(
            agent_id=agent_id,
            instance_id=instance_id,
            run_id=run_id,
            operation_id=operation_id,
            kind=kind_str,
            side_effect="read_only",
        )
        return {"operation_id": view.operation_id, "state": view.state}

    def get(self, operation_id: str) -> Dict[str, Any]:
        """Get current state of an Operation."""
        view = self._client.get(operation_id)
        return {"operation_id": view.operation_id, "state": view.state}

    def cancel(self, operation_id: str) -> Dict[str, Any]:
        """Cancel a running Operation."""
        view = self._client.cancel(operation_id)
        return {"operation_id": view.operation_id, "state": view.state}

    def watch(
        self, agent_id: str, run_id: str, cursor: str = "0"
    ) -> Iterator[Dict[str, Any]]:
        """Stream Operation events for an agent/run pair."""
        cursor_int = int(cursor) if cursor.isdigit() else 0
        events = self._client.watch(agent_id, run_id, cursor_int)
        for event in events:
            yield {"event_type": event.event_type, "cursor": str(event.cursor)}

    def describe_capabilities(self) -> Dict[str, Any]:
        """Return the kernel's declared capability list.

        Not implemented: the UDS development protocol has no capability
        descriptor. This was previously served by trying gRPC/Path-B first;
        this class no longer does that (see module docstring) — not
        something to fake a response for.
        """
        raise NotImplementedError(
            "describe_capabilities has no UDS equivalent (it was gRPC/Path-B-only)"
        )

    def get_artifact(self, artifact_id: str) -> bytes:
        """Fetch an artifact's content bytes.

        Not implemented: the UDS development protocol has no artifact
        fetch. This was previously served by trying gRPC/Path-B first; this
        class no longer does that (see module docstring) — not something to
        fake a response for.
        """
        raise NotImplementedError(
            "get_artifact has no UDS equivalent (it was gRPC/Path-B-only)"
        )

    def close(self) -> None:
        """No persistent connection to close: every call opens and closes
        its own socket (see `OperationClient._request`)."""
