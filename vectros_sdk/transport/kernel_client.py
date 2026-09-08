"""
Unified KernelClient that negotiates connection to aiosd.
Tries JSON-over-TCP on port 50051 first, falls back to UDS OperationClient.
"""
import logging
import uuid
from typing import Iterator, Optional, Dict, Any, List

from vectros_sdk.transport.grpc_transport import GrpcTransport
from vectros_sdk.operation import OperationClient, OperationInterfaceError
from vectros_sdk.domain import OperationKind, KIND_NAMES_UDS

logger = logging.getLogger(__name__)



class KernelClient:
    """
    Unified client for the AIOS kernel Operation Interface.

    Automatically negotiates the best available transport:
    - TCP JSON (port 50051) when aiosd is running with the JSON-over-TCP server
    - UDS (/tmp/aiosd-dev.sock) as a fallback for the development socket

    Example::

        from vectros_sdk import KernelClient
        client = KernelClient()
        print("mode:", client.mode)   # "tcp" or "uds"
    """

    def __init__(
        self,
        grpc_address: str = "localhost:50051",
        uds_path: str = "/tmp/aiosd-dev.sock",
        timeout: float = 5.0,
    ):
        self._mode = "none"
        self._tcp_client: Optional[GrpcTransport] = None
        self._uds_client: Optional[OperationClient] = None

        # Try TCP JSON transport first
        try:
            client = GrpcTransport(address=grpc_address, timeout=timeout)
            client.describe_capabilities()  # probe
            self._tcp_client = client
            self._mode = "tcp"
            logger.info("KernelClient: using TCP JSON transport on %s", grpc_address)
        except Exception as e:
            logger.warning("TCP JSON transport unavailable (%s). Falling back to UDS.", e)
            self._tcp_client = None
            try:
                self._uds_client = OperationClient(uds_path, timeout_seconds=timeout)
                self._mode = "uds"
                logger.info("KernelClient: using UDS transport at %s", uds_path)
            except Exception as e2:
                logger.error("UDS transport also unavailable: %s", e2)

    @property
    def mode(self) -> str:
        """Returns 'tcp', 'uds', or 'none'."""
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
        **kwargs,
    ) -> Dict[str, Any]:
        """Submit an Operation to the kernel. Returns the OperationView dict."""
        if self._mode == "tcp":
            return self._tcp_client.submit(
                agent_id, instance_id, run_id, operation_id, kind_int, payload
            )
        elif self._mode == "uds":
            kind_str = KIND_NAMES_UDS.get(kind_int, "infer")
            view = self._uds_client.submit(
                agent_id=agent_id,
                instance_id=instance_id,
                run_id=run_id,
                operation_id=operation_id,
                kind=kind_str,
                side_effect="read_only",
            )
            return {"operation_id": view.operation_id, "state": view.state}
        raise RuntimeError("No transport available")

    def get(self, operation_id: str) -> Dict[str, Any]:
        """Get current state of an Operation."""
        if self._mode == "tcp":
            return self._tcp_client.get(operation_id)
        elif self._mode == "uds":
            view = self._uds_client.get(operation_id)
            return {"operation_id": view.operation_id, "state": view.state}
        raise RuntimeError("No transport available")

    def cancel(self, operation_id: str) -> Dict[str, Any]:
        """Cancel a running Operation."""
        if self._mode == "tcp":
            return self._tcp_client.cancel(operation_id)
        elif self._mode == "uds":
            view = self._uds_client.cancel(operation_id)
            return {"operation_id": view.operation_id, "state": view.state}
        raise RuntimeError("No transport available")

    def watch(
        self, agent_id: str, run_id: str, cursor: str = "0"
    ) -> Iterator[Dict[str, Any]]:
        """Stream Operation events for an agent/run pair."""
        if self._mode == "tcp":
            yield from self._tcp_client.watch(agent_id, run_id, cursor)
        elif self._mode == "uds":
            # UDS watch returns a list of OperationEvent objects
            cursor_int = int(cursor) if cursor.isdigit() else 0
            events = self._uds_client.watch(agent_id, run_id, cursor_int)
            for e in events:
                yield {"event_type": e.event_type, "cursor": str(e.cursor)}
        else:
            raise RuntimeError("No transport available")

    def describe_capabilities(self) -> Dict[str, Any]:
        """Return the kernel's declared capability list."""
        if self._mode == "tcp":
            return self._tcp_client.describe_capabilities()
        raise NotImplementedError("describe_capabilities only available on TCP transport")

    def get_artifact(self, artifact_id: str) -> bytes:
        """Fetch an artifact's content bytes."""
        if self._mode == "tcp":
            return self._tcp_client.get_artifact(artifact_id)
        raise NotImplementedError("get_artifact only available on TCP transport")

    def close(self):
        if self._tcp_client:
            self._tcp_client.close()
