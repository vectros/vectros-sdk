"""
JSON-over-TCP transport client for the AIOS kernel OperationInterface (port 50051).

The aiosd daemon exposes a JSON-over-TCP server matching the proto field names.
Each request is a single JSON line; the server responds with a JSON line per reply.
Watch responses stream multiple JSON lines until the connection is closed.
"""
import json
import socket
import threading
from typing import Iterator, Optional, Dict, Any, List

DEFAULT_TCP_ADDR = "localhost"
DEFAULT_TCP_PORT = 50051


class GrpcTransport:
    """JSON-over-TCP client matching the aiosd grpc_server.rs protocol."""

    def __init__(self, address: str = f"{DEFAULT_TCP_ADDR}:{DEFAULT_TCP_PORT}", timeout: float = 10.0):
        host, port_str = address.rsplit(":", 1)
        self._host = host
        self._port = int(port_str)
        self._timeout = timeout
        # Probe connection
        self._probe()

    def _probe(self):
        """Check the server is reachable by attempting a TCP connect."""
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        s.close()

    def _send(self, payload: dict) -> dict:
        """Send one JSON line request and receive one JSON line response."""
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        try:
            s.sendall((json.dumps(payload) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            line = buf.split(b"\n")[0].decode()
            return json.loads(line)
        finally:
            s.close()

    def submit(
        self,
        agent_id: str,
        instance_id: str,
        run_id: str,
        operation_id: str,
        kind_int: int,
        payload: str = "{}",
        **kwargs,
    ) -> dict:
        req = {
            "submit": {
                "agent_id": agent_id,
                "agent_instance_id": instance_id,
                "run_id": run_id,
                "operation_id": operation_id,
                "kind": kind_int,
                "payload": payload,  # FIX #3: forward real prompt to daemon
            }
        }
        return self._send(req)

    def get(self, operation_id: str) -> Dict[str, Any]:
        req = {"get": {"operation_id": operation_id}}
        return self._send(req)

    def cancel(self, operation_id: str) -> Dict[str, Any]:
        req = {"cancel": {"operation_id": operation_id, "reason": None}}
        return self._send(req)

    def watch(self, agent_id: str, run_id: str, cursor: str = "0") -> Iterator[Dict[str, Any]]:
        """Stream watch events from the server until connection closes."""
        req = {"watch": {"agent_id": agent_id, "run_id": run_id, "cursor": cursor}}
        s = socket.create_connection((self._host, self._port), timeout=60.0)
        try:
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while True:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            event = json.loads(line.decode())
                            # Normalise the state field as event_type for consumers
                            event["event_type"] = event.get("state", "Unknown")
                            yield event
                        except json.JSONDecodeError:
                            pass
        finally:
            s.close()

    def describe_capabilities(self) -> Dict[str, Any]:
        req = {"_describe": {}}
        return self._send(req)

    def get_artifact(self, artifact_id: str) -> bytes:
        import base64
        req = {
            "get_artifact": {
                "artifact_id": artifact_id,
            }
        }
        resp = self._send(req)
        if "chunk" in resp:
            return base64.b64decode(resp["chunk"])
        return b""

    def close(self):
        pass  # Connections are per-request; nothing to close
