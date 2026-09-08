"""
gRPC transport client for the AIOS kernel OperationInterface (port 50051).
"""
import grpc
from typing import Iterator, Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vectros_sdk.proto.aios.v1 import operation_pb2, operation_pb2_grpc

DEFAULT_GRPC_ADDR = "localhost:50051"

class GrpcTransport:
    def __init__(self, address: str = DEFAULT_GRPC_ADDR, timeout: float = 10.0, token: str = "super-secret-token"):
        self._address = address
        self._timeout = timeout
        self._token = token
        # Real gRPC requires metadata for interceptors
        self._metadata = (("authorization", f"Bearer {token}"),)
        
        # Connect to server
        self._channel = grpc.insecure_channel(address)
        self._stub = operation_pb2_grpc.OperationInterfaceStub(self._channel)

        # Probe connection via describe capabilities
        self.describe_capabilities()

    def submit(
        self,
        agent_id: str,
        instance_id: str,
        run_id: str,
        operation_id: str,
        kind_int: int,
        payload: Optional[bytes] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        req = operation_pb2.OperationEnvelope(
            agent_id=agent_id,
            agent_instance_id=instance_id,
            run_id=run_id,
            operation_id=operation_id,
            kind=kind_int,
            idempotency_key=idempotency_key or "",
            payload=payload or b"",
        )
        res = self._stub.Submit(req, timeout=self._timeout, metadata=self._metadata)
        return {
            "operation_id": res.operation_id,
            "state": res.state,
            "cursor": res.cursor,
            "local_outcome": res.local_outcome,
            "external_outcome": res.external_outcome,
        }

    def get(self, operation_id: str) -> Dict[str, Any]:
        req = operation_pb2.OperationIdRequest(operation_id=operation_id)
        res = self._stub.Get(req, timeout=self._timeout, metadata=self._metadata)
        return {
            "operation_id": res.operation_id,
            "state": res.state,
            "cursor": res.cursor,
            "local_outcome": res.local_outcome,
            "external_outcome": res.external_outcome,
        }

    def cancel(self, operation_id: str) -> Dict[str, Any]:
        req = operation_pb2.CancelRequest(operation_id=operation_id, reason="Requested via SDK")
        res = self._stub.Cancel(req, timeout=self._timeout, metadata=self._metadata)
        return {
            "operation_id": res.operation_id,
            "state": res.state,
            "cursor": res.cursor,
            "local_outcome": res.local_outcome,
            "external_outcome": res.external_outcome,
        }

    def watch(self, agent_id: str, run_id: str, cursor: str = "") -> Iterator[Dict[str, Any]]:
        req = operation_pb2.WatchRequest(
            agent_id=agent_id,
            run_id=run_id,
            cursor=cursor
        )
        for res in self._stub.Watch(req, metadata=self._metadata):
            yield {
                "operation_id": res.operation_id,
                "state": res.state,
                "cursor": res.cursor,
                "local_outcome": res.local_outcome,
                "external_outcome": res.external_outcome,
            }

    def describe_capabilities(self) -> list:
        req = operation_pb2.Empty()
        res = self._stub.DescribeCapabilities(req, timeout=self._timeout, metadata=self._metadata)
        return list(res.capabilities)

    def close(self):
        self._channel.close()
