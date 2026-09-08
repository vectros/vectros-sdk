"""
Canonical OperationKind constants matching proto/aios/v1/operation.proto.

All SDK transports (TCP JSON and UDS) translate through these constants,
eliminating the bug where adding a new kind required updates in 3 places.
"""
from __future__ import annotations
from enum import IntEnum

__all__ = ["OperationKind", "KIND_NAMES_UDS"]


class OperationKind(IntEnum):
    """Mirrors OperationKind enum in operation.proto."""
    UNSPECIFIED = 0
    INFER = 1
    INVOKE_TOOL = 2
    READ_MEMORY = 3
    WRITE_MEMORY = 4
    IMPORT_ARTIFACT = 5
    SHARE_ARTIFACT = 6


# Map int kind → string used by the UDS OperationClient
KIND_NAMES_UDS: dict[int, str] = {
    OperationKind.INFER: "infer",
    OperationKind.INVOKE_TOOL: "invoke_tool",
    OperationKind.READ_MEMORY: "read_memory",
    OperationKind.WRITE_MEMORY: "write_memory",
    OperationKind.IMPORT_ARTIFACT: "import_artifact",
    OperationKind.SHARE_ARTIFACT: "share_artifact",
}
