"""
Fake, in-process AIOS Kernel HTTP server package -- an offline demo
stand-in for the example agents, not the real kernel. See
`mock_kernel_server`'s own module docstring.
"""

from vectros_sdk.examples.server.mock_kernel_server import (
    AIOSKernelHTTPHandler,
    FakeAIOSKernelServer,
    start_kernel_server,
)

__all__ = [
    "AIOSKernelHTTPHandler",
    "FakeAIOSKernelServer",
    "start_kernel_server",
]
