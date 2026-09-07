"""Client communication, orchestrator, and configuration module"""

from vectros_sdk.client.client import AIOSClient, CerebrumClient
from vectros_sdk.client.config import DEFAULT_AIOS_KERNEL_URL, aios_kernel_url, get_kernel_url, set_kernel_url
from vectros_sdk.client.send_request import AIOSKernelError, send_request

__all__ = [
    "aios_kernel_url",
    "get_kernel_url",
    "set_kernel_url",
    "DEFAULT_AIOS_KERNEL_URL",
    "send_request",
    "AIOSKernelError",
    "AIOSClient",
    "CerebrumClient",
]
