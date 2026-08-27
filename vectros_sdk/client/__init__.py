"""Client communication and configuration module"""

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request

__all__ = ["aios_kernel_url", "send_request"]
