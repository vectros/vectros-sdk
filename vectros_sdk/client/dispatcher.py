"""Alias module for send_request dispatcher"""

from vectros_sdk.client.send_request import AIOSKernelError, send_request

__all__ = ["send_request", "AIOSKernelError"]
