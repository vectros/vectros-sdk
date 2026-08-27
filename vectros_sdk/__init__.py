"""Vectros SDK: AIOS-Agent SDK"""

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.core.models import Query, Response

__all__ = [
    "aios_kernel_url",
    "send_request",
    "Query",
    "Response",
]
