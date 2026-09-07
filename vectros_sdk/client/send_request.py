"""
HTTP Request Dispatcher module for communicating with the AIOS Kernel.

Provides central `send_request` dispatcher and custom `AIOSKernelError` exception.
"""

import json
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel
import requests

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.core.models import Query


class AIOSKernelError(Exception):
    """
    Exception raised when communication with the AIOS kernel fails or kernel returns non-2xx status.

    Attributes:
        message (str): Explanatory error message.
        status_code (Optional[int]): HTTP status code from kernel response if available.
        response_data (Optional[Any]): Raw or parsed JSON error response body from kernel.

    Example:
        >>> try:
        ...     raise AIOSKernelError("Server error", status_code=500)
        ... except AIOSKernelError as exc:
        ...     print(exc.status_code)
        500
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


def send_request(
    query: Union[Query, BaseModel, Dict[str, Any]],
    base_url: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Central dispatcher function to communicate with AIOS kernel via HTTP POST requests.

    Serializes the given Query model or dictionary payload, ensures the endpoint URL
    targets `/query`, sends the HTTP POST request, validates the response, and returns
    the parsed JSON payload.

    Args:
        query (Union[Query, BaseModel, Dict[str, Any]]): Query object or dictionary payload
            containing the operation parameters.
        base_url (Optional[str], optional): AIOS kernel base endpoint URL. If None, defaults
            to the configured `aios_kernel_url`. Defaults to None.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.

    Returns:
        Dict[str, Any]: Parsed JSON response dictionary returned by the AIOS kernel.

    Raises:
        ValueError: If query argument is not an instance of Query, BaseModel, or dict.
        AIOSKernelError: If connection fails, request times out, or kernel returns non-2xx HTTP status.

    Example:
        >>> from vectros_sdk.core.models import Query
        >>> from vectros_sdk.client.send_request import send_request
        >>> q = Query(query_class="llm", agent_name="demo_bot")
        >>> # Dispatches payload to kernel:
        >>> # resp = send_request(q, base_url="http://localhost:8000")
    """
    url = (base_url or aios_kernel_url).rstrip("/")
    if not url.endswith("/query"):
        endpoint = f"{url}/query"
    else:
        endpoint = url

    if isinstance(query, BaseModel):
        payload = query.model_dump()
    elif isinstance(query, dict):
        payload = query
    else:
        raise ValueError(f"Expected Query, BaseModel, or dict, got {type(query).__name__}")

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise AIOSKernelError(f"Request to AIOS kernel timed out: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise AIOSKernelError(f"Failed to connect to AIOS kernel at {endpoint}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise AIOSKernelError(f"Error communicating with AIOS kernel: {exc}") from exc

    if not response.ok:
        error_msg = f"AIOS kernel returned status code {response.status_code}: {response.text}"
        try:
            res_json = response.json()
        except Exception:
            res_json = None
        raise AIOSKernelError(error_msg, status_code=response.status_code, response_data=res_json)

    try:
        return response.json()
    except Exception as exc:
        raise AIOSKernelError(f"Failed to parse JSON response from AIOS kernel: {exc}") from exc
